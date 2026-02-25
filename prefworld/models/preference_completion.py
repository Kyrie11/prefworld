from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.models.gaussian import DiagGaussian, NaturalDiagGaussian


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to (-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    if dim is None:
        return (x * mask_f).sum() / (mask_f.sum() + eps)
    return (x * mask_f).sum(dim=dim) / (mask_f.sum(dim=dim) + eps)


@dataclass
class PreferencePosterior:
    """Posterior belief q(z) for each agent."""

    q: DiagGaussian               # [B,N,Dz]
    nat: NaturalDiagGaussian      # [B,N,Dz]
    alpha: torch.Tensor          # [B,N,T] evidence reliability gates


@dataclass
class PreferenceCompletionOutput:
    """Outputs for preference completion training."""

    # posteriors
    post_full: PreferencePosterior
    post_ctx: PreferencePosterior

    # masks
    ctx_mask: torch.Tensor       # [B,N,T]
    query_mask: torch.Tensor     # [B,N,T]

    # losses (scalars)
    loss_total: torch.Tensor
    loss_query_nll: torch.Tensor
    loss_kl_full_ctx: torch.Tensor
    loss_kl_ctx_prior: torch.Tensor
    loss_prec_reg: torch.Tensor


class EvidenceEncoder(nn.Module):
    """Encode each evidence token (X_{i,t}, τ_{i,t}) into natural-parameter increments.

    This follows the paper's neural evidence accumulation scheme:
      (Δη, ΔΛ, α) = f(e_{i,t})
      η = η0 + Σ α Δη,   Λ = Λ0 + Σ α ΔΛ

    Here we implement a diagonal-precision version for stability.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        in_dim = int(x_dim + tau_dim + ctx_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_eta = nn.Linear(hidden_dim, z_dim)
        self.to_L = nn.Linear(hidden_dim, z_dim)
        self.to_alpha = nn.Linear(hidden_dim, 1)

        # init small updates so posterior starts near prior
        nn.init.zeros_(self.to_eta.weight)
        nn.init.zeros_(self.to_eta.bias)
        nn.init.zeros_(self.to_L.weight)
        nn.init.constant_(self.to_L.bias, -3.0)  # softplus(-3) ~ 0.05

    def forward(
        self,
        x: torch.Tensor,         # [B,N,T,Dx]
        tau: torch.Tensor,       # [B,N,T,Dt]
        ctx: torch.Tensor,       # [B,N,T,Dc]
        mask: torch.Tensor,      # [B,N,T] (bool/float)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, tau, ctx], dim=-1))
        delta_eta = self.to_eta(h)                       # [B,N,T,Dz]
        delta_L = F.softplus(self.to_L(h))              # [B,N,T,Dz] >= 0
        alpha = torch.sigmoid(self.to_alpha(h)).squeeze(-1)  # [B,N,T]

        m = mask > 0.5
        delta_eta = delta_eta * m.unsqueeze(-1).to(dtype=delta_eta.dtype)
        delta_L = delta_L * m.unsqueeze(-1).to(dtype=delta_L.dtype)
        alpha = alpha * m.to(dtype=alpha.dtype)
        return delta_eta, delta_L, alpha


class ActionDecoder(nn.Module):
    """Template-conditioned action likelihood p(X | z, τ).

    We decode each token x_t (e.g., Δpose) as a diagonal Gaussian whose mean
    is modulated by z through FiLM on τ.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        pred_logstd: bool = True,
        min_logstd: float = -5.0,
        max_logstd: float = 2.0,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.tau_dim = int(tau_dim)
        self.z_dim = int(z_dim)
        self.pred_logstd = bool(pred_logstd)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)

        self.tau_proj = nn.Sequential(
            nn.Linear(tau_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.z_to_film = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out_mu = nn.Linear(hidden_dim, x_dim)
        if pred_logstd:
            self.out_logstd = nn.Linear(hidden_dim, x_dim)
        else:
            self.logstd_param = nn.Parameter(torch.zeros(x_dim))

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
        z:   [B,N,Dz]
        tau: [B,N,T,Dt]
        Returns:
          mu_x:     [B,N,T,Dx]
          logstd_x: [B,N,T,Dx]
        """
        B, N, T, _ = tau.shape
        h_tau = self.tau_proj(tau)
        film = self.z_to_film(z)  # [B,N,2H]
        gamma, beta = film.chunk(2, dim=-1)
        h = h_tau * (1.0 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        h = self.dec(h)
        mu = self.out_mu(h)
        if self.pred_logstd:
            logstd = torch.clamp(self.out_logstd(h), min=self.min_logstd, max=self.max_logstd)
        else:
            logstd = torch.clamp(self.logstd_param.view(1, 1, 1, -1).expand_as(mu), min=self.min_logstd, max=self.max_logstd)
        return mu, logstd


class PreferenceCompletion(nn.Module):
    """Preference completion via neural evidence accumulation (paper-aligned).

    This module implements:
      q(z | S) where S is any subset of evidence tokens (X_{i,t}, τ_{i,t})
    and the conditional variational objective used in preference completion:

      L = -E_{q(z|C∪Q)}[ Σ_{t∈Q} log p(X_t | z, τ_t) ]
          + λ_kl KL(q(z|C∪Q) || q(z|C))
          + λ_prior KL(q(z|C) || p(z))
          + λ_Λ Σ_t ||ΔΛ_t||_F^2

    We use a diagonal precision Λ for stability.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        prior_logvar: float = 0.0,
        lambda_u: float = 1.0,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.tau_dim = int(tau_dim)
        self.ctx_dim = int(ctx_dim)
        self.z_dim = int(z_dim)

        self.evidence = EvidenceEncoder(x_dim=x_dim, tau_dim=tau_dim, ctx_dim=ctx_dim, z_dim=z_dim, hidden_dim=hidden_dim)
        self.decoder = ActionDecoder(x_dim=x_dim, tau_dim=tau_dim, z_dim=z_dim, hidden_dim=hidden_dim)

        # fixed diagonal Gaussian prior p(z) = N(0, exp(prior_logvar) I)
        self.register_buffer("prior_mean", torch.zeros(z_dim))
        self.register_buffer("prior_logvar", torch.full((z_dim,), float(prior_logvar)))

        # not used here, but kept for compatibility with the paper notation
        self.lambda_u = float(lambda_u)

    def prior(self, batch_shape: Tuple[int, ...], device=None, dtype=None) -> DiagGaussian:
        mean = self.prior_mean.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        logvar = self.prior_logvar.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        return DiagGaussian(mean=mean, logvar=logvar)

    def _posterior_from_mask(
        self,
        *,
        delta_eta: torch.Tensor,     # [B,N,T,Dz]
        delta_L: torch.Tensor,       # [B,N,T,Dz]
        alpha: torch.Tensor,         # [B,N,T]
        mask: torch.Tensor,          # [B,N,T]
    ) -> PreferencePosterior:
        B, N, T, Dz = delta_eta.shape
        m = mask > 0.5
        a = alpha.unsqueeze(-1)

        # prior in natural form (standard normal or fixed diag)
        prior = self.prior((B, N), device=delta_eta.device, dtype=delta_eta.dtype)
        nat0 = NaturalDiagGaussian.from_moment(prior)

        # pool natural increments over selected tokens
        pooled_eta = nat0.eta + (a * delta_eta * m.unsqueeze(-1).to(dtype=delta_eta.dtype)).sum(dim=2)
        pooled_L = nat0.Lambda + (a * delta_L * m.unsqueeze(-1).to(dtype=delta_L.dtype)).sum(dim=2)
        nat = NaturalDiagGaussian(eta=pooled_eta, Lambda=pooled_L.clamp_min(1e-6))
        q = nat.to_moment()
        return PreferencePosterior(q=q, nat=nat, alpha=alpha)

    @staticmethod
    def _ensure_nonempty(mask_all: torch.Tensor, mask_ctx: torch.Tensor) -> torch.Tensor:
        """Guarantee each (B,N) has at least one context token if it has any valid token."""
        B, N, T = mask_all.shape
        valid = mask_all.sum(dim=-1) > 0
        ctx = mask_ctx.sum(dim=-1) > 0
        need = valid & (~ctx)
        if need.any():
            # pick the first valid token as context
            idx = mask_all.float().argmax(dim=-1)  # [B,N]
            b, n = need.nonzero(as_tuple=True)
            mask_ctx = mask_ctx.clone()
            mask_ctx[b, n, idx[b, n]] = 1.0
        return mask_ctx

    def split_context_query(
        self,
        valid_mask: torch.Tensor,  # [B,N,T]
        *,
        mode: str = "random",      # "random" or "prefix"
        query_ratio: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split valid_mask into (ctx_mask, query_mask)."""
        m = valid_mask > 0.5
        B, N, T = m.shape
        if mode == "prefix":
            # choose a cutoff per (B,N) among valid tokens
            # If length < 2, no query.
            lengths = m.sum(dim=-1).clamp(min=0).to(torch.int64)  # [B,N]
            # cutoff in [1, len-1]
            cutoff = torch.zeros((B, N), device=m.device, dtype=torch.int64)
            # sample only where length >= 2
            ok = lengths >= 2
            if ok.any():
                # randint high is exclusive
                cutoff[ok] = torch.randint(low=1, high=(lengths[ok]).min().item() + 1, size=(int(ok.sum().item()),), device=m.device)
                # The above uses a shared upper bound; refine per element using modulo trick
                # to avoid python loops.
                cutoff[ok] = 1 + (torch.randint(0, 10_000, (int(ok.sum().item()),), device=m.device) % (lengths[ok] - 1))
            t_idx = torch.arange(T, device=m.device).view(1, 1, T)
            # prefix over time index, but only for valid tokens
            ctx_mask = (t_idx < cutoff.unsqueeze(-1)) & m
            query_mask = m & (~ctx_mask)
        else:
            # random subset as query
            rnd = torch.rand((B, N, T), device=m.device)
            query_mask = (rnd < float(query_ratio)) & m
            ctx_mask = m & (~query_mask)
            ctx_mask = self._ensure_nonempty(m.to(dtype=torch.float32), ctx_mask.to(dtype=torch.float32)) > 0.5
            query_mask = m & (~ctx_mask)

        # ensure query non-empty when possible (otherwise loss_query=0)
        # If there are >=2 valid tokens and query is empty, move the last valid token to query.
        lengths = m.sum(dim=-1)
        q_counts = query_mask.sum(dim=-1)
        need_q = (lengths >= 2) & (q_counts == 0)
        if need_q.any():
            last_valid = (m.to(torch.int64).cumsum(dim=-1) == lengths.unsqueeze(-1).to(torch.int64)).to(torch.int64).argmax(dim=-1)
            b, n = need_q.nonzero(as_tuple=True)
            t = last_valid[b, n]
            query_mask = query_mask.clone()
            ctx_mask = ctx_mask.clone()
            query_mask[b, n, t] = True
            ctx_mask[b, n, t] = False

        return ctx_mask.to(dtype=torch.float32), query_mask.to(dtype=torch.float32)

    def forward(
        self,
        *,
        x: torch.Tensor,           # [B,N,T,Dx]
        tau: torch.Tensor,         # [B,N,T,Dt]
        ctx: torch.Tensor,         # [B,N,T,Dc]
        mask: torch.Tensor,        # [B,N,T]
        split_mode: str = "random",
        query_ratio: float = 0.3,
        lambda_kl: float = 1.0,
        lambda_prior: float = 1.0,
        lambda_prec: float = 1e-3,
        n_z_samples: int = 1,
        free_bits: float = 0.0,
    ) -> PreferenceCompletionOutput:
        """Run preference completion objective."""
        B, N, T, _ = x.shape
        valid_mask = mask > 0.5
        ctx_mask, query_mask = self.split_context_query(valid_mask.to(dtype=torch.float32), mode=split_mode, query_ratio=query_ratio)

        delta_eta, delta_L, alpha = self.evidence(x, tau, ctx, valid_mask)

        post_full = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=valid_mask)
        post_ctx = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=ctx_mask)

        # KL terms
        kl_full_ctx = post_full.q.kl_to(post_ctx.q)  # [B,N]
        prior = self.prior((B, N), device=x.device, dtype=x.dtype)
        kl_ctx_prior = post_ctx.q.kl_to(prior)  # [B,N]

        if free_bits > 0.0:
            # apply free-bits per dimension approximately by clamping the total KL
            # (coarser than per-dim but prevents collapse in practice)
            kl_full_ctx = torch.clamp(kl_full_ctx, min=free_bits * self.z_dim)
            kl_ctx_prior = torch.clamp(kl_ctx_prior, min=free_bits * self.z_dim)

        # Query likelihood under q(z|C∪Q)
        # Monte Carlo over z
        B, N, T, Dx = x.shape
        S = int(max(1, n_z_samples))
        z = post_full.q.rsample()  # [B,N,Dz]
        if S > 1:
            zs = [z]
            for _ in range(S - 1):
                zs.append(post_full.q.rsample())
            z = torch.stack(zs, dim=0)  # [S,B,N,Dz]

        def nll_given_z(z_s: torch.Tensor) -> torch.Tensor:
            # z_s: [B,N,Dz]
            mu_x, logstd_x = self.decoder(z_s, tau)
            inv_var = torch.exp(-2.0 * logstd_x)
            nll = 0.5 * ((x - mu_x).pow(2) * inv_var + 2.0 * logstd_x + math.log(2.0 * math.pi))
            return nll.sum(dim=-1)  # [B,N,T]

        if S > 1:
            nll = torch.stack([nll_given_z(z[s]) for s in range(S)], dim=0).mean(dim=0)
        else:
            nll = nll_given_z(z)

        loss_query_nll = masked_mean(nll, query_mask, dim=-1)  # [B,N]

        # Precision increment regularizer (paper: ||ΔΛ||_F^2)
        prec_reg = (delta_L.pow(2).sum(dim=-1))  # [B,N,T]
        loss_prec_reg = masked_mean(prec_reg, valid_mask.to(dtype=torch.float32), dim=-1)  # [B,N]

        # Total (average over valid agents)
        agent_valid = valid_mask[..., -1].to(dtype=torch.float32)  # [B,N]
        loss_total = (
            loss_query_nll
            + float(lambda_kl) * kl_full_ctx
            + float(lambda_prior) * kl_ctx_prior
            + float(lambda_prec) * loss_prec_reg
        )
        loss_total = masked_mean(loss_total, agent_valid)

        return PreferenceCompletionOutput(
            post_full=post_full,
            post_ctx=post_ctx,
            ctx_mask=ctx_mask,
            query_mask=query_mask,
            loss_total=loss_total,
            loss_query_nll=masked_mean(loss_query_nll, agent_valid),
            loss_kl_full_ctx=masked_mean(kl_full_ctx, agent_valid),
            loss_kl_ctx_prior=masked_mean(kl_ctx_prior, agent_valid),
            loss_prec_reg=masked_mean(loss_prec_reg, agent_valid),
        )
