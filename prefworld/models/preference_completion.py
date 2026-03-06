from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.models.gaussian import DiagGaussian, NaturalDiagGaussian
from prefworld.models.motion_primitives import MotionPrimitiveDecoder, PrimitiveDecodeOutput


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None, eps: float = 1e-6) -> torch.Tensor:
    """Mean of x over entries where mask==1.

    mask can be bool or float.
    """
    mask_f = mask.to(dtype=x.dtype)
    if dim is None:
        return (x * mask_f).sum() / (mask_f.sum() + eps)
    return (x * mask_f).sum(dim=dim) / (mask_f.sum(dim=dim) + eps)


@dataclass
class PreferencePosterior:
    """Posterior belief q(z) for each agent."""

    q: DiagGaussian               # [B,N,Dz]
    nat: NaturalDiagGaussian      # [B,N,Dz]
    alpha: torch.Tensor           # [B,N,T] evidence reliability gates


@dataclass
class PreferenceCompletionOutput:
    """Outputs for preference completion training."""

    post_full: PreferencePosterior
    post_ctx: PreferencePosterior

    ctx_mask: torch.Tensor        # [B,N,T]
    query_mask: torch.Tensor      # [B,N,T]

    maneuver_logits_last: torch.Tensor  # [B,N,M]

    # scalar losses
    loss_total: torch.Tensor

    loss_query_nll: torch.Tensor
    loss_distill_mu: torch.Tensor
    loss_distill_cov: torch.Tensor
    loss_kl_ctx_prior: torch.Tensor
    loss_contrastive: torch.Tensor
    loss_overlap: torch.Tensor
    loss_modulation: torch.Tensor


class EvidenceEncoder(nn.Module):
    """Encode each evidence token into natural-parameter increments.

    Paper Eq.(15):
      (α_t, ΔΛ_t, Δη_t) = f_φ(e_t),  α∈(0,1],  ΔΛ≽0

    We implement a diagonal-precision version for stability.
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
        mask: torch.Tensor,      # [B,N,T]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([x, tau, ctx], dim=-1))
        delta_eta = self.to_eta(h)                              # [B,N,T,Dz]
        delta_L = F.softplus(self.to_L(h))                      # [B,N,T,Dz] >= 0
        alpha = torch.sigmoid(self.to_alpha(h)).squeeze(-1)     # [B,N,T]

        m = mask > 0.5
        delta_eta = delta_eta * m.unsqueeze(-1).to(dtype=delta_eta.dtype)
        delta_L = delta_L * m.unsqueeze(-1).to(dtype=delta_L.dtype)
        alpha = alpha * m.to(dtype=alpha.dtype)
        return delta_eta, delta_L, alpha


class PreferenceCompletion(nn.Module):
    """Preference completion via tempered neural evidence accumulation.

    Implements paper Sec.4 objective (Eq.18):

      L_PC = -E_{q(z|C)}[ Σ_{t∈Q} log p(X_t | z, τ_t) ]
             + λ^μ ||μ_C - sg(μ_{C∪Q})||^2
             + λ^Σ KL(N(0, sg(Σ_{C∪Q})) || N(0, Σ_C))
             + λ_prior KL(q(z|C) || p(z))
             + λ_con L_con
             + λ_Λ R_Λ

    Notes:
      - We use diagonal Gaussians for q(z).
      - Maneuvers are latent and marginalized inside p(X_t|z,τ_t).
      - The primitive emission model is pluggable via MotionPrimitiveDecoder.
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
        num_maneuvers: int = 6,
        # recommended stability improvements
        context_dependent_prior: bool = True,
        prior_hidden_dim: Optional[int] = None,
        maneuver_beta: float = 1.0,
        feasible_action_penalty: float = 5.0,
        feasible_action_soft_penalty_train: bool = True,
        feasible_action_hard_mask_eval: bool = True,
        # contrastive head
        con_proj_dim: int = 32,
        con_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.tau_dim = int(tau_dim)
        self.ctx_dim = int(ctx_dim)
        self.z_dim = int(z_dim)
        self.num_maneuvers = int(num_maneuvers)
        self.con_temperature = float(con_temperature)

        self.evidence = EvidenceEncoder(x_dim=x_dim, tau_dim=tau_dim, ctx_dim=ctx_dim, z_dim=z_dim, hidden_dim=hidden_dim)
        self.decoder = MotionPrimitiveDecoder(
            x_dim=x_dim,
            tau_dim=tau_dim,
            ctx_dim=ctx_dim,
            z_dim=z_dim,
            num_maneuvers=self.num_maneuvers,
            hidden_dim=hidden_dim,
            beta=float(maneuver_beta),
            feasible_action_penalty=float(feasible_action_penalty),
            feasible_action_soft_penalty_train=bool(feasible_action_soft_penalty_train),
            feasible_action_hard_mask_eval=bool(feasible_action_hard_mask_eval),
        )

        # Projection head g for contrastive embeddings r=normalize(g(μ))
        self.con_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(con_proj_dim)),
        )

        # fixed diagonal Gaussian prior p(z) = N(0, exp(prior_logvar) I)
        self.register_buffer("prior_mean", torch.zeros(z_dim))
        self.register_buffer("prior_logvar", torch.full((z_dim,), float(prior_logvar)))

        # Context-dependent prior p(z | c) (recommended; improves PC stability and expressiveness)
        self.use_context_prior = bool(context_dependent_prior)
        if prior_hidden_dim is None:
            prior_hidden_dim = int(hidden_dim)
        self._prior_hidden_dim = int(prior_hidden_dim)
        if self.use_context_prior:
            self.prior_net = nn.Sequential(
                nn.Linear(int(tau_dim + ctx_dim), int(prior_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(prior_hidden_dim), 2 * int(z_dim)),
            )
            # initialize close to the fixed prior: mean≈0, logvar≈prior_logvar
            last: nn.Linear = self.prior_net[-1]  # type: ignore[assignment]
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias[: int(z_dim)].zero_()
                last.bias[int(z_dim) :].fill_(float(prior_logvar))
        else:
            self.prior_net = None

    def prior(self, batch_shape: Tuple[int, ...], device=None, dtype=None) -> DiagGaussian:
        mean = self.prior_mean.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        logvar = self.prior_logvar.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        return DiagGaussian(mean=mean, logvar=logvar)

    def prior_from_context(
        self,
        *,
        tau: torch.Tensor,   # [B,N,T,Dt]
        ctx: torch.Tensor,   # [B,N,T,Dc]
        mask: torch.Tensor,  # [B,N,T] bool/float
    ) -> DiagGaussian:
        """Context-dependent prior p(z|c).

        If context_dependent_prior is disabled, this returns the fixed N(0,I) prior.
        """
        B, N, T, _ = tau.shape
        if not self.use_context_prior or self.prior_net is None:
            return self.prior((B, N), device=tau.device, dtype=tau.dtype)

        m = (mask > 0.5).to(dtype=tau.dtype)
        denom = m.sum(dim=2, keepdim=True).clamp_min(1.0)
        tau_mean = (tau * m.unsqueeze(-1)).sum(dim=2) / denom
        ctx_mean = (ctx * m.unsqueeze(-1)).sum(dim=2) / denom
        h = torch.cat([tau_mean, ctx_mean], dim=-1)
        out = self.prior_net(h)
        mean = out[..., : self.z_dim]
        logvar = out[..., self.z_dim :].clamp(min=-10.0, max=10.0)
        return DiagGaussian(mean=mean, logvar=logvar)

    def _posterior_from_mask(
        self,
        *,
        delta_eta: torch.Tensor,     # [B,N,T,Dz]
        delta_L: torch.Tensor,       # [B,N,T,Dz]
        alpha: torch.Tensor,         # [B,N,T]
        mask: torch.Tensor,          # [B,N,T]
        prior: Optional[DiagGaussian] = None,
    ) -> PreferencePosterior:
        B, N, T, Dz = delta_eta.shape
        m = mask > 0.5
        a = alpha.unsqueeze(-1)

        if prior is None:
            prior = self.prior((B, N), device=delta_eta.device, dtype=delta_eta.dtype)
        nat0 = NaturalDiagGaussian.from_moment(prior)

        pooled_eta = nat0.eta + (a * delta_eta * m.unsqueeze(-1).to(dtype=delta_eta.dtype)).sum(dim=2)
        pooled_L = nat0.Lambda + (a * delta_L * m.unsqueeze(-1).to(dtype=delta_L.dtype)).sum(dim=2)
        nat = NaturalDiagGaussian(eta=pooled_eta, Lambda=pooled_L.clamp_min(1e-6))
        q = nat.to_moment()
        return PreferencePosterior(q=q, nat=nat, alpha=alpha)

    @staticmethod
    def _ensure_nonempty(mask_all: torch.Tensor, mask_sub: torch.Tensor) -> torch.Tensor:
        """Ensure each (B,N) with any valid token has >=1 token in mask_sub."""
        B, N, T = mask_all.shape
        valid = mask_all.sum(dim=-1) > 0
        sub_ok = mask_sub.sum(dim=-1) > 0
        need = valid & (~sub_ok)
        if need.any():
            idx = mask_all.float().argmax(dim=-1)  # [B,N]
            b, n = need.nonzero(as_tuple=True)
            mask_sub = mask_sub.clone()
            mask_sub[b, n, idx[b, n]] = 1.0
        return mask_sub

    def split_context_query(
        self,
        valid_mask: torch.Tensor,  # [B,N,T]
        *,
        mode: str = "random",      # random | prefix
        query_ratio: float = 0.3,
        ensure_query_nonempty: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split valid_mask into (ctx_mask, query_mask).

        If query_ratio <= 0, returns ctx=valid and query=0.
        """
        m = valid_mask > 0.5
        B, N, T = m.shape

        if float(query_ratio) <= 0.0:
            ctx_mask = m.to(dtype=torch.float32)
            query_mask = torch.zeros_like(ctx_mask)
            return ctx_mask, query_mask

        if mode == "prefix":
            lengths = m.sum(dim=-1).clamp(min=0).to(torch.int64)  # [B,N]
            cutoff = torch.zeros((B, N), device=m.device, dtype=torch.int64)
            ok = lengths >= 2
            if ok.any():
                rnd = torch.randint(0, 10_000, (int(ok.sum().item()),), device=m.device)
                cutoff[ok] = 1 + (rnd % (lengths[ok] - 1))
            t_idx = torch.arange(T, device=m.device).view(1, 1, T)
            ctx_mask = (t_idx < cutoff.unsqueeze(-1)) & m
            query_mask = m & (~ctx_mask)
        else:
            rnd = torch.rand((B, N, T), device=m.device)
            query_mask = (rnd < float(query_ratio)) & m
            ctx_mask = m & (~query_mask)
            ctx_mask = self._ensure_nonempty(m.to(dtype=torch.float32), ctx_mask.to(dtype=torch.float32)) > 0.5
            query_mask = m & (~ctx_mask)

        if ensure_query_nonempty:
            lengths = m.sum(dim=-1)
            q_counts = query_mask.sum(dim=-1)
            need_q = (lengths >= 2) & (q_counts == 0)
            if need_q.any():
                # pick the last valid token as query
                last_valid = (m.to(torch.int64).cumsum(dim=-1) == lengths.unsqueeze(-1).to(torch.int64)).to(torch.int64).argmax(dim=-1)
                b, n = need_q.nonzero(as_tuple=True)
                t = last_valid[b, n]
                query_mask = query_mask.clone()
                ctx_mask = ctx_mask.clone()
                query_mask[b, n, t] = True
                ctx_mask[b, n, t] = False

        return ctx_mask.to(dtype=torch.float32), query_mask.to(dtype=torch.float32)

    @staticmethod
    def _kl_zero_mean_diag(teacher_logvar: torch.Tensor, student_logvar: torch.Tensor) -> torch.Tensor:
        """KL(N(0, Σ_t) || N(0, Σ_s)) for diagonal covariances.

        Args:
          teacher_logvar: [...,D]
          student_logvar: [...,D]
        Returns:
          kl: [...]
        """
        v_t = torch.exp(teacher_logvar)
        v_s = torch.exp(student_logvar)
        return 0.5 * torch.sum(student_logvar - teacher_logvar + v_t / (v_s + 1e-8) - 1.0, dim=-1)

    def _contrastive_loss(self, mu1: torch.Tensor, mu2: torch.Tensor, agent_valid: torch.Tensor) -> torch.Tensor:
        """InfoNCE over (B,N) agents in a batch.

        Args:
          mu1,mu2: [B,N,Dz]
          agent_valid: [B,N] bool/float
        Returns:
          scalar loss
        """
        m = agent_valid > 0.5
        if m.sum() <= 1:
            return torch.zeros((), device=mu1.device, dtype=torch.float32)

        # flatten valid agents
        mu1_f = mu1[m]
        mu2_f = mu2[m]

        r1 = F.normalize(self.con_head(mu1_f), dim=-1)
        r2 = F.normalize(self.con_head(mu2_f), dim=-1)

        sim = (r1 @ r2.t()) / max(1e-6, self.con_temperature)  # [M,M]
        logp = F.log_softmax(sim, dim=1)
        idx = torch.arange(sim.shape[0], device=sim.device)
        loss = -logp[idx, idx].mean()
        return loss

    def forward(
        self,
        *,
        x: torch.Tensor,           # [B,N,T,Dx]
        tau: torch.Tensor,         # [B,N,T,Dt]
        ctx: torch.Tensor,         # [B,N,T,Dc]
        mask: torch.Tensor,        # [B,N,T]
        feasible_actions: Optional[torch.Tensor] = None,  # [B,N,T,M] bool
        # episode split
        split_mode: str = "random",
        query_ratio: float = 0.3,
        # Loss weights
        lambda_distill_mu: float = 1.0,
        lambda_distill_cov: float = 0.05,
        lambda_prior: float = 0.1,
        lambda_con: float = 0.05,
        lambda_overlap: float = 1e-3,
        lambda_mod: float = 1e-3,
        # MC
        n_z_samples: int = 1,
        free_bits: float = 0.0,
        # optional externally provided split masks
        ctx_mask_override: Optional[torch.Tensor] = None,
        query_mask_override: Optional[torch.Tensor] = None,
        ensure_query_nonempty: Optional[bool] = None,
    ) -> PreferenceCompletionOutput:
        """Compute PC losses."""
        B, N, T, _ = x.shape
        valid_mask = mask > 0.5

        if ctx_mask_override is not None and query_mask_override is not None:
            ctx_mask = ctx_mask_override.to(dtype=torch.float32)
            query_mask = query_mask_override.to(dtype=torch.float32)
        else:
            if ensure_query_nonempty is None:
                ensure_query_nonempty = float(query_ratio) > 0.0
            ctx_mask, query_mask = self.split_context_query(
                valid_mask.to(dtype=torch.float32),
                mode=str(split_mode),
                query_ratio=float(query_ratio),
                ensure_query_nonempty=bool(ensure_query_nonempty),
            )

        # evidence -> natural increments
        delta_eta, delta_L, alpha = self.evidence(x, tau, ctx, valid_mask)

        # Context-dependent prior p(z|c) (if enabled). We compute it from *context tokens* only.
        prior_ctx = self.prior_from_context(tau=tau, ctx=ctx, mask=(ctx_mask > 0.5) & valid_mask)

        # teacher on full evidence (C∪Q)
        full_mask = (ctx_mask + query_mask) > 0.5
        full_mask = full_mask & valid_mask

        post_full = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=full_mask, prior=prior_ctx)
        post_ctx = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=ctx_mask, prior=prior_ctx)

        # agent validity (at least one token)
        agent_valid = (valid_mask.sum(dim=-1) > 0).to(dtype=torch.float32)  # [B,N]

        # ---- Query likelihood under q(z|C) ----
        S = int(max(1, n_z_samples))
        if S == 1:
            z_samps = post_ctx.q.rsample().unsqueeze(0)
        else:
            z_samps = torch.stack([post_ctx.q.rsample() for _ in range(S)], dim=0)

        query_counts = query_mask.sum(dim=-1).clamp_min(0.0)  # [B,N]
        denom = (query_counts + 1e-6)

        def nll_given_z(z_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            dec: PrimitiveDecodeOutput = self.decoder.token_log_prob(
                x=x,
                tau=tau,
                ctx=ctx,
                mask=valid_mask.to(dtype=torch.float32),
                feasible_actions=feasible_actions,
                z=z_s,
            )
            log_pi = F.log_softmax(dec.maneuver_logits, dim=-1)
            log_mix = torch.logsumexp(log_pi + dec.logp_x_given_m, dim=-1)  # [B,N,T]
            log_mix = torch.nan_to_num(log_mix, nan=0.0, posinf=0.0, neginf=0.0)
            # average NLL over query tokens
            # 把 query_mask=0 的位置显式置零，避免 NaN*0 仍 NaN
            qm = query_mask > 0.5
            log_mix_q = log_mix.masked_fill(~qm, 0.0)
            nll = -(log_mix_q).sum(dim=-1) / denom
            # modulation reg proxy (per agent)
            mod = (dec.z_mod_delta.pow(2).sum(dim=-1) * valid_mask.to(dtype=torch.float32)).sum(dim=-1) / (valid_mask.to(dtype=torch.float32).sum(dim=-1) + 1e-6)

            print("DBG logp_x_given_m:", dec.logp_x_given_m.min().item(), dec.logp_x_given_m.max().item())
            print("DBG logits:", dec.maneuver_logits.min().item(), dec.maneuver_logits.max().item())
            print("DBG log_mix:", log_mix.min().item(), log_mix.max().item())

            return nll, mod

        nll_s = []
        mod_s = []
        for s in range(S):
            nll_one, mod_one = nll_given_z(z_samps[s])
            nll_s.append(nll_one)
            mod_s.append(mod_one)
        loss_query_nll = torch.stack(nll_s, dim=0).mean(dim=0)  # [B,N]
        loss_modulation = torch.stack(mod_s, dim=0).mean(dim=0)  # [B,N]

        # ---- Distillation: mean + covariance ----
        mu_ctx = post_ctx.q.mean
        mu_full = post_full.q.mean.detach()
        loss_distill_mu = ((mu_ctx - mu_full) ** 2).sum(dim=-1)  # [B,N]

        # KL(N(0, Σ_full)||N(0, Σ_ctx))
        loss_distill_cov = self._kl_zero_mean_diag(post_full.q.logvar.detach(), post_ctx.q.logvar)  # [B,N]

        # ---- Prior KL ----
        loss_kl_ctx_prior = post_ctx.q.kl_to(prior_ctx)  # [B,N]

        if free_bits > 0.0:
            # clamp KL terms only
            fb = float(free_bits) * float(self.z_dim)
            loss_distill_cov = torch.clamp(loss_distill_cov, min=0.0)  # keep non-negative
            loss_kl_ctx_prior = torch.clamp(loss_kl_ctx_prior, min=fb)

        # ---- Contrastive (InfoNCE) ----
        # sample two random views from full evidence tokens
        rnd1 = (torch.rand((B, N, T), device=x.device) < 0.5) & full_mask
        rnd2 = (torch.rand((B, N, T), device=x.device) < 0.5) & full_mask
        view1 = self._ensure_nonempty(full_mask.to(dtype=torch.float32), rnd1.to(dtype=torch.float32)) > 0.5
        view2 = self._ensure_nonempty(full_mask.to(dtype=torch.float32), rnd2.to(dtype=torch.float32)) > 0.5
        prior_v1 = self.prior_from_context(tau=tau, ctx=ctx, mask=view1 & valid_mask)
        prior_v2 = self.prior_from_context(tau=tau, ctx=ctx, mask=view2 & valid_mask)
        post_v1 = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=view1, prior=prior_v1)
        post_v2 = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=view2, prior=prior_v2)
        loss_contrastive = self._contrastive_loss(post_v1.q.mean, post_v2.q.mean, agent_valid)

        # ---- Overlap penalty R_Λ ----
        # v_t = α_t ΔΛ_t (diagonal)
        v = (alpha.unsqueeze(-1) * delta_L) * valid_mask.unsqueeze(-1).to(dtype=delta_L.dtype)  # [B,N,T,Dz]
        sum_v = v.sum(dim=2)  # [B,N,Dz]
        sq_sum = (sum_v**2).sum(dim=-1)  # [B,N]
        sq_each = (v**2).sum(dim=-1).sum(dim=2)  # [B,N]
        overlap = 0.5 * (sq_sum - sq_each)
        overlap = overlap.clamp_min(0.0)
        loss_overlap = overlap  # [B,N]

        # ---- Total ----
        loss_per_agent = (
            loss_query_nll
            + float(lambda_distill_mu) * loss_distill_mu
            + float(lambda_distill_cov) * loss_distill_cov
            + float(lambda_prior) * loss_kl_ctx_prior
            + float(lambda_overlap) * loss_overlap
            + float(lambda_mod) * loss_modulation
        )

        loss_total = masked_mean(loss_per_agent, agent_valid)
        loss_total = loss_total + float(lambda_con) * loss_contrastive

        # Last-token maneuver logits under q(z|C) mean (for logging)
        lengths = valid_mask.to(torch.int64).sum(dim=-1).clamp(min=1)  # [B,N]
        idx = (lengths - 1).view(B, N, 1, 1)
        tau_last = tau.gather(dim=2, index=idx.expand(B, N, 1, tau.shape[-1])).squeeze(2)
        ctx_last = ctx.gather(dim=2, index=idx.expand(B, N, 1, ctx.shape[-1])).squeeze(2)
        maneuver_logits_last = self.decoder.maneuver_logits_last(z=post_ctx.q.mean, tau_last=tau_last, ctx_last=ctx_last).detach()
        if feasible_actions is not None:
            fa_last = feasible_actions.gather(dim=2, index=idx.expand(B, N, 1, feasible_actions.shape[-1])).squeeze(2)
            maneuver_logits_last = self.decoder.maneuver_logits_last(
                z=post_ctx.q.mean,
                tau_last=tau_last,
                ctx_last=ctx_last,
                feasible_actions_last=fa_last,
            ).detach()

        return PreferenceCompletionOutput(
            post_full=post_full,
            post_ctx=post_ctx,
            ctx_mask=ctx_mask,
            query_mask=query_mask,
            maneuver_logits_last=maneuver_logits_last,
            loss_total=loss_total,
            loss_query_nll=masked_mean(loss_query_nll, agent_valid),
            loss_distill_mu=masked_mean(loss_distill_mu, agent_valid),
            loss_distill_cov=masked_mean(loss_distill_cov, agent_valid),
            loss_kl_ctx_prior=masked_mean(loss_kl_ctx_prior, agent_valid),
            loss_contrastive=loss_contrastive.detach() if torch.is_tensor(loss_contrastive) else torch.tensor(float(loss_contrastive)),
            loss_overlap=masked_mean(loss_overlap, agent_valid),
            loss_modulation=masked_mean(loss_modulation, agent_valid),
        )
