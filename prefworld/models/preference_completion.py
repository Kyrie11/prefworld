from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.models.gaussian import DiagGaussian, NaturalDiagGaussian
from prefworld.models.motion_primitives import MotionPrimitiveDecoder


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean of x over entries where mask==1.

    Supports broadcasting between x and mask.
    """
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum() / (mask_f.sum() + eps)


@dataclass
class PreferencePosterior:
    q: DiagGaussian          # [B,N,Dz]
    nat: NaturalDiagGaussian # [B,N,Dz]
    alpha: torch.Tensor      # [B,N,T]


@dataclass
class PreferenceCompletionOutput:
    post_full: PreferencePosterior
    post_ctx: PreferencePosterior

    ctx_mask: torch.Tensor
    query_mask: torch.Tensor

    maneuver_logits_last: torch.Tensor
    action_logits_last: Optional[torch.Tensor] = None
    action_family_last: Optional[torch.Tensor] = None

    loss_total: torch.Tensor = None  # type: ignore[assignment]

    # kept name for backward compatibility; implements paper's choice-predictive CE
    loss_query_nll: torch.Tensor = None  # type: ignore[assignment]

    loss_distill_mu: torch.Tensor = None  # type: ignore[assignment]
    loss_distill_cov: torch.Tensor = None  # type: ignore[assignment]
    loss_kl_ctx_prior: torch.Tensor = None  # type: ignore[assignment]
    loss_contrastive: torch.Tensor = None  # type: ignore[assignment]

    # no longer used in paper-aligned implementation
    loss_overlap: torch.Tensor = None  # type: ignore[assignment]
    loss_modulation: torch.Tensor = None  # type: ignore[assignment]

    # paper-aligned u_ctx regularizer
    loss_u_ctx: torch.Tensor = None  # type: ignore[assignment]


class EvidenceEncoder(nn.Module):
    """Encode per-token *choice evidence* into diagonal natural-parameter increments.

    Inputs are constructed from the recognition distribution q_χ(m|X_t,τ_t^{det}) and
    the physically-comparable feature vector f(m,τ_t^{det}).

    We use:
        \bar f_t = E_{m~q_χ}[f(m,τ_t^{det})]
        conf_t = 1 - H(q_χ)/log|A_t|

    and output Δη_t, ΔΛ_t with ΔΛ_t >= 0.
    """

    def __init__(
        self,
        *,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        in_dim = int(z_dim + tau_dim + ctx_dim + 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_eta = nn.Linear(hidden_dim, z_dim)
        self.to_L = nn.Linear(hidden_dim, z_dim)

        nn.init.zeros_(self.to_eta.weight)
        nn.init.zeros_(self.to_eta.bias)
        nn.init.zeros_(self.to_L.weight)
        nn.init.constant_(self.to_L.bias, -3.0)  # softplus(-3) ~ 0.05

    def forward(
        self,
        *,
        f_bar: torch.Tensor,  # [B,N,T,Dz]
        tau: torch.Tensor,    # [B,N,T,Dt]
        ctx: torch.Tensor,    # [B,N,T,Dc]
        conf: torch.Tensor,   # [B,N,T] in [0,1]
        mask: torch.Tensor,   # [B,N,T]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conf = conf.clamp(0.0, 1.0)
        inp = torch.cat([f_bar, tau, ctx, conf.unsqueeze(-1)], dim=-1)
        inp = F.layer_norm(inp, (inp.shape[-1],))
        h = self.net(inp)
        delta_eta = self.to_eta(h)
        delta_L = F.softplus(self.to_L(h)).clamp_max(10.0)
        alpha = conf

        m = mask > 0.5
        delta_eta = delta_eta * m.unsqueeze(-1).to(dtype=delta_eta.dtype)
        delta_L = delta_L * m.unsqueeze(-1).to(dtype=delta_L.dtype)
        alpha = alpha * m.to(dtype=alpha.dtype)
        return delta_eta, delta_L, alpha


class PreferenceCompletion(nn.Module):
    """Preference completion (paper-aligned).

    Major alignment points:
    - Primitive likelihood p_n(X|m,τ^{det}) is independent of z.
    - Evidence updates are driven by recognition q_χ(m|X,τ^{det}), not raw X.
    - Training uses a choice-predictive objective (paper Eq. pc_choice).
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int,
        num_maneuvers: int,
        action_feature_dim: int,
        hidden_dim: int = 128,
        prior_logvar: float = 0.0,
        num_z_samples: int = 8,
        # default weights (can be overridden per forward call for schedules)
        lambda_query: float = 1.0,
        lambda_distill_mu: float = 0.5,
        lambda_distill_cov: float = 0.1,
        lambda_prior: float = 0.01,
        lambda_con: float = 0.01,
        lambda_u_ctx: float = 0.01,
        # context-dependent prior
        context_dependent_prior: bool = False,
        prior_hidden_dim: Optional[int] = None,
        # decoder
        maneuver_beta: float = 1.0,
        feasible_action_penalty: float = 5.0,
        feasible_action_soft_penalty_train: bool = True,
        feasible_action_hard_mask_eval: bool = True,
        # primitive likelihood params
        dt: float = 0.1,
        lane_width: float = 3.6,
        topo_horizon_m: float = 50.0,
        conflict_time_s: float = 5.0,
        a_min: float = -4.0,
        a_max: float = 2.0,
        k_yield: float = 1.5,
        rho_grid: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
        # contrastive
        con_proj_dim: int = 32,
        con_temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.tau_dim = int(tau_dim)
        self.ctx_dim = int(ctx_dim)
        self.z_dim = int(z_dim)
        self.num_maneuvers = int(num_maneuvers)
        self.action_feature_dim = int(action_feature_dim)  # kept for legacy

        self.num_z_samples = int(num_z_samples)

        # default weights
        self.lambda_query = float(lambda_query)
        self.lambda_distill_mu = float(lambda_distill_mu)
        self.lambda_distill_cov = float(lambda_distill_cov)
        self.lambda_prior = float(lambda_prior)
        self.lambda_con = float(lambda_con)
        self.lambda_u_ctx = float(lambda_u_ctx)

        self.evidence = EvidenceEncoder(tau_dim=tau_dim, ctx_dim=ctx_dim, z_dim=z_dim, hidden_dim=hidden_dim)

        self.decoder = MotionPrimitiveDecoder(
            x_dim=x_dim,
            tau_dim=tau_dim,
            ctx_dim=ctx_dim,
            z_dim=z_dim,
            num_maneuvers=num_maneuvers,
            beta=float(maneuver_beta),
            feasible_action_penalty=float(feasible_action_penalty),
            feasible_action_soft_penalty_train=bool(feasible_action_soft_penalty_train),
            feasible_action_hard_mask_eval=bool(feasible_action_hard_mask_eval),
            dt=float(dt),
            lane_width=float(lane_width),
            topo_horizon_m=float(topo_horizon_m),
            conflict_time_s=float(conflict_time_s),
            a_min=float(a_min),
            a_max=float(a_max),
            k_yield=float(k_yield),
            rho_grid=rho_grid,
        )

        self.con_temperature = float(con_temperature)
        self.con_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(con_proj_dim)),
        )

        # fixed diagonal prior (if context prior disabled)
        self.register_buffer("prior_mean", torch.zeros(z_dim))
        self.register_buffer("prior_logvar", torch.full((z_dim,), float(prior_logvar)))

        self.use_context_prior = bool(context_dependent_prior)
        if prior_hidden_dim is None:
            prior_hidden_dim = int(hidden_dim)
        if self.use_context_prior:
            self.prior_net = nn.Sequential(
                nn.Linear(int(tau_dim + ctx_dim), int(prior_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(prior_hidden_dim), 2 * int(z_dim)),
            )
            last: nn.Linear = self.prior_net[-1]  # type: ignore[assignment]
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias[: int(z_dim)].zero_()
                last.bias[int(z_dim) :].fill_(float(prior_logvar))
        else:
            self.prior_net = None

    # ----------------------
    # Prior
    # ----------------------

    def prior(self, batch_shape: Tuple[int, ...], device=None, dtype=None) -> DiagGaussian:
        mean = self.prior_mean.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        logvar = self.prior_logvar.to(device=device, dtype=dtype).expand(*batch_shape, self.z_dim)
        return DiagGaussian(mean=mean, logvar=logvar)

    def prior_from_context(self, *, tau: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor) -> DiagGaussian:
        """Context-dependent prior p(z|c) (optional)."""
        B, N, T, _ = tau.shape
        if not self.use_context_prior or self.prior_net is None:
            return self.prior((B, N), device=tau.device, dtype=tau.dtype)

        m = (mask > 0.5).to(dtype=tau.dtype)
        denom = m.sum(dim=2, keepdim=True).clamp_min(1.0)
        tau_mean = (tau * m.unsqueeze(-1)).sum(dim=2) / denom
        ctx_mean = (ctx * m.unsqueeze(-1)).sum(dim=2) / denom
        h = torch.cat([tau_mean, ctx_mean], dim=-1)
        h = F.layer_norm(h, (h.shape[-1],))
        out = self.prior_net(h)
        mean = out[..., : self.z_dim]
        logvar = out[..., self.z_dim :].clamp(min=-10.0, max=2.0)
        return DiagGaussian(mean=mean, logvar=logvar)

    # ----------------------
    # Posterior from masks
    # ----------------------

    def _posterior_from_mask(
        self,
        *,
        delta_eta: torch.Tensor,     # [B,N,T,Dz]
        delta_L: torch.Tensor,       # [B,N,T,Dz]
        alpha: torch.Tensor,         # [B,N,T]
        mask: torch.Tensor,          # [B,N,T]
        prior: DiagGaussian,
    ) -> PreferencePosterior:
        m = (mask > 0.5).to(dtype=delta_eta.dtype)
        a = alpha.unsqueeze(-1).to(dtype=delta_eta.dtype)
        nat0 = NaturalDiagGaussian.from_moment(prior)
        pooled_eta = nat0.eta + (a * delta_eta * m.unsqueeze(-1)).sum(dim=2)
        pooled_L = nat0.Lambda + (a * delta_L * m.unsqueeze(-1)).sum(dim=2)

        L_MIN = 1e-6
        L_MAX = 50.0
        nat = NaturalDiagGaussian(
            eta=pooled_eta.clamp(-1e3, 1e3),
            Lambda=pooled_L.clamp(L_MIN, L_MAX),
        )
        q = nat.to_moment()
        return PreferencePosterior(q=q, nat=nat, alpha=alpha)

    # ----------------------
    # Mask splitting
    # ----------------------

    @staticmethod
    def _ensure_nonempty(mask_all: torch.Tensor, mask_sub: torch.Tensor) -> torch.Tensor:
        """Ensure each (B,N) with any valid token has >=1 token in mask_sub."""
        valid = (mask_all > 0.5).sum(dim=-1) > 0
        sub_ok = (mask_sub > 0.5).sum(dim=-1) > 0
        need = valid & (~sub_ok)
        if need.any():
            # pick last valid index
            idx = (mask_all > 0.5).float().flip(dims=[-1]).argmax(dim=-1)
            idx = (mask_all.shape[-1] - 1 - idx).long()
            b, n = need.nonzero(as_tuple=True)
            mask_sub = mask_sub.clone()
            mask_sub[b, n, idx[b, n]] = 1.0
        return mask_sub

    def split_context_query(
        self,
        valid_mask: torch.Tensor,
        *,
        mode: str = "random",
        query_ratio: float = 0.3,
        ensure_query_nonempty: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split valid_mask into (ctx_mask, query_mask)."""
        m = valid_mask > 0.5
        if query_ratio <= 0.0:
            ctx = valid_mask.clone()
            qry = torch.zeros_like(valid_mask)
            return ctx, qry

        if mode not in ("random", "prefix"):
            raise ValueError(f"Unsupported split_mode={mode}")

        if mode == "random":
            rnd = torch.rand_like(valid_mask)
            qry = (rnd < query_ratio) & m
            ctx = m & (~qry)
        else:
            # prefix context, suffix query
            B, N, T = valid_mask.shape
            ctx = torch.zeros_like(m)
            qry = torch.zeros_like(m)
            for b in range(B):
                for n in range(N):
                    valid_idx = torch.nonzero(m[b, n], as_tuple=False).squeeze(-1)
                    if valid_idx.numel() == 0:
                        continue
                    qn = max(1, int(math.ceil(valid_idx.numel() * query_ratio)))
                    qn = min(qn, valid_idx.numel())
                    qry_idx = valid_idx[-qn:]
                    ctx_idx = valid_idx[:-qn]
                    qry[b, n, qry_idx] = True
                    ctx[b, n, ctx_idx] = True

        ctx = ctx.to(dtype=valid_mask.dtype)
        qry = qry.to(dtype=valid_mask.dtype)
        if ensure_query_nonempty:
            qry = self._ensure_nonempty(valid_mask, qry)
            ctx = (valid_mask > 0.5).to(dtype=valid_mask.dtype) - qry
            ctx = ctx.clamp(0.0, 1.0)
        return ctx, qry


    # ----------------------
    # Invariance regularizer R_inv (paper Eq. rinv_def)
    # ----------------------

    def _sample_evidence_matched_subcontexts(self, ctx_mask: torch.Tensor, conf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample two sub-context masks with roughly matched evidence strength.

        Evidence strength is approximated by the recognition confidence conf_t.
        """
        B, N, T = ctx_mask.shape
        device = ctx_mask.device
        c1 = torch.zeros_like(ctx_mask)
        c2 = torch.zeros_like(ctx_mask)
        m = ctx_mask > 0.5
        for b in range(B):
            for n in range(N):
                idx = torch.nonzero(m[b, n], as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                if idx.numel() == 1:
                    t0 = int(idx[0].item())
                    c1[b, n, t0] = 1.0
                    c2[b, n, t0] = 1.0
                    continue
                s = conf[b, n, idx].detach().clamp(0.0, 1.0)
                target = 0.5 * float(s.sum().item())

                # greedy fill on a random permutation
                perm1 = idx[torch.randperm(idx.numel(), device=device)]
                cum = 0.0
                for t in perm1.tolist():
                    c1[b, n, t] = 1.0
                    cum += float(conf[b, n, t].item())
                    if cum >= target and (c1[b, n] > 0.5).sum() >= 1:
                        break

                perm2 = idx[torch.randperm(idx.numel(), device=device)]
                cum = 0.0
                for t in perm2.tolist():
                    c2[b, n, t] = 1.0
                    cum += float(conf[b, n, t].item())
                    if cum >= target and (c2[b, n] > 0.5).sum() >= 1:
                        break

        c1 = self._ensure_nonempty(ctx_mask, c1)
        c2 = self._ensure_nonempty(ctx_mask, c2)
        return c1, c2

    @staticmethod
    def _sym_kl(q1: DiagGaussian, q2: DiagGaussian) -> torch.Tensor:
        return 0.5 * (q1.kl_to(q2) + q2.kl_to(q1))
    # ----------------------
    # Forward
    # ----------------------

    def forward(
        self,
        *,
        x: torch.Tensor,
        tau: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor,
        feasible_actions: Optional[torch.Tensor] = None,
        action_features: Optional[torch.Tensor] = None,
        action_family: Optional[torch.Tensor] = None,
        comparable_metrics: Optional[torch.Tensor] = None,
        dynamic_metrics: Optional[torch.Tensor] = None,
        action_path_type: Optional[torch.Tensor] = None,
        action_constraint_type: Optional[torch.Tensor] = None,
        path_polyline_idx: Optional[torch.Tensor] = None,
        map_polylines: Optional[torch.Tensor] = None,
        # overrides (kept for compatibility with training scripts)
        split_mode: str = "random",
        query_ratio: float = 0.5,
        lambda_distill_mu: Optional[float] = None,
        lambda_distill_cov: Optional[float] = None,
        lambda_prior: Optional[float] = None,
        lambda_con: Optional[float] = None,
        lambda_overlap: Optional[float] = None,
        lambda_mod: Optional[float] = None,
        lambda_u_ctx: Optional[float] = None,
        n_z_samples: Optional[int] = None,
        free_bits: float = 0.0,
        ctx_mask_override: Optional[torch.Tensor] = None,
        query_mask_override: Optional[torch.Tensor] = None,
        ensure_query_nonempty: bool = True,
    ) -> PreferenceCompletionOutput:
        B, N, T, _ = x.shape
        device = x.device

        valid_mask = (mask > 0.5).to(dtype=mask.dtype)
        agent_valid = valid_mask.sum(dim=-1) > 0.5

        if ctx_mask_override is not None and query_mask_override is not None:
            ctx_mask = ctx_mask_override.to(dtype=mask.dtype)
            query_mask = query_mask_override.to(dtype=mask.dtype)
        else:
            ctx_mask, query_mask = self.split_context_query(
                valid_mask,
                mode=split_mode,
                query_ratio=query_ratio,
                ensure_query_nonempty=ensure_query_nonempty,
            )

        # Recognition + decision features (independent of z)
        z_dummy = torch.zeros((B, N, self.z_dim), device=device, dtype=x.dtype)
        dec0 = self.decoder.token_log_prob(
            x=x,
            tau=tau,
            ctx=ctx,
            mask=valid_mask,
            feasible_actions=feasible_actions,
            action_features=action_features,
            comparable_metrics=comparable_metrics,
            dynamic_metrics=dynamic_metrics,
            action_path_type=action_path_type,
            action_constraint_type=action_constraint_type,
            path_polyline_idx=path_polyline_idx,
            map_polylines=map_polylines,
            z=z_dummy,
        )

        if dec0.recog_probs is None or dec0.recog_conf is None or dec0.decision_features is None or dec0.u_ctx is None:
            raise RuntimeError("Decoder must provide recog_probs, recog_conf, decision_features, and u_ctx")

        q_chi = dec0.recog_probs          # [B,N,T,A]
        conf = dec0.recog_conf            # [B,N,T]
        f_m = dec0.decision_features      # [B,N,T,A,Dz]
        u_ctx = dec0.u_ctx                # [B,N,T,A]

        # expected decision features under recognition
        f_bar = (q_chi.unsqueeze(-1) * f_m).sum(dim=-2)  # [B,N,T,Dz]

        # evidence increments
        delta_eta, delta_L, alpha = self.evidence(f_bar=f_bar, tau=tau, ctx=ctx, conf=conf, mask=valid_mask)

        # prior from context (or fixed)
        prior_ctx = self.prior_from_context(tau=tau, ctx=ctx, mask=ctx_mask)

        # posteriors
        post_ctx = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=ctx_mask, prior=prior_ctx)
        post_full = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=valid_mask, prior=prior_ctx)

        # weights / overrides
        w_query = self.lambda_query
        w_mu = self.lambda_distill_mu if lambda_distill_mu is None else float(lambda_distill_mu)
        w_cov = self.lambda_distill_cov if lambda_distill_cov is None else float(lambda_distill_cov)
        w_prior = self.lambda_prior if lambda_prior is None else float(lambda_prior)
        w_con = self.lambda_con if lambda_con is None else float(lambda_con)
        w_u = self.lambda_u_ctx if lambda_u_ctx is None else float(lambda_u_ctx)

        S = self.num_z_samples if n_z_samples is None else int(n_z_samples)
        S = max(1, S)

        # ----------------------
        # Choice-predictive query loss
        # ----------------------
        z_samps = torch.stack([post_ctx.q.rsample() for _ in range(S)], dim=0)  # [S,B,N,Dz]
        logits_s = []
        for s in range(S):
            z = z_samps[s]
            z_e = z.unsqueeze(2).unsqueeze(-2).expand(B, N, T, f_m.shape[-2], self.z_dim)
            logits = (z_e * f_m).sum(dim=-1) + u_ctx
            logits = logits / max(1e-6, self.decoder.policy.beta)

            # feasible action handling consistent with decoder
            if feasible_actions is not None:
                fa = feasible_actions.to(dtype=torch.bool)
                any_fa = fa.any(dim=-1, keepdim=True)
                fa = torch.where(any_fa, fa, torch.ones_like(fa))
                if self.training and self.decoder.feasible_action_soft_penalty_train:
                    logits = logits - (~fa).to(dtype=logits.dtype) * float(self.decoder.feasible_action_penalty)
                else:
                    if self.decoder.feasible_action_hard_mask_eval:
                        logits = logits.masked_fill(~fa, float("-inf"))
                    else:
                        logits = logits - (~fa).to(dtype=logits.dtype) * float(self.decoder.feasible_action_penalty)

            logits_s.append(logits)
        logits_s = torch.stack(logits_s, dim=0)
        log_pi_s = torch.log_softmax(logits_s, dim=-1)

        ce_s = -(q_chi.unsqueeze(0) * log_pi_s).sum(dim=-1)  # [S,B,N,T]
        ce = ce_s.mean(dim=0)  # [B,N,T]
        loss_query = masked_mean(ce, query_mask)

        # ----------------------
        # Distillation to full posterior
        # ----------------------
        loss_distill_mu = masked_mean(((post_ctx.q.mean - post_full.q.mean) ** 2).sum(dim=-1), agent_valid)
        loss_distill_cov = masked_mean((post_ctx.q.logvar - post_full.q.logvar).abs().sum(dim=-1), agent_valid)

        # ----------------------
        # KL(q(z|C) || p(z|C))
        # ----------------------
        kl = post_ctx.q.kl_to(prior_ctx)  # [B,N]
        if free_bits > 0.0:
            # simple scalar free-bits on total KL per agent
            kl = torch.clamp(kl, min=float(free_bits))
        loss_kl = masked_mean(kl, agent_valid)

        # ----------------------
        # u_ctx regularizer (small norm)
        # ----------------------
        if feasible_actions is not None:
            fa = feasible_actions.to(dtype=u_ctx.dtype)
            loss_u = masked_mean((u_ctx**2) * fa, valid_mask.unsqueeze(-1))
        else:
            loss_u = masked_mean(u_ctx**2, valid_mask.unsqueeze(-1))

        # ----------------------
        # ----------------------
        # Invariance regularizer R_inv (symmetric KL between two sub-context posteriors)
        # ----------------------
        if w_con > 0.0:
            c1, c2 = self._sample_evidence_matched_subcontexts(ctx_mask, conf)
            post_1 = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=c1, prior=prior_ctx)
            post_2 = self._posterior_from_mask(delta_eta=delta_eta, delta_L=delta_L, alpha=alpha, mask=c2, prior=prior_ctx)
            sym = self._sym_kl(post_1.q, post_2.q)
            loss_con = masked_mean(sym, agent_valid)
        else:
            loss_con = loss_query.new_zeros(())

        # ----------------------
        # total
        # ----------------------
        loss_total = (
            w_query * loss_query
            + w_mu * loss_distill_mu
            + w_cov * loss_distill_cov
            + w_prior * loss_kl
            + w_con * loss_con
            + w_u * loss_u
        )

        # ----------------------
        # last-token logits (for logging/compat)
        # ----------------------
        with torch.no_grad():
            t_last = T - 1
            z_mean = post_ctx.q.mean
            f_last = f_m[:, :, t_last, :, :]  # [B,N,A,Dz]
            u_last = u_ctx[:, :, t_last, :]   # [B,N,A]
            z_e = z_mean.unsqueeze(-2).expand(B, N, f_last.shape[-2], self.z_dim)
            action_logits_last = (z_e * f_last).sum(dim=-1) + u_last
            action_logits_last = action_logits_last / max(1e-6, self.decoder.policy.beta)

            feas_last = feasible_actions[:, :, t_last, :] if feasible_actions is not None else None
            fam_last = action_family[:, :, t_last, :] if action_family is not None else None

            if fam_last is not None:
                maneuver_logits_last = self.decoder.aggregate_family_logits(
                    action_logits_last.unsqueeze(2),
                    action_family=fam_last.unsqueeze(2),
                    feasible_actions=feas_last.unsqueeze(2) if feas_last is not None else None,
                    num_families=self.num_maneuvers,
                ).squeeze(2)
            else:
                maneuver_logits_last = action_logits_last.new_zeros((B, N, self.num_maneuvers))

        return PreferenceCompletionOutput(
            post_full=post_full,
            post_ctx=post_ctx,
            ctx_mask=ctx_mask,
            query_mask=query_mask,
            maneuver_logits_last=maneuver_logits_last,
            action_logits_last=action_logits_last,
            action_family_last=fam_last,
            loss_total=loss_total,
            loss_query_nll=loss_query,
            loss_distill_mu=loss_distill_mu,
            loss_distill_cov=loss_distill_cov,
            loss_kl_ctx_prior=loss_kl,
            loss_contrastive=loss_con,
            loss_overlap=loss_total.new_zeros(()),
            loss_modulation=loss_total.new_zeros(()),
            loss_u_ctx=loss_u,
        )
