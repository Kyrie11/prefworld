from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.data.labels import NUM_MANEUVERS


@dataclass
class PrimitiveDecodeOutput:
    """Per-token primitive decode outputs."""

    maneuver_logits: torch.Tensor  # [B,N,T,M]
    logp_x_given_m: torch.Tensor   # [B,N,T,M]
    z_mod: torch.Tensor            # [B,N,T,Dz]
    z_mod_delta: torch.Tensor      # [B,N,T,Dz]


class LowRankTemplateModulation(nn.Module):
    """Low-rank template-conditioned modulation in latent preference space.

    Paper Eq.(7):
        \tilde z = z + A(h_\tau, c) B z,  rank(A B) << d_z

    We implement a practical version without explicit c_i (context id), using only h_\tau.
    The low-rank form is enforced by mapping z -> r then back to d_z with a tau-conditioned matrix.

    Notes:
      - This module is meant to be *weak*; regularize ||\tilde z - z||.
      - We modulate the *sampled z* (or mean), and keep the base z coordinate system shared.
    """

    def __init__(
        self,
        *,
        z_dim: int,
        tau_dim: int,
        rank: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.tau_dim = int(tau_dim)
        self.rank = int(rank)

        # B: z -> r
        self.B = nn.Linear(self.z_dim, self.rank, bias=False)

        # A_theta: tau -> (d_z x r)
        self.A = nn.Sequential(
            nn.Linear(self.tau_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.z_dim * self.rank),
        )

        # Initialize near-zero modulation
        nn.init.zeros_(self.A[-1].weight)
        nn.init.zeros_(self.A[-1].bias)

    def forward(self, z: torch.Tensor, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute modulated preference.

        Args:
          z:   [B,N,Dz]
          tau: [B,N,T,Dt] or [B,N,Dt]
        Returns:
          z_mod:       [B,N,T,Dz] (or [B,N,1,Dz] if tau is [B,N,Dt])
          z_mod_delta: [B,N,T,Dz]
        """
        if tau.dim() == 3:
            tau_e = tau.unsqueeze(2)  # [B,N,1,Dt]
        elif tau.dim() == 4:
            tau_e = tau
        else:
            raise ValueError(f"tau must be [B,N,Dt] or [B,N,T,Dt], got {tau.shape}")

        B, N, T, Dt = tau_e.shape
        assert Dt == self.tau_dim
        Dz = z.shape[-1]
        assert Dz == self.z_dim

        # z -> r
        bz = self.B(z)  # [B,N,r]
        bz = bz.unsqueeze(2).expand(B, N, T, self.rank)  # [B,N,T,r]

        # tau -> A matrices
        A = self.A(tau_e).view(B, N, T, self.z_dim, self.rank)  # [B,N,T,Dz,r]

        delta = torch.einsum("bntdr,bntr->bntd", A, bz)
        z_base = z.unsqueeze(2).expand(B, N, T, self.z_dim)
        return z_base + delta, delta


class RandomUtilityManeuverPolicy(nn.Module):
    """Random-utility maneuver model (paper Eq.(8)-(9)).

      U(m; z, tau) = z^T Phi_int(m, tau) + u_ctx(m, tau)
      p(m|z,tau) ∝ exp(U/β)

    We implement Phi_int and u_ctx with small MLPs on (tau, maneuver_emb).
    """

    def __init__(
        self,
        *,
        tau_dim: int,
        z_dim: int,
        num_maneuvers: int,
        m_emb_dim: int = 16,
        hidden_dim: int = 64,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.tau_dim = int(tau_dim)
        self.z_dim = int(z_dim)
        self.num_maneuvers = int(num_maneuvers)
        self.beta = float(beta)

        self.m_emb = nn.Embedding(self.num_maneuvers, int(m_emb_dim))

        in_dim = self.tau_dim + int(m_emb_dim)
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.z_dim),
        )

        # low-capacity context-only baseline
        self.u_ctx = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # initialize near 0
        nn.init.zeros_(self.phi[-1].weight)
        nn.init.zeros_(self.phi[-1].bias)
        nn.init.zeros_(self.u_ctx[-1].weight)
        nn.init.zeros_(self.u_ctx[-1].bias)

    def forward(self, z_mod: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Return maneuver logits per token.

        Args:
          z_mod: [B,N,T,Dz]
          tau:   [B,N,T,Dt]
        Returns:
          logits: [B,N,T,M]
        """
        B, N, T, Dz = z_mod.shape
        assert Dz == self.z_dim
        assert tau.shape[:3] == (B, N, T)

        m_ids = torch.arange(self.num_maneuvers, device=z_mod.device)
        m_emb = self.m_emb(m_ids)  # [M,Dm]

        tau_e = tau.unsqueeze(-2).expand(B, N, T, self.num_maneuvers, self.tau_dim)
        m_e = m_emb.view(1, 1, 1, self.num_maneuvers, -1).expand(B, N, T, self.num_maneuvers, -1)
        inp = torch.cat([tau_e, m_e], dim=-1)  # [B,N,T,M,*]

        phi = self.phi(inp)  # [B,N,T,M,Dz]
        u = self.u_ctx(inp).squeeze(-1)  # [B,N,T,M]

        # dot product preference term
        logits = (z_mod.unsqueeze(-2) * phi).sum(dim=-1) + u
        logits = logits / max(1e-6, self.beta)
        return logits


class TokenEmissionDecoder(nn.Module):
    """Per-token emission model p(x_t | m, z, tau, ctx).

    This is a lightweight surrogate for the paper's primitive likelihood with xi marginalization.
    It keeps the API compatible with latent-maneuver marginalization in PreferenceCompletion.

    You can replace this module with a Frenet primitive likelihood later without changing
    the preference-completion training code.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int,
        num_maneuvers: int,
        hidden_dim: int = 128,
        min_logstd: float = -5.0,
        max_logstd: float = 2.0,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.tau_dim = int(tau_dim)
        self.ctx_dim = int(ctx_dim)
        self.z_dim = int(z_dim)
        self.num_maneuvers = int(num_maneuvers)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)

        in_dim = self.x_dim * 0 + self.tau_dim + self.ctx_dim + self.z_dim  # (no x in params)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.to_mean = nn.Linear(hidden_dim, self.num_maneuvers * self.x_dim)
        self.to_logstd = nn.Linear(hidden_dim, self.num_maneuvers * self.x_dim)

        nn.init.zeros_(self.to_mean.weight)
        nn.init.zeros_(self.to_mean.bias)
        nn.init.zeros_(self.to_logstd.weight)
        nn.init.constant_(self.to_logstd.bias, -1.0)

    def forward(self, *, z_mod: torch.Tensor, tau: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-maneuver Gaussian parameters for each token.

        Args:
          z_mod: [B,N,T,Dz]
          tau:   [B,N,T,Dt]
          ctx:   [B,N,T,Dc]
        Returns:
          mean:   [B,N,T,M,Dx]
          logstd: [B,N,T,M,Dx]
        """
        h = self.trunk(torch.cat([z_mod, tau, ctx], dim=-1))
        B, N, T, _ = h.shape
        mean = self.to_mean(h).view(B, N, T, self.num_maneuvers, self.x_dim)
        logstd = self.to_logstd(h).view(B, N, T, self.num_maneuvers, self.x_dim)
        logstd = logstd.clamp(min=self.min_logstd, max=self.max_logstd)
        return mean, logstd


class FrenetMarginalEmission(nn.Module):
    """Template-conditioned Frenet-like motion primitives with analytic xi marginalization.

    This is a *paper-aligned* implementation of p(x_t | m, τ_t) with a continuous
    primitive parameter xi marginalized out (analytic linear-Gaussian case).

    We model one-step action tokens x = [Δx, Δy, Δyaw]. We rotate (Δx,Δy) into the
    agent heading frame defined by ctx yaw, yielding x' = [Δs, Δd, Δyaw].

    For each maneuver m we use:
      x' = b(ctx) + A_m xi + ε
      xi ~ N(μ_m(τ,ctx), diag(σ^2_xi,m(τ,ctx)))
      ε  ~ N(0, diag(σ^2_ε,m(τ,ctx)))

    Integrating out xi yields a diagonal Gaussian with
      mean = b + A_m μ_m
      var  = σ^2_ε,m + diag(A_m^2) σ^2_xi,m

    Notes:
      - We keep xi dimension fixed at 2: [a_s, aux].
        aux is interpreted as lateral acceleration (lane-change) or yaw-rate (turn).
      - This keeps training stable while matching the paper's discrete+continuous primitive form.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int = 0,
        num_maneuvers: int,
        hidden_dim: int = 128,
        dt: float = 0.1,
        min_logstd: float = -5.0,
        max_logstd: float = 2.0,
    ) -> None:
        super().__init__()
        if int(x_dim) != 3:
            raise ValueError("FrenetMarginalEmission expects x_dim=3 for [dx,dy,dyaw].")
        self.x_dim = int(x_dim)
        self.tau_dim = int(tau_dim)
        self.ctx_dim = int(ctx_dim)
        self.z_dim = int(z_dim)
        self.num_maneuvers = int(num_maneuvers)
        self.dt = float(dt)
        self.min_logstd = float(min_logstd)
        self.max_logstd = float(max_logstd)

        in_dim = self.tau_dim + self.ctx_dim + (self.z_dim if self.z_dim > 0 else 0)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # token-conditioned deltas for primitive parameters
        self.to_dmu = nn.Linear(hidden_dim, self.num_maneuvers * 2)
        self.to_dlogvar = nn.Linear(hidden_dim, self.num_maneuvers * 2)
        self.to_dlogstd_noise = nn.Linear(hidden_dim, self.num_maneuvers * 3)

        # initialize deltas near 0
        nn.init.zeros_(self.to_dmu.weight)
        nn.init.zeros_(self.to_dmu.bias)
        nn.init.zeros_(self.to_dlogvar.weight)
        nn.init.zeros_(self.to_dlogvar.bias)
        nn.init.zeros_(self.to_dlogstd_noise.weight)
        nn.init.zeros_(self.to_dlogstd_noise.bias)

        # Base priors (heuristic init aligned with MANEUVER_NAMES order)
        mu0 = torch.zeros((self.num_maneuvers, 2))
        if self.num_maneuvers >= 6:
            # [keep_lane, lane_change_left, lane_change_right, turn_left, turn_right, stop]
            mu0[0] = torch.tensor([0.0, 0.0])
            mu0[1] = torch.tensor([0.0, 1.0])
            mu0[2] = torch.tensor([0.0, -1.0])
            mu0[3] = torch.tensor([0.0, 0.4])
            mu0[4] = torch.tensor([0.0, -0.4])
            mu0[5] = torch.tensor([-2.0, 0.0])
        self.mu_xi_base = nn.Parameter(mu0)
        self.logvar_xi_base = nn.Parameter(torch.full((self.num_maneuvers, 2), -0.5))
        self.logstd_noise_base = nn.Parameter(torch.full((self.num_maneuvers, 3), -1.0))

        # Maneuver-type masks (for xi[1] contribution)
        lane_mask = torch.zeros((self.num_maneuvers,), dtype=torch.float32)
        turn_mask = torch.zeros((self.num_maneuvers,), dtype=torch.float32)
        if self.num_maneuvers >= 6:
            lane_mask[1] = 1.0
            lane_mask[2] = 1.0
            turn_mask[3] = 1.0
            turn_mask[4] = 1.0
        self.register_buffer("lane_change_mask", lane_mask)
        self.register_buffer("turn_mask", turn_mask)

    @staticmethod
    def _rotate_to_heading(x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        s = cy * x + sy * y
        d = -sy * x + cy * y
        return s, d

    @staticmethod
    def _log_prob_diag_normal(x: torch.Tensor, mean: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        var = torch.exp(2.0 * logstd)
        return -0.5 * (((x - mean) ** 2) / (var + 1e-8) + 2.0 * logstd + math.log(2.0 * math.pi)).sum(dim=-1)

    def log_prob(
        self,
        *,
        x: torch.Tensor,
        tau: torch.Tensor,
        ctx: torch.Tensor,
        z_mod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return log p(x | m, τ, ctx) for each maneuver.

        Args:
          x:   [B,N,T,3] action tokens (dx,dy,dyaw) in ego frame
          tau: [B,N,T,Dt]
          ctx: [B,N,T,Dc] where ctx[...,2]=yaw, ctx[...,3:5]=vx,vy
          z_mod: [B,N,T,Dz] (optional). If provided and z_dim>0, we condition primitive
            parameters on preferences as in the paper p(xi | m, z, tau).
        Returns:
          logp: [B,N,T,M]
        """
        B, N, T, Dx = x.shape
        if Dx != 3:
            raise ValueError("x must be [...,3] for [dx,dy,dyaw]")

        yaw = ctx[..., 2]
        dx, dy, dyaw = x[..., 0], x[..., 1], x[..., 2]
        ds_obs, dd_obs = self._rotate_to_heading(dx, dy, yaw)

        vx, vy = ctx[..., 3], ctx[..., 4]
        vs, vd = self._rotate_to_heading(vx, vy, yaw)

        dt = self.dt
        c_acc = 0.5 * (dt**2)
        c_yaw = dt

        # base increment b(ctx)
        b_ds = vs * dt
        b_dd = vd * dt

        if self.z_dim > 0:
            if z_mod is None:
                raise ValueError("FrenetMarginalEmission requires z_mod when z_dim>0")
            if z_mod.shape[:3] != tau.shape[:3] or z_mod.shape[-1] != self.z_dim:
                raise ValueError(f"z_mod must have shape [B,N,T,{self.z_dim}], got {tuple(z_mod.shape)}")
            h_in = torch.cat([tau, ctx, z_mod], dim=-1)
        else:
            h_in = torch.cat([tau, ctx], dim=-1)
        h = self.trunk(h_in)
        dmu = self.to_dmu(h).view(B, N, T, self.num_maneuvers, 2)
        dlogvar = self.to_dlogvar(h).view(B, N, T, self.num_maneuvers, 2)
        dlogstd_noise = self.to_dlogstd_noise(h).view(B, N, T, self.num_maneuvers, 3)

        mu_xi = self.mu_xi_base.view(1, 1, 1, self.num_maneuvers, 2) + dmu
        logvar_xi = self.logvar_xi_base.view(1, 1, 1, self.num_maneuvers, 2) + dlogvar
        logvar_xi = logvar_xi.clamp(min=-10.0, max=10.0)
        logstd_noise = (self.logstd_noise_base.view(1, 1, 1, self.num_maneuvers, 3) + dlogstd_noise).clamp(
            min=self.min_logstd, max=self.max_logstd
        )

        var_xi = torch.exp(logvar_xi)
        var_noise = torch.exp(2.0 * logstd_noise)

        lane = self.lane_change_mask.view(1, 1, 1, self.num_maneuvers)
        turn = self.turn_mask.view(1, 1, 1, self.num_maneuvers)

        mean_ds = b_ds.unsqueeze(-1) + c_acc * mu_xi[..., 0]
        mean_dd = b_dd.unsqueeze(-1) + lane * (c_acc * mu_xi[..., 1])
        mean_dyaw = turn * (c_yaw * mu_xi[..., 1])

        var_ds = var_noise[..., 0] + (c_acc**2) * var_xi[..., 0]
        var_dd = var_noise[..., 1] + lane * ((c_acc**2) * var_xi[..., 1])
        var_dyaw = var_noise[..., 2] + turn * ((c_yaw**2) * var_xi[..., 1])

        var = torch.stack([var_ds, var_dd, var_dyaw], dim=-1).clamp_min(1e-6)
        mean = torch.stack([mean_ds, mean_dd, mean_dyaw], dim=-1)
        logstd = 0.5 * torch.log(var)

        x_obs = torch.stack([ds_obs, dd_obs, dyaw], dim=-1).unsqueeze(-2)  # [B,N,T,1,3]
        return self._log_prob_diag_normal(x_obs, mean, logstd)  # [B,N,T,M]


class MotionPrimitiveDecoder(nn.Module):
    """Maneuver + token emission decoder used by preference completion.

    The paper specifies:
      - random-utility maneuver model p(m|z, tau)
      - continuous parameters xi and a template-conditioned primitive likelihood p(X|m,xi,tau)

    This repo implements a faithful *interface*:
      - p(m|z,tau) via a random-utility model in latent space
      - p(x|m,z,tau,ctx) as a small diagonal-Gaussian emission model

    The emission can be swapped for a full Frenet primitive likelihood without touching
    PreferenceCompletion (stage-1 training).
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int,
        num_maneuvers: int = NUM_MANEUVERS,
        hidden_dim: int = 128,
        mod_rank: int = 4,
        mod_hidden: int = 64,
        utility_hidden: int = 64,
        beta: float = 1.0,
        # Feasible-action handling
        feasible_action_penalty: float = 5.0,
        feasible_action_soft_penalty_train: bool = True,
        feasible_action_hard_mask_eval: bool = True,
        emission_type: str = "frenet",
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.tau_dim = int(tau_dim)
        self.ctx_dim = int(ctx_dim)
        self.z_dim = int(z_dim)
        self.num_maneuvers = int(num_maneuvers)

        self.feasible_action_penalty = float(feasible_action_penalty)
        self.feasible_action_soft_penalty_train = bool(feasible_action_soft_penalty_train)
        self.feasible_action_hard_mask_eval = bool(feasible_action_hard_mask_eval)

        self.modulation = LowRankTemplateModulation(z_dim=z_dim, tau_dim=tau_dim, rank=mod_rank, hidden_dim=mod_hidden)
        self.policy = RandomUtilityManeuverPolicy(
            tau_dim=tau_dim,
            z_dim=z_dim,
            num_maneuvers=num_maneuvers,
            hidden_dim=utility_hidden,
            beta=beta,
        )
        etype = str(emission_type).lower()
        if etype in ("frenet", "primitive", "marginal_frenet"):
            self.emission: nn.Module = FrenetMarginalEmission(
                x_dim=x_dim,
                tau_dim=tau_dim,
                ctx_dim=ctx_dim,
                z_dim=z_dim,
                num_maneuvers=num_maneuvers,
                hidden_dim=hidden_dim,
                dt=float(dt),
            )
        elif etype in ("gaussian", "mlp", "surrogate"):
            self.emission = TokenEmissionDecoder(
                x_dim=x_dim,
                tau_dim=tau_dim,
                ctx_dim=ctx_dim,
                z_dim=z_dim,
                num_maneuvers=num_maneuvers,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown emission_type={emission_type!r}. Use 'frenet' or 'gaussian'.")

    @staticmethod
    def _log_prob_diag_normal(x: torch.Tensor, mean: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        # x: [...,Dx], mean/logstd: [...,Dx]
        var = torch.exp(2.0 * logstd)
        return -0.5 * (((x - mean) ** 2) / (var + 1e-8) + 2.0 * logstd + math.log(2.0 * math.pi)).sum(dim=-1)

    def token_log_prob(
        self,
        *,
        x: torch.Tensor,          # [B,N,T,Dx]
        tau: torch.Tensor,        # [B,N,T,Dt]
        ctx: torch.Tensor,        # [B,N,T,Dc]
        mask: torch.Tensor,       # [B,N,T]
        feasible_actions: Optional[torch.Tensor] = None,  # [B,N,T,M] bool
        z: torch.Tensor,          # [B,N,Dz]
    ) -> PrimitiveDecodeOutput:
        """Return per-token maneuver logits and log-likelihood for each maneuver."""
        m = mask > 0.5

        z_mod, z_delta = self.modulation(z, tau)
        logits = self.policy(z_mod, tau)  # [B,N,T,M]

        # ------------------------------------------------------------------
        # Feasible-action masking (paper: m \in A_{i,t})
        # ------------------------------------------------------------------
        if feasible_actions is not None:
            fa = feasible_actions.to(dtype=torch.bool)
            if fa.shape != logits.shape:
                raise ValueError(f"feasible_actions must have shape {tuple(logits.shape)}, got {tuple(fa.shape)}")

            # If a token has no feasible maneuver (rare edge case), fall back to all-feasible
            # to avoid NaNs in log-softmax.
            any_feas = fa.any(dim=-1, keepdim=True)
            fa = torch.where(any_feas, fa, torch.ones_like(fa))

            if self.training and self.feasible_action_soft_penalty_train:
                # Soft penalty during training to avoid brittle masking from imperfect map inference.
                # Infeasible maneuvers are discouraged but not impossible.
                penalty = float(self.feasible_action_penalty)
                logits = logits - (~fa).to(dtype=logits.dtype) * penalty
            else:
                # Hard mask at evaluation / planning time.
                if self.feasible_action_hard_mask_eval:
                    logits = logits.masked_fill(~fa, float("-inf"))
                else:
                    penalty = float(self.feasible_action_penalty)
                    logits = logits - (~fa).to(dtype=logits.dtype) * penalty

        if isinstance(self.emission, FrenetMarginalEmission):
            logp = self.emission.log_prob(x=x, tau=tau, ctx=ctx, z_mod=z_mod)  # [B,N,T,M]
        else:
            mean, logstd = self.emission(z_mod=z_mod, tau=tau, ctx=ctx)  # [B,N,T,M,Dx]
            x_e = x.unsqueeze(-2)  # [B,N,T,1,Dx]
            logp = self._log_prob_diag_normal(x_e, mean, logstd)  # [B,N,T,M]

        # mask invalid tokens
        m_e = m.unsqueeze(-1)

        # 先把 logp 里的非有限数清掉（可选但非常稳）
        logp = torch.nan_to_num(logp, nan=0.0, posinf=0.0, neginf=0.0)

        # 用 masked_fill 而不是乘 mask（NaN*0 仍是 NaN）
        logp = logp.masked_fill(~m_e, 0.0)

        # logits 这行保留
        logits = logits.masked_fill(~m_e, 0.0)

        # z_mod / z_delta 同理
        z_mod = z_mod.masked_fill(~m_e, 0.0)
        z_delta = z_delta.masked_fill(~m_e, 0.0)

        return PrimitiveDecodeOutput(maneuver_logits=logits, logp_x_given_m=logp, z_mod=z_mod, z_mod_delta=z_delta)

    @torch.no_grad()
    def maneuver_logits_last(
        self,
        *,
        z: torch.Tensor,         # [B,N,Dz]
        tau_last: torch.Tensor,  # [B,N,Dt]
        ctx_last: torch.Tensor,  # [B,N,Dc]
        feasible_actions_last: Optional[torch.Tensor] = None,  # [B,N,M]
    ) -> torch.Tensor:
        """Convenience: maneuver logits for the last token (T=1)."""
        tau = tau_last.unsqueeze(2)  # [B,N,1,Dt]
        ctx = ctx_last.unsqueeze(2)  # [B,N,1,Dc]
        # dummy x/mask; only logits are used
        dummy_x = torch.zeros((z.shape[0], z.shape[1], 1, self.x_dim), device=z.device, dtype=z.dtype)
        mask = torch.ones((z.shape[0], z.shape[1], 1), device=z.device, dtype=z.dtype)
        fa = None
        if feasible_actions_last is not None:
            fa = feasible_actions_last.unsqueeze(2)  # [B,N,1,M]
        out = self.token_log_prob(x=dummy_x, tau=tau, ctx=ctx, mask=mask, feasible_actions=fa, z=z)
        return out.maneuver_logits.squeeze(2)
