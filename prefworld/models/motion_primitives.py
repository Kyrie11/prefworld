from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.data.labels import (
    LonConstraint,
    Maneuver,
    NUM_MANEUVERS,
    NUM_LON_CONSTRAINTS,
    NUM_PATH_TYPES,
    PathType,
    path_constraint_to_maneuver,
)


@dataclass
class PrimitiveDecodeOutput:
    """Per-token primitive decode outputs.

    Notes
    -----
    * ``maneuver_logits`` is kept for backward compatibility, but it contains **logits
      over structured action slots** a=(path branch, longitudinal constraint source).
    * ``logp_x_given_m`` is the motion/kinematics likelihood term p_n(X|m,xi,\tau^{det})
      marginalized by a lightweight MAP approximation. Importantly, it is **independent
      of z** (paper Sec.\ref{sec:primitive_likelihood}).
    * ``z_mod``/``z_mod_delta`` are kept to avoid touching downstream code. In the paper
      design preferences are episode-stationary (no template-conditioned modulation), so
      we set z_mod=z broadcast over time and z_mod_delta=0.
    """

    maneuver_logits: torch.Tensor  # [B,N,T,A] structured action-slot logits (policy π)
    logp_x_given_m: torch.Tensor   # [B,N,T,A] motion token log-likelihood per slot

    # kept for compatibility with older losses / logging
    z_mod: torch.Tensor            # [B,N,T,Dz]
    z_mod_delta: torch.Tensor      # [B,N,T,Dz]

    # optional diagnostics for paper-style regularizers
    u_ctx: Optional[torch.Tensor] = None        # [B,N,T,A]
    decision_features: Optional[torch.Tensor] = None  # [B,N,T,A,Dz]
    recog_probs: Optional[torch.Tensor] = None  # [B,N,T,A]
    recog_conf: Optional[torch.Tensor] = None   # [B,N,T]


# ------------------------------
# Geometry helpers
# ------------------------------

def _smoothstep(u: torch.Tensor) -> torch.Tensor:
    """Quintic smoothstep σ(u)=10u^3-15u^4+6u^5, with u assumed in [0,1]."""
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _project_to_polyline(
    pts: torch.Tensor,          # [...,2]
    poly: torch.Tensor,         # [...,L,2]
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project points onto a polyline and return (s, d, tangent).

    This is a differentiable-ish (piecewise) projection that chooses the nearest segment.

    Returns
    -------
    s : [...]
        Arc-length coordinate along the polyline.
    d : [...]
        Signed lateral offset (left positive w.r.t. tangent direction).
    tan : [...,2]
        Unit tangent direction of the closest segment.
    """
    if pts.shape[-1] != 2 or poly.shape[-1] != 2:
        raise ValueError("pts/poly must end with 2")
    if poly.shape[-2] < 2:
        s = torch.zeros_like(pts[..., 0])
        d = torch.zeros_like(pts[..., 0])
        tan = torch.zeros_like(pts)
        tan[..., 0] = 1.0
        return s, d, tan

    p0 = poly[..., :-1, :]  # [...,L-1,2]
    v = poly[..., 1:, :] - poly[..., :-1, :]  # [...,L-1,2]
    v2 = (v * v).sum(dim=-1).clamp_min(eps)   # [...,L-1]
    seg_len = torch.sqrt(v2)

    # cumulative length before each segment
    cum = torch.cumsum(seg_len, dim=-1)  # [...,L-1]
    cum0 = torch.cat([torch.zeros_like(cum[..., :1]), cum[..., :-1]], dim=-1)

    w = pts.unsqueeze(-2) - p0  # [...,L-1,2]
    t = (w * v).sum(dim=-1) / v2
    t = t.clamp(0.0, 1.0)
    proj = p0 + t.unsqueeze(-1) * v
    diff = pts.unsqueeze(-2) - proj
    dist2 = (diff * diff).sum(dim=-1)

    idx = dist2.argmin(dim=-1)  # [...]

    # gather per-point segment quantities
    idx1 = idx.unsqueeze(-1)  # [...,1]
    t_sel = torch.gather(t, dim=-1, index=idx1).squeeze(-1)
    len_sel = torch.gather(seg_len, dim=-1, index=idx1).squeeze(-1)
    cum_sel = torch.gather(cum0, dim=-1, index=idx1).squeeze(-1)
    dist2_sel = torch.gather(dist2, dim=-1, index=idx1).squeeze(-1)

    idx2 = idx1.unsqueeze(-1)
    v_sel = torch.gather(v, dim=-2, index=idx2.expand(*v.shape[:-2], 1, 2)).squeeze(-2)
    diff_sel = torch.gather(diff, dim=-2, index=idx2.expand(*diff.shape[:-2], 1, 2)).squeeze(-2)
    s = cum_sel + t_sel * len_sel

    cross = v_sel[..., 0] * diff_sel[..., 1] - v_sel[..., 1] * diff_sel[..., 0]
    sign = torch.sign(cross)
    sign = torch.where(sign == 0.0, torch.ones_like(sign), sign)
    d = sign * torch.sqrt(dist2_sel.clamp_min(eps))

    tan = v_sel / len_sel.unsqueeze(-1).clamp_min(eps)
    return s, d, tan


def _gather_polylines(
    map_polylines: torch.Tensor,   # [B,M,L,2]
    idx: torch.Tensor,             # [B,*,] int64
) -> torch.Tensor:
    """Gather polylines along M using advanced indexing."""
    B, M, L, _ = map_polylines.shape
    if idx.dtype != torch.long:
        idx = idx.long()
    flat = idx.reshape(B, -1)
    b = torch.arange(B, device=map_polylines.device).view(B, 1).expand_as(flat)
    gathered = map_polylines[b, flat]  # [B,*,L,2]
    return gathered.view(*idx.shape, L, 2)


# ------------------------------
# Paper-style random utility policy π(m | z, τ^{det})
# ------------------------------


class PaperStructuredActionUtility(nn.Module):
    """Random-utility policy over structured action slots.

    Implements the paper form:

        U(m; z, τ^{det}) = z^T f(m, τ^{det}) + u_ctx(m, τ^{det})

    where f is a vector of physically-comparable features (dim=z_dim), and u_ctx is a
    low-capacity preference-independent baseline encoding hard rules/feasibility.

    We avoid any template-conditioned modulation of z.
    """

    def __init__(
        self,
        *,
        z_dim: int,
        beta: float = 1.0,
        ctx_in_dim: int = 3,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.beta = float(beta)

        # low-capacity u_ctx: linear in a small hand-designed context feature vector.
        self.u_ctx_w = nn.Parameter(torch.zeros(ctx_in_dim))
        self.u_ctx_b = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        *,
        z: torch.Tensor,                 # [B,N,Dz]
        decision_features: torch.Tensor,  # [B,N,T,A,Dz]
        ctx_features: torch.Tensor,       # [B,N,T,A,Dc]
        feasible_actions: Optional[torch.Tensor] = None,  # [B,N,T,A]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"z last dim must be {self.z_dim}, got {z.shape[-1]}")
        if decision_features.shape[-1] != self.z_dim:
            raise ValueError(f"decision_features last dim must be {self.z_dim}, got {decision_features.shape[-1]}")

        B, N, T, A, _ = decision_features.shape
        z_e = z.unsqueeze(2).unsqueeze(-2).expand(B, N, T, A, self.z_dim)

        # u_ctx baseline
        u = (ctx_features * self.u_ctx_w.view(1, 1, 1, 1, -1)).sum(dim=-1) + self.u_ctx_b

        # enforce zero-mean u_ctx within each (B,N,T) template family, over feasible actions
        if feasible_actions is not None:
            fa = feasible_actions.to(dtype=torch.bool)
            any_fa = fa.any(dim=-1, keepdim=True)
            fa = torch.where(any_fa, fa, torch.ones_like(fa))
            u_mean = (u.masked_fill(~fa, 0.0).sum(dim=-1, keepdim=True) / fa.to(dtype=u.dtype).sum(dim=-1, keepdim=True).clamp_min(1e-6))
            u = u - u_mean
        else:
            u = u - u.mean(dim=-1, keepdim=True)

        logits = (z_e * decision_features).sum(dim=-1) + u
        logits = logits / max(1e-6, self.beta)
        return logits, u


# ------------------------------
# Paper-style primitive likelihood p_n(X | m, ξ, τ^{det}) (MAP approximation)
# ------------------------------


class TemplatePrimitiveLikelihood(nn.Module):
    """Lightweight template-conditioned primitive likelihood.

    This is a pragmatic, *deterministic* instantiation of Appendix Sec.\ref{sec:primitive_likelihood}.

    - We compute action-conditioned residuals in a Frenet-like frame defined by the selected
      reference path branch.
    - We use fixed maneuver-dependent weights w_m and diagonal noise scales.
    - Continuous primitive parameters ξ are approximated by a small MAP proxy:
        * lane-change splice parameter ρ is selected by a short grid search
        * δd and a_adj are not explicitly optimized here (we use δd=0 and a_adj absorbed
          into the baseline acceleration heuristic)

    The goal is to supply a stable recognition signal q_χ(m|X,τ^{det}) without allowing
    z → motion leakage.
    """

    def __init__(
        self,
        *,
        dt: float = 0.1,
        lane_width: float = 3.6,
        topo_horizon_m: float = 50.0,
        conflict_time_s: float = 5.0,
        a_min: float = -4.0,
        a_max: float = 2.0,
        k_yield: float = 1.5,
        rho_grid: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
        L_prep: float = 10.0,
        L_lc: float = 30.0,
        # base noise scales for residual components [s, d, v, a]
        sigma_s: float = 1.0,
        sigma_d: float = 0.5,
        sigma_v: float = 1.0,
        sigma_a: float = 1.5,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.lane_width = float(lane_width)
        self.topo_horizon_m = float(topo_horizon_m)
        self.conflict_time_s = float(conflict_time_s)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.k_yield = float(k_yield)
        self.rho_grid = tuple(float(r) for r in rho_grid)
        self.L_prep = float(L_prep)
        self.L_lc = float(L_lc)

        self.sigma = torch.tensor([sigma_s, sigma_d, sigma_v, sigma_a], dtype=torch.float32)

        # Fixed semantic weights w_m (Appendix "Fixed primitive weights")
        # Each is [w_s, w_d, w_v, w_a].
        self.register_buffer(
            "w_by_family",
            torch.tensor(
                [
                    # KEEP
                    [1.2, 0.2, 1.0, 0.6],
                    # LCL
                    [0.9, 1.2, 0.9, 0.6],
                    # LCR
                    [0.9, 1.2, 0.9, 0.6],
                    # TURN_LEFT
                    [1.0, 0.4, 0.9, 0.6],
                    # TURN_RIGHT
                    [1.0, 0.4, 0.9, 0.6],
                    # STOP
                    [1.1, 0.2, 1.0, 0.2],
                ],
                dtype=torch.float32,
            ),
        )

    def _baseline_acc(
        self,
        *,
        speed: torch.Tensor,                # [B,N,T]
        constraint_type: torch.Tensor,       # [A] (int)
        comparable_metrics: torch.Tensor,    # [B,N,T,A,C]
    ) -> torch.Tensor:
        """Low-capacity heuristic acceleration baseline \bar a(K_dyn).

        This is intentionally simple and preference-independent.
        """
        # constraint_type is per-slot constant, broadcast
        B, N, T, A, _ = comparable_metrics.shape
        c = constraint_type.view(1, 1, 1, A).expand(B, N, T, A)

        speed_e = speed.unsqueeze(-1).expand(B, N, T, A)

        # gap in meters (only meaningful for Follow/Yield slots as encoded by extractor)
        gap = comparable_metrics[..., 1] * self.topo_horizon_m
        ttc = comparable_metrics[..., 2] * self.conflict_time_s

        a = torch.zeros_like(gap)

        # STOP_LINE: decelerate if moving
        stop_mask = c == int(LonConstraint.STOP_LINE)
        a_stop = torch.where(speed_e > 0.5, -1.5 * torch.ones_like(a), torch.zeros_like(a))
        a = torch.where(stop_mask, a_stop, a)

        # FOLLOW: proportional gap control
        follow_mask = c == int(LonConstraint.FOLLOW)
        desired = 1.5 * speed_e + 2.0
        a_follow = 0.3 * (gap - desired)
        a_follow = a_follow.clamp(min=self.a_min, max=self.a_max)
        a = torch.where(follow_mask, a_follow, a)

        # YIELD_TO: decelerate if TTC is small
        yield_mask = c == int(LonConstraint.YIELD_TO)
        a_y = torch.where(ttc < 2.0, -self.k_yield * torch.ones_like(a), torch.zeros_like(a))
        a = torch.where(yield_mask, a_y, a)

        return a.clamp(min=self.a_min, max=self.a_max)

    def _lane_change_splice(
        self,
        *,
        keep_poly: torch.Tensor,     # [B,N,T,L,2]
        tgt_poly: torch.Tensor,      # [B,N,T,L,2]
        s0_keep: torch.Tensor,       # [B,N,T]
        rho: float,
    ) -> torch.Tensor:
        """Blend keep → target lane using Appendix splice rule (approx in Cartesian)."""
        # arc-length along keep points
        seg = keep_poly[..., 1:, :] - keep_poly[..., :-1, :]
        seg_len = torch.sqrt((seg * seg).sum(dim=-1).clamp_min(1e-8))
        s_pts = torch.cumsum(seg_len, dim=-1)
        s_pts = torch.cat([torch.zeros_like(s_pts[..., :1]), s_pts], dim=-1)  # [B,N,T,L]

        s_start = s0_keep + float(rho) * self.L_prep
        u = ((s_pts - s_start.unsqueeze(-1)) / max(1e-6, self.L_lc)).clamp(0.0, 1.0)
        alpha = _smoothstep(u)
        # Cartesian blend (pragmatic)
        return keep_poly * (1.0 - alpha.unsqueeze(-1)) + tgt_poly * alpha.unsqueeze(-1)

    def _log_prob_diag(self, e: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute log N(e; 0, diag(sigma^2 / w)).

        e: [...,4], w: [...,4]
        """
        sigma = self.sigma.to(device=e.device, dtype=e.dtype)
        var = (sigma**2).view(1, 1, 1, 1, 4) / w.clamp_min(1e-6)
        log_det = torch.log(var.clamp_min(1e-12)).sum(dim=-1)
        quad = (e**2 / var.clamp_min(1e-12)).sum(dim=-1)
        return -0.5 * (quad + log_det + 4.0 * math.log(2.0 * math.pi))

    def forward(
        self,
        *,
        x: torch.Tensor,                     # [B,N,T,3] (Δx,Δy,Δyaw) ego frame
        ctx: torch.Tensor,                   # [B,N,T,5] (x,y,yaw,vx,vy)
        feasible_actions: Optional[torch.Tensor],   # [B,N,T,A]
        action_path_type: torch.Tensor,      # [A]
        action_constraint_type: torch.Tensor,  # [A]
        comparable_metrics: torch.Tensor,     # [B,N,T,A,C]
        path_polyline_idx: torch.Tensor,      # [B,N,T,P]
        map_polylines: torch.Tensor,          # [B,M,L,2]
    ) -> torch.Tensor:
        """Compute logp_x_given_m for each structured action slot."""
        B, N, T, Dx = x.shape
        if Dx < 2:
            raise ValueError("x must contain at least Δx,Δy")

        # current / next positions in ego frame
        p0 = ctx[..., 0:2]
        p1 = p0 + x[..., 0:2]

        # current velocity
        vx = ctx[..., 3]
        vy = ctx[..., 4]
        v_vec = torch.stack([vx, vy], dim=-1)  # [B,N,T,2]

        # gather per-path polylines (P path branches)
        # path_polyline_idx: [B,N,T,P]
        P = path_polyline_idx.shape[-1]
        if P != NUM_PATH_TYPES:
            raise ValueError(f"path_polyline_idx last dim must be {NUM_PATH_TYPES}, got {P}")

        # pre-gather keep and each path type polyline
        poly_by_path = _gather_polylines(map_polylines, path_polyline_idx)  # [B,N,T,P,L,2]

        # project onto KEEP (needed for lane-change splice start)
        keep_poly = poly_by_path[..., int(PathType.KEEP), :, :]
        s0_keep, _, _ = _project_to_polyline(p0, keep_poly)

        # compute s,d,tan for each path type, with lane-change splice + rho grid
        # We'll assemble tensors [B,N,T,P] for s0,s1,d1,tan0.
        s0_all = p0.new_zeros((B, N, T, P))
        s1_all = p0.new_zeros((B, N, T, P))
        d1_all = p0.new_zeros((B, N, T, P))
        tan0_all = p0.new_zeros((B, N, T, P, 2))

        for p in range(P):
            if p in (int(PathType.LANE_CHANGE_LEFT), int(PathType.LANE_CHANGE_RIGHT)):
                # MAP over rho via small grid search
                tgt_poly = poly_by_path[..., p, :, :]
                best_lp = None
                best_s0 = None
                best_s1 = None
                best_d1 = None
                best_tan0 = None
                for rho in self.rho_grid:
                    splice = self._lane_change_splice(keep_poly=keep_poly, tgt_poly=tgt_poly, s0_keep=s0_keep, rho=rho)
                    s0, d0, tan0 = _project_to_polyline(p0, splice)
                    s1, d1, _ = _project_to_polyline(p1, splice)

                    # quick score: prefer small |d1| and consistent longitudinal step
                    v_s = (v_vec * tan0).sum(dim=-1)
                    s_hat = s0 + v_s * self.dt
                    e_s = s1 - s_hat
                    score = -(d1.abs() + 0.1 * e_s.abs())  # higher better

                    if best_lp is None:
                        best_lp = score
                        best_s0, best_s1, best_d1, best_tan0 = s0, s1, d1, tan0
                    else:
                        better = score > best_lp
                        best_lp = torch.where(better, score, best_lp)
                        best_s0 = torch.where(better, s0, best_s0)
                        best_s1 = torch.where(better, s1, best_s1)
                        best_d1 = torch.where(better, d1, best_d1)
                        best_tan0 = torch.where(better.unsqueeze(-1), tan0, best_tan0)

                s0_all[..., p] = best_s0
                s1_all[..., p] = best_s1
                d1_all[..., p] = best_d1
                tan0_all[..., p, :] = best_tan0
            else:
                poly = poly_by_path[..., p, :, :]
                s0, d0, tan0 = _project_to_polyline(p0, poly)
                s1, d1, _ = _project_to_polyline(p1, poly)
                s0_all[..., p] = s0
                s1_all[..., p] = s1
                d1_all[..., p] = d1
                tan0_all[..., p, :] = tan0

        # now compute per-slot residuals and log probs
        A = action_path_type.numel()
        p_slot = action_path_type.view(1, 1, 1, A).expand(B, N, T, A)

        # select path-coordinates for each slot
        # gather along P
        s0 = torch.gather(s0_all, dim=-1, index=p_slot)
        s1 = torch.gather(s1_all, dim=-1, index=p_slot)
        d1 = torch.gather(d1_all, dim=-1, index=p_slot)

        tan0 = torch.gather(tan0_all, dim=-2, index=p_slot.unsqueeze(-1).expand(B, N, T, A, 2))

        # along-path speed
        v_s = (v_vec.unsqueeze(-2) * tan0).sum(dim=-1)

        # observed along-path speed/acc
        v_obs = (s1 - s0) / max(1e-6, self.dt)
        a_obs = (v_obs - v_s) / max(1e-6, self.dt)

        # baseline acc per slot
        a_bar = self._baseline_acc(speed=torch.sqrt(vx**2 + vy**2 + 1e-8), constraint_type=action_constraint_type, comparable_metrics=comparable_metrics)

        v_hat = v_s + a_bar * self.dt
        s_hat = s0 + v_s * self.dt + 0.5 * a_bar * (self.dt**2)

        e_s = s1 - s_hat
        e_d = d1  # δd=0
        e_v = v_obs - v_hat
        e_a = a_obs - a_bar

        e = torch.stack([e_s, e_d, e_v, e_a], dim=-1)  # [B,N,T,A,4]

        # maneuver-family weights
        fam = torch.tensor(
            [path_constraint_to_maneuver(int(action_path_type[a]), int(action_constraint_type[a])) for a in range(A)],
            device=x.device,
            dtype=torch.long,
        )
        w = self.w_by_family[fam]  # [A,4]
        w = w.view(1, 1, 1, A, 4).to(device=x.device, dtype=x.dtype).expand(B, N, T, A, 4)

        # speed-dependent down-weighting for STOP acceleration near standstill
        is_stop = (fam.view(1, 1, 1, A) == int(Maneuver.STOP)).to(dtype=x.dtype)
        speed = torch.sqrt(vx**2 + vy**2 + 1e-8).unsqueeze(-1)
        w_a = w[..., 3] * (0.1 + 0.9 * (speed > 1.0).to(dtype=x.dtype))
        w = torch.cat([w[..., :3], w_a.unsqueeze(-1)], dim=-1)

        logp = self._log_prob_diag(e, w)  # [B,N,T,A]

        # mask infeasible to avoid corrupting recognition
        if feasible_actions is not None:
            fa = feasible_actions.to(dtype=torch.bool)
            any_fa = fa.any(dim=-1, keepdim=True)
            fa = torch.where(any_fa, fa, torch.ones_like(fa))
            logp = logp.masked_fill(~fa, float("-inf"))

        return logp


# ------------------------------
# Decoder wrapper
# ------------------------------


class MotionPrimitiveDecoder(nn.Module):
    """Structured action decoder used by preference completion.

    This decoder exposes:
    - π(m | z, τ^{det}) via a paper-style random-utility policy
    - p_n(X | m, τ^{det}) via a template-conditioned primitive likelihood

    Importantly, the motion likelihood does not depend on z (no leakage).
    """

    def __init__(
        self,
        *,
        x_dim: int,
        tau_dim: int,
        ctx_dim: int,
        z_dim: int,
        num_maneuvers: int = NUM_MANEUVERS,
        beta: float = 1.0,
        feasible_action_penalty: float = 5.0,
        feasible_action_soft_penalty_train: bool = True,
        feasible_action_hard_mask_eval: bool = True,
        dt: float = 0.1,
        lane_width: float = 3.6,
        topo_horizon_m: float = 50.0,
        conflict_time_s: float = 5.0,
        a_min: float = -4.0,
        a_max: float = 2.0,
        k_yield: float = 1.5,
        rho_grid: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
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

        self.policy = PaperStructuredActionUtility(z_dim=self.z_dim, beta=float(beta), ctx_in_dim=3)
        self.primitive = TemplatePrimitiveLikelihood(
            dt=float(dt),
            lane_width=float(lane_width),
            topo_horizon_m=float(topo_horizon_m),
            conflict_time_s=float(conflict_time_s),
            a_min=float(a_min),
            a_max=float(a_max),
            k_yield=float(k_yield),
            rho_grid=rho_grid,
        )

        # fixed slot definitions (path × lon), used if caller does not supply per-slot tensors
        slot_path = []
        slot_lon = []
        for p in range(NUM_PATH_TYPES):
            for c in range(NUM_LON_CONSTRAINTS):
                slot_path.append(p)
                slot_lon.append(c)
        self.register_buffer("slot_path_type", torch.tensor(slot_path, dtype=torch.long), persistent=False)
        self.register_buffer("slot_constraint_type", torch.tensor(slot_lon, dtype=torch.long), persistent=False)

    @staticmethod
    def aggregate_family_logits(
        action_logits: torch.Tensor,
        *,
        action_family: Optional[torch.Tensor],
        feasible_actions: Optional[torch.Tensor] = None,
        num_families: int = NUM_MANEUVERS,
    ) -> torch.Tensor:
        """Aggregate slot logits into legacy maneuver-family logits via log-sum-exp."""
        if action_family is None:
            return action_logits
        if action_family.shape != action_logits.shape:
            raise ValueError(f"action_family must match logits shape, got {action_family.shape} vs {action_logits.shape}")

        out = action_logits.new_full(action_logits.shape[:-1] + (int(num_families),), float("-inf"))
        for m in range(int(num_families)):
            sel = action_family == int(m)
            if feasible_actions is not None:
                sel = sel & feasible_actions.to(dtype=torch.bool)
            logits_m = action_logits.masked_fill(~sel, float("-inf"))
            out[..., m] = torch.logsumexp(logits_m, dim=-1)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out

    @staticmethod
    def _compute_decision_features(
        *,
        ctx: torch.Tensor,                    # [B,N,T,5]
        comparable_metrics: torch.Tensor,      # [B,N,T,A,C]
        dynamic_metrics: Optional[torch.Tensor],  # [B,N,T,A,D]
        action_path_type: torch.Tensor,        # [A]
        action_constraint_type: torch.Tensor,  # [A]
        topo_horizon_m: float,
        conflict_time_s: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute paper feature vector f(m,τ^{det}) and u_ctx inputs.

        Returns
        -------
        f : [B,N,T,A,8]
        ctx_u : [B,N,T,A,3]
        """
        B, N, T, A, C = comparable_metrics.shape
        speed = torch.sqrt(ctx[..., 3] ** 2 + ctx[..., 4] ** 2 + 1e-8)  # [B,N,T]
        speed_n = (speed / 30.0).clamp(0.0, 2.0).unsqueeze(-1).expand(B, N, T, A)

        path = action_path_type.view(1, 1, 1, A).to(device=ctx.device).expand(B, N, T, A)
        lon = action_constraint_type.view(1, 1, 1, A).to(device=ctx.device).expand(B, N, T, A)

        gap_m = comparable_metrics[..., 1] * float(topo_horizon_m)
        ttc_s = comparable_metrics[..., 2] * float(conflict_time_s)
        ttc_n = (ttc_s / float(conflict_time_s)).clamp(0.0, 2.0)

        on_route = comparable_metrics[..., 5].clamp(0.0, 1.0)
        stop_flag = comparable_metrics[..., 4].clamp(0.0, 1.0)

        # progress (reward): speed + route bonus + along-path lookahead (cheap proxy)
        if dynamic_metrics is not None and dynamic_metrics.shape[-1] >= 4:
            vlim_n = dynamic_metrics[..., 3].clamp(0.0, 2.0)
        else:
            vlim_n = speed_n
        f_prog = (0.7 * vlim_n + 0.3 * on_route).clamp(0.0, 2.0)

        # comfort (cost): curvature + lateral offset
        if dynamic_metrics is not None and dynamic_metrics.shape[-1] >= 1:
            curv = dynamic_metrics[..., 0].abs().clamp(0.0, 2.0)
        else:
            curv = torch.zeros_like(f_prog)
        lat = comparable_metrics[..., 3].abs().clamp(0.0, 2.0)
        f_comfort = (curv + 0.5 * lat).clamp(0.0, 3.0)

        # lane-change cost indicator
        is_lc = ((path == int(PathType.LANE_CHANGE_LEFT)) | (path == int(PathType.LANE_CHANGE_RIGHT))).to(dtype=f_prog.dtype)
        f_lc = is_lc

        # stop action indicator
        f_stop = (lon == int(LonConstraint.STOP_LINE)).to(dtype=f_prog.dtype)

        # interaction safety margins
        f_ttc = ttc_n
        f_gap = (gap_m / float(topo_horizon_m)).clamp(0.0, 2.0)

        # priority / stop-rule violation proxy: going through a stop-required branch without STOP_LINE
        vio = (stop_flag > 0.5) & (lon != int(LonConstraint.STOP_LINE))
        f_prio = vio.to(dtype=f_prog.dtype)

        # headway cost for follow
        follow = lon == int(LonConstraint.FOLLOW)
        thw = gap_m / speed.clamp_min(0.2).unsqueeze(-1)
        headway_cost = F.relu(1.5 - thw) / 1.5
        f_headway = torch.where(follow, headway_cost, torch.zeros_like(headway_cost)).clamp(0.0, 2.0)

        f = torch.stack([
            f_prog,
            f_comfort,
            f_lc,
            f_stop,
            f_ttc,
            f_gap,
            f_prio,
            f_headway,
        ], dim=-1)

        # u_ctx inputs (low capacity): [on_route, stop_violation, is_branch]
        is_branch = ((path == int(PathType.BRANCH_LEFT)) | (path == int(PathType.BRANCH_RIGHT))).to(dtype=f_prog.dtype)
        ctx_u = torch.stack([
            on_route,
            f_prio,
            is_branch,
        ], dim=-1)

        return f, ctx_u

    def token_log_prob(
        self,
        *,
        x: torch.Tensor,                 # [B,N,T,Dx]
        tau: torch.Tensor,               # [B,N,T,Dt] (unused for policy/emission but kept)
        ctx: torch.Tensor,               # [B,N,T,Dc]
        mask: torch.Tensor,              # [B,N,T]
        feasible_actions: Optional[torch.Tensor] = None,   # [B,N,T,A]
        action_features: Optional[torch.Tensor] = None,    # kept for API compat (ignored)
        comparable_metrics: Optional[torch.Tensor] = None, # [B,N,T,A,C]
        dynamic_metrics: Optional[torch.Tensor] = None,    # [B,N,T,A,D]
        action_path_type: Optional[torch.Tensor] = None,   # [B,N,T,A] or [A]
        action_constraint_type: Optional[torch.Tensor] = None,  # [B,N,T,A] or [A]
        path_polyline_idx: Optional[torch.Tensor] = None,  # [B,N,T,P]
        map_polylines: Optional[torch.Tensor] = None,      # [B,M,L,2]
        z: torch.Tensor = None,                 # [B,N,Dz]
    ) -> PrimitiveDecodeOutput:
        """Return per-token action-slot logits and token log-likelihood."""
        if z is None:
            raise ValueError("z is required")
        m = mask > 0.5

        if comparable_metrics is None:
            raise ValueError("Paper decoder requires comparable_metrics")

        if dynamic_metrics is None:
            # allow None (we have fallbacks), but still shape-check if provided
            pass

        # slot type tensors
        if action_path_type is None:
            action_path_type = self.slot_path_type
        if action_constraint_type is None:
            action_constraint_type = self.slot_constraint_type

        # allow caller to pass per-token slot types; reduce to [A]
        if action_path_type.dim() == 4:
            # [B,N,T,A] -> assume constant over batch/time
            action_path_type = action_path_type[0, 0, 0]
        if action_constraint_type.dim() == 4:
            action_constraint_type = action_constraint_type[0, 0, 0]

        A = int(action_path_type.numel())

        # compute features for policy
        decision_f, ctx_u = self._compute_decision_features(
            ctx=ctx,
            comparable_metrics=comparable_metrics,
            dynamic_metrics=dynamic_metrics,
            action_path_type=action_path_type,
            action_constraint_type=action_constraint_type,
            topo_horizon_m=self.primitive.topo_horizon_m,
            conflict_time_s=self.primitive.conflict_time_s,
        )

        logits, u_ctx = self.policy(z=z, decision_features=decision_f, ctx_features=ctx_u, feasible_actions=feasible_actions)

        # feasible action handling (policy)
        if feasible_actions is not None:
            fa = feasible_actions.to(dtype=torch.bool)
            if fa.shape != logits.shape:
                raise ValueError(f"feasible_actions must have shape {tuple(logits.shape)}, got {tuple(fa.shape)}")
            any_feas = fa.any(dim=-1, keepdim=True)
            fa = torch.where(any_feas, fa, torch.ones_like(fa))
            if self.training and self.feasible_action_soft_penalty_train:
                logits = logits - (~fa).to(dtype=logits.dtype) * float(self.feasible_action_penalty)
            else:
                if self.feasible_action_hard_mask_eval:
                    logits = logits.masked_fill(~fa, float("-inf"))
                else:
                    logits = logits - (~fa).to(dtype=logits.dtype) * float(self.feasible_action_penalty)

        # primitive likelihood for recognition
        if path_polyline_idx is None or map_polylines is None:
            # fallback: if geometry is missing, use a weak CV likelihood in Cartesian
            # (still independent of z)
            vx = ctx[..., 3]
            vy = ctx[..., 4]
            base = torch.stack([vx * self.primitive.dt, vy * self.primitive.dt, torch.zeros_like(vx)], dim=-1)
            base = base.unsqueeze(-2).expand(*logits.shape, 3)
            x_e = x.unsqueeze(-2).expand_as(base)
            err = x_e - base
            logp = -0.5 * (err**2).sum(dim=-1)
            if feasible_actions is not None:
                fa = feasible_actions.to(dtype=torch.bool)
                any_fa = fa.any(dim=-1, keepdim=True)
                fa = torch.where(any_fa, fa, torch.ones_like(fa))
                logp = logp.masked_fill(~fa, float("-inf"))
        else:
            logp = self.primitive(
                x=x,
                ctx=ctx,
                feasible_actions=feasible_actions,
                action_path_type=action_path_type,
                action_constraint_type=action_constraint_type,
                comparable_metrics=comparable_metrics,
                path_polyline_idx=path_polyline_idx,
                map_polylines=map_polylines,
            )

        # recognition distribution q_χ(m|X,τ): softmax over logp
        if feasible_actions is not None:
            fa = feasible_actions.to(dtype=torch.bool)
            any_fa = fa.any(dim=-1, keepdim=True)
            fa = torch.where(any_fa, fa, torch.ones_like(fa))
            logp_for_softmax = logp.masked_fill(~fa, float("-inf"))
        else:
            logp_for_softmax = logp

        recog_probs = torch.softmax(logp_for_softmax, dim=-1)
        # normalized entropy confidence: 1 - H(q)/log|A|
        eps = 1e-9
        H = -(recog_probs.clamp_min(eps) * recog_probs.clamp_min(eps).log()).sum(dim=-1)
        if feasible_actions is not None:
            A_eff = feasible_actions.to(dtype=torch.float32).sum(dim=-1).clamp_min(1.0)
        else:
            A_eff = torch.full_like(H, float(A))
        recog_conf = 1.0 - H / (A_eff.log().clamp_min(eps))
        recog_conf = recog_conf.clamp(0.0, 1.0)

        # broadcast z and set delta=0
        z_mod = z.unsqueeze(2).expand(z.shape[0], z.shape[1], x.shape[2], z.shape[-1])
        z_delta = torch.zeros_like(z_mod)

        # apply token mask
        m_e = m.unsqueeze(-1)
        logits = logits.masked_fill(~m_e, 0.0)
        u_ctx = u_ctx.masked_fill(~m_e, 0.0)
        decision_f = decision_f.masked_fill(~m_e.unsqueeze(-1), 0.0)

        # logp already has -inf for infeasible; for masked timesteps set 0 to avoid NaNs
        logp = torch.nan_to_num(logp, nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf"))
        logp = torch.where(m_e, logp, torch.zeros_like(logp))

        recog_probs = recog_probs.masked_fill(~m_e, 0.0)
        recog_conf = recog_conf * m.to(dtype=recog_conf.dtype)

        return PrimitiveDecodeOutput(
            maneuver_logits=logits,
            logp_x_given_m=logp,
            z_mod=z_mod,
            z_mod_delta=z_delta,
            u_ctx=u_ctx,
            decision_features=decision_f,
            recog_probs=recog_probs,
            recog_conf=recog_conf,
        )

    @torch.no_grad()
    def maneuver_logits_last(
        self,
        *,
        z: torch.Tensor,         # [B,N,Dz]
        tau_last: torch.Tensor,  # [B,N,Dt]
        ctx_last: torch.Tensor,  # [B,N,Dc]
        feasible_actions_last: Optional[torch.Tensor] = None,  # [B,N,A]
        action_features_last: Optional[torch.Tensor] = None,
        action_family_last: Optional[torch.Tensor] = None,     # [B,N,A]
        comparable_metrics_last: Optional[torch.Tensor] = None,
        dynamic_metrics_last: Optional[torch.Tensor] = None,
        action_path_type_last: Optional[torch.Tensor] = None,
        action_constraint_type_last: Optional[torch.Tensor] = None,
        path_polyline_idx_last: Optional[torch.Tensor] = None,
        map_polylines: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Legacy maneuver-family logits for the last token."""
        tau = tau_last.unsqueeze(2)
        ctx = ctx_last.unsqueeze(2)
        dummy_x = torch.zeros((z.shape[0], z.shape[1], 1, self.x_dim), device=z.device, dtype=z.dtype)
        mask = torch.ones((z.shape[0], z.shape[1], 1), device=z.device, dtype=z.dtype)
        fa = feasible_actions_last.unsqueeze(2) if feasible_actions_last is not None else None

        cmp = comparable_metrics_last.unsqueeze(2) if comparable_metrics_last is not None else None
        dyn = dynamic_metrics_last.unsqueeze(2) if dynamic_metrics_last is not None else None
        ppi = path_polyline_idx_last.unsqueeze(2) if path_polyline_idx_last is not None else None

        out = self.token_log_prob(
            x=dummy_x,
            tau=tau,
            ctx=ctx,
            mask=mask,
            feasible_actions=fa,
            action_features=action_features_last.unsqueeze(2) if action_features_last is not None else None,
            comparable_metrics=cmp,
            dynamic_metrics=dyn,
            action_path_type=action_path_type_last,
            action_constraint_type=action_constraint_type_last,
            path_polyline_idx=ppi,
            map_polylines=map_polylines,
            z=z,
        )

        action_logits = out.maneuver_logits.squeeze(2)
        if action_family_last is None:
            return action_logits

        family_logits = self.aggregate_family_logits(
            action_logits.unsqueeze(2),
            action_family=action_family_last.unsqueeze(2),
            feasible_actions=fa,
            num_families=self.num_maneuvers,
        )
        return family_logits.squeeze(2)
