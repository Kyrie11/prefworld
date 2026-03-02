from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from prefworld.planning.risk import compute_cvar


@dataclass
class BicycleState:
    x: torch.Tensor   # [...]
    y: torch.Tensor   # [...]
    yaw: torch.Tensor # [...]
    v: torch.Tensor   # [...]


def rollout_bicycle(
    state0: BicycleState,
    controls: torch.Tensor,   # [...,H,2] (accel, steer)
    dt: float = 0.1,
    wheelbase: float = 2.8,
) -> BicycleState:
    """Forward-simulate a simple kinematic bicycle model."""
    a = controls[..., :, 0]
    delta = controls[..., :, 1].clamp(min=-0.6, max=0.6)

    x = [state0.x]
    y = [state0.y]
    yaw = [state0.yaw]
    v = [state0.v]

    for t in range(controls.shape[-2]):
        vt = v[-1]
        yawt = yaw[-1]
        xt = x[-1]
        yt = y[-1]

        vt1 = (vt + a[..., t] * dt).clamp(min=0.0)
        beta = torch.atan2(torch.tan(delta[..., t]), torch.tensor(1.0, device=delta.device, dtype=delta.dtype))
        yawt1 = yawt + (vt1 / wheelbase) * torch.tan(delta[..., t]) * dt
        xt1 = xt + vt1 * torch.cos(yawt + beta) * dt
        yt1 = yt + vt1 * torch.sin(yawt + beta) * dt

        v.append(vt1)
        yaw.append(yawt1)
        x.append(xt1)
        y.append(yt1)

    # stack, drop initial
    x = torch.stack(x[1:], dim=-1)
    y = torch.stack(y[1:], dim=-1)
    yaw = torch.stack(yaw[1:], dim=-1)
    v = torch.stack(v[1:], dim=-1)
    return BicycleState(x=x, y=y, yaw=yaw, v=v)


def mpc_track(
    state0: BicycleState,
    ref_xy: torch.Tensor,      # [B,H,2]
    ref_yaw: Optional[torch.Tensor] = None,  # [B,H]
    dt: float = 0.1,
    horizon_steps: int = 20,
    wheelbase: float = 2.8,
    iters: int = 40,
    lr: float = 0.3,
    w_pos: float = 1.0,
    w_yaw: float = 0.2,
    w_u: float = 0.05,
    w_du: float = 0.05,
) -> torch.Tensor:
    """A small gradient-based MPC tracker (torch-only, no external solvers).

    Returns:
      controls: [B,H,2] (accel, steer)
    """
    B = ref_xy.shape[0]
    H = int(horizon_steps)
    device = ref_xy.device
    dtype = ref_xy.dtype

    u = torch.zeros((B, H, 2), device=device, dtype=dtype, requires_grad=True)

    opt = torch.optim.Adam([u], lr=float(lr))
    for _ in range(int(iters)):
        opt.zero_grad(set_to_none=True)
        st = rollout_bicycle(state0, u, dt=dt, wheelbase=wheelbase)
        xy = torch.stack([st.x, st.y], dim=-1)  # [B,H,2]
        pos_err = ((xy - ref_xy[:, :H]).pow(2)).sum(dim=-1).mean()
        yaw_err = torch.tensor(0.0, device=device, dtype=dtype)
        if ref_yaw is not None:
            yaw_err = ((st.yaw - ref_yaw[:, :H]).pow(2)).mean()

        u_cost = (u.pow(2)).mean()
        du = u[:, 1:] - u[:, :-1]
        du_cost = (du.pow(2)).mean() if du.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)

        loss = float(w_pos) * pos_err + float(w_yaw) * yaw_err + float(w_u) * u_cost + float(w_du) * du_cost
        loss.backward()
        opt.step()

        # clamp to plausible ranges
        with torch.no_grad():
            u[:, :, 0].clamp_(-4.0, 3.0)   # accel
            u[:, :, 1].clamp_(-0.6, 0.6)   # steer

    return u.detach()


def _softmin(x: torch.Tensor, beta: float = 20.0, dim: int = -1) -> torch.Tensor:
    """Differentiable approximation to min(x) along `dim`."""
    b = float(beta)
    if b <= 0.0:
        return x.min(dim=dim).values
    return -(1.0 / b) * torch.logsumexp(-b * x, dim=dim)


def mpc_track_robust(
    state0: BicycleState,
    ref_xy: torch.Tensor,      # [B,H,2]
    agent_traj_samples: torch.Tensor,  # [B,S,K,H,2]
    weights: Optional[torch.Tensor] = None,  # [B,S]
    agent_radius: Optional[torch.Tensor] = None,  # [B,K]
    ref_yaw: Optional[torch.Tensor] = None,  # [B,H]
    dt: float = 0.1,
    horizon_steps: int = 20,
    wheelbase: float = 2.8,
    iters: int = 50,
    lr: float = 0.2,
    # Tracking costs
    w_pos: float = 1.0,
    w_yaw: float = 0.2,
    w_u: float = 0.05,
    w_du: float = 0.05,
    # Chance constraints (soft penalties)
    chance_collision_epsilon: float = 0.05,
    cvar_alpha: float = 0.9,
    chance_cvar_rho: float = 0.1,
    w_chance: float = 50.0,
    w_cvar: float = 50.0,
    # Robustification knobs
    ego_radius: float = 1.2,
    tighten_margin: float = 0.3,
    softmin_beta: float = 20.0,
    sigmoid_temp: float = 0.25,
) -> torch.Tensor:
    """Gradient-based MPC tracker with differentiable chance constraints.

    This is a pragmatic, torch-only implementation that approximates the paper's robust MPC:
      - We track a reference trajectory.
      - We penalize violations of chance constraints under an uncertainty set of sampled
        world rollouts (agent trajectories), using a *soft* collision indicator.

    Args:
      agent_traj_samples: trajectory samples for other agents under structure (and possibly other)
                          uncertainties. Shape [B,S,K,H,2]. K excludes ego.
      weights:            sample weights (simplex) [B,S]. If None, uniform.
      agent_radius:       per-agent radius [B,K]. If None, constant 1.5m.

    Returns:
      controls: [B,H,2] (accel, steer)
    """
    B = ref_xy.shape[0]
    H = int(horizon_steps)
    device = ref_xy.device
    dtype = ref_xy.dtype

    # Normalize weights
    S = agent_traj_samples.shape[1]
    if weights is None:
        weights = torch.full((B, S), 1.0 / float(S), device=device, dtype=dtype)
    else:
        weights = weights.to(device=device, dtype=dtype)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

    if agent_radius is None:
        agent_radius = torch.full((B, agent_traj_samples.shape[2]), 1.5, device=device, dtype=dtype)
    else:
        agent_radius = agent_radius.to(device=device, dtype=dtype)

    # Optimization variable
    u = torch.zeros((B, H, 2), device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([u], lr=float(lr))

    # Pre-slice to horizon
    ref_xy_h = ref_xy[:, :H]
    ref_yaw_h = ref_yaw[:, :H] if ref_yaw is not None else None
    agent_traj_h = agent_traj_samples[:, :, :, :H]

    for _ in range(int(iters)):
        opt.zero_grad(set_to_none=True)

        st = rollout_bicycle(state0, u, dt=dt, wheelbase=wheelbase)
        ego_xy = torch.stack([st.x, st.y], dim=-1)  # [B,H,2]

        # Tracking losses
        pos_err = ((ego_xy - ref_xy_h).pow(2)).sum(dim=-1).mean()
        yaw_err = torch.tensor(0.0, device=device, dtype=dtype)
        if ref_yaw_h is not None:
            yaw_err = ((st.yaw - ref_yaw_h).pow(2)).mean()
        u_cost = (u.pow(2)).mean()
        du = u[:, 1:] - u[:, :-1]
        du_cost = (du.pow(2)).mean() if du.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)

        # ----------------------------
        # Differentiable collision risk
        # ----------------------------
        # Compute signed margin: d - (r_ego + r_agent + tighten)
        ego_xy_e = ego_xy.unsqueeze(1).unsqueeze(1)  # [B,1,1,H,2]
        d = torch.norm(agent_traj_h - ego_xy_e, dim=-1)  # [B,S,K,H]
        thresh = float(ego_radius + tighten_margin) + agent_radius.unsqueeze(1).unsqueeze(-1)  # [B,1,K,1]
        margin = d - thresh  # [B,S,K,H]
        # Soft minimum margin across agents+time per sample
        margin_flat = margin.reshape(B, S, -1)  # [B,S,K*H]
        m_min = _softmin(margin_flat, beta=float(softmin_beta), dim=-1)  # [B,S]

        # Soft collision indicator in (0,1)
        temp = max(1e-6, float(sigmoid_temp))
        coll_s = torch.sigmoid((-m_min) / temp)  # [B,S]

        exp_risk = (coll_s * weights).sum(dim=-1)  # [B]
        cvar_risk = compute_cvar(coll_s, alpha=float(cvar_alpha), weights=weights)  # [B]

        # Chance-constraint penalties (soft)
        pen_chance = torch.relu(exp_risk - float(chance_collision_epsilon))
        pen_cvar = torch.relu(cvar_risk - float(chance_cvar_rho))
        risk_pen = float(w_chance) * (pen_chance.pow(2)).mean() + float(w_cvar) * (pen_cvar.pow(2)).mean()

        loss = (
            float(w_pos) * pos_err
            + float(w_yaw) * yaw_err
            + float(w_u) * u_cost
            + float(w_du) * du_cost
            + risk_pen
        )

        loss.backward()
        opt.step()

        with torch.no_grad():
            u[:, :, 0].clamp_(-4.0, 3.0)
            u[:, :, 1].clamp_(-0.6, 0.6)

    return u.detach()
