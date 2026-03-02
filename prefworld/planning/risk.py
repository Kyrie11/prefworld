from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def compute_cvar(values: torch.Tensor, alpha: float = 0.9, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute weighted CVaR of 'loss-like' values in the upper tail.

    Args:
      values:  [B,S]
      alpha:   tail quantile (e.g., 0.9 keeps worst 10%)
      weights: [B,S] simplex (optional). If None, uniform weights.
    Returns:
      cvar:    [B]
    """
    B, S = values.shape
    tail = float(1.0 - alpha)
    if tail <= 0.0:
        return values.max(dim=-1).values

    if weights is None:
        weights = torch.full_like(values, 1.0 / float(S))
    else:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

    v_sorted, idx = torch.sort(values, dim=-1, descending=True)
    w_sorted = weights.gather(dim=-1, index=idx)

    cumsum = torch.cumsum(w_sorted, dim=-1)
    in_tail = cumsum <= tail
    boundary = torch.argmax((cumsum >= tail).to(torch.int64), dim=-1)  # [B]

    w_tail = w_sorted * in_tail.to(dtype=w_sorted.dtype)
    b = torch.arange(B, device=values.device)
    w_before = torch.where(boundary > 0, cumsum[b, boundary - 1], torch.zeros((B,), device=values.device, dtype=values.dtype))
    w_part = (tail - w_before).clamp(min=0.0)
    w_tail[b, boundary] = w_tail[b, boundary] + w_part

    denom = w_tail.sum(dim=-1).clamp_min(1e-12)
    return (v_sorted * w_tail).sum(dim=-1) / denom


def collision_indicator(
    ego_traj: torch.Tensor,         # [B,H,2]
    agent_traj: torch.Tensor,       # [B,K,H,2]
    ego_radius: float = 1.2,
    agent_radius: Optional[torch.Tensor] = None,  # [B,K] or None
) -> torch.Tensor:
    """Simple circle-based collision indicator.

    Returns:
      coll: [B,K,H] bool
    """
    B, H, _ = ego_traj.shape
    _, K, H2, _ = agent_traj.shape
    assert H2 == H
    ego_xy = ego_traj.unsqueeze(1)  # [B,1,H,2]
    d = torch.norm(agent_traj - ego_xy, dim=-1)  # [B,K,H]
    if agent_radius is None:
        thresh = float(ego_radius)
    else:
        thresh = ego_radius + agent_radius.unsqueeze(-1)  # [B,K,1]
    return d < thresh


def collision_risk(
    ego_traj: torch.Tensor,           # [B,H,2]
    agent_traj_samples: torch.Tensor, # [B,S,K,H,2]
    weights: Optional[torch.Tensor] = None,  # [B,S]
    ego_radius: float = 1.2,
    agent_radius: Optional[torch.Tensor] = None,  # [B,K]
    cvar_alpha: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute expected collision probability and CVaR over trajectory samples.

    We treat each sample s as a possible world realization and compute its collision indicator over time.
    The sample-level loss is max_t max_k I(collision), then:
      - expected risk = E[loss]
      - CVaR risk    = CVaR_alpha(loss)
    """
    B, S, K, H, _ = agent_traj_samples.shape
    if weights is None:
        weights = torch.full((B, S), 1.0 / float(S), device=agent_traj_samples.device, dtype=torch.float32)
    else:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)

    # loss per sample: any collision at any time with any agent
    ego_xy = ego_traj.unsqueeze(1).unsqueeze(1).expand(B, S, K, H, 2)  # [B,S,K,H,2]
    d = torch.norm(agent_traj_samples - ego_xy, dim=-1)  # [B,S,K,H]
    if agent_radius is None:
        thresh = float(ego_radius)
        coll = d < thresh
    else:
        thresh = ego_radius + agent_radius.unsqueeze(1).unsqueeze(-1)  # [B,1,K,1]
        coll = d < thresh
    loss_s = coll.any(dim=-1).any(dim=-1).to(torch.float32)  # [B,S] 0/1

    exp_risk = (loss_s * weights).sum(dim=-1)  # [B]
    cvar_risk = compute_cvar(loss_s, alpha=float(cvar_alpha), weights=weights)  # [B]
    return exp_risk, cvar_risk
