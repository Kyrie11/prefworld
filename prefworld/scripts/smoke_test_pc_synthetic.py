"""Synthetic smoke test for PreferenceCompletion + MotionPrimitiveDecoder.

This test does NOT require nuPlan data. It only checks:
- shapes are consistent
- forward pass runs without NaNs

Run:
    PYTHONPATH=. python -m prefworld.scripts.smoke_test_pc_synthetic
"""

from __future__ import annotations

import torch

from prefworld.models.preference_completion import PreferenceCompletion
from prefworld.data.labels import NUM_MANEUVERS, NUM_PATH_TYPES, NUM_LON_CONSTRAINTS


def main() -> None:
    torch.manual_seed(0)

    B, N, T = 2, 3, 4
    A = NUM_PATH_TYPES * NUM_LON_CONSTRAINTS
    P = NUM_PATH_TYPES
    M, L = 12, 16

    pc = PreferenceCompletion(
        x_dim=3,
        tau_dim=64,
        ctx_dim=5,
        z_dim=8,
        num_maneuvers=NUM_MANEUVERS,
        action_feature_dim=32,
        hidden_dim=64,
        prior_logvar=0.0,
        num_z_samples=3,
    )

    x = torch.randn(B, N, T, 3)
    tau = torch.randn(B, N, T, 64)
    ctx = torch.randn(B, N, T, 5)
    mask = (torch.rand(B, N, T) > 0.2).float()

    feasible_actions = (torch.rand(B, N, T, A) > 0.3)
    feasible_actions[..., 0] = True

    comparable_metrics = torch.randn(B, N, T, A, 6)
    dynamic_metrics = torch.randn(B, N, T, A, 6)
    path_polyline_idx = torch.randint(0, M, (B, N, T, P))
    map_polylines = torch.randn(B, M, L, 2)

    action_family = torch.randint(0, NUM_MANEUVERS, (B, N, T, A))

    out = pc(
        x=x,
        tau=tau,
        ctx=ctx,
        mask=mask,
        feasible_actions=feasible_actions,
        action_family=action_family,
        comparable_metrics=comparable_metrics,
        dynamic_metrics=dynamic_metrics,
        path_polyline_idx=path_polyline_idx,
        map_polylines=map_polylines,
        split_mode="random",
        query_ratio=0.5,
    )

    print("[OK] forward pass")
    print(" loss_total:", float(out.loss_total))
    print(" loss_query_nll:", float(out.loss_query_nll))
    print(" loss_kl_ctx_prior:", float(out.loss_kl_ctx_prior))
    print(" loss_invariance(R_inv):", float(out.loss_contrastive))
    print(" action_logits_last:", tuple(out.action_logits_last.shape) if out.action_logits_last is not None else None)
    print(" maneuver_logits_last:", tuple(out.maneuver_logits_last.shape))

    assert torch.isfinite(out.loss_total), "loss_total is not finite"
    assert torch.isfinite(out.loss_query_nll), "loss_query_nll is not finite"


if __name__ == "__main__":
    main()
