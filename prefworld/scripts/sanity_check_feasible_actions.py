from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch
from prefworld.models.prefworld_model import PrefWorldModel


def _percentiles(x: np.ndarray, ps=(10, 50, 90)) -> dict:
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser("Sanity-check feasible_actions masks vs. maneuver labels")
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    ds = CachedNuPlanDataset(cache_dir=args.cache_dir, split=args.split, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    # TemplateEncoder computes feasible_actions deterministically from map geometry,
    # so weights do not matter here.
    model = PrefWorldModel().to(device).eval()

    total_valid_agents = 0
    total_mismatch_agents = 0
    total_no_feas = 0
    per_label = defaultdict(lambda: {"valid": 0, "mismatch": 0})

    # ego stats (optional)
    total_valid_ego = 0
    total_mismatch_ego = 0

    for batch in loader:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        template, ego_valid, agents_valid = model.encode_templates(batch)
        fa = template.feasible_actions  # [B, 1+N, T, M]
        if fa is None:
            raise RuntimeError("TemplateEncoder did not return feasible_actions.")

        # last token
        fa_last_agents = fa[:, 1:, -1, :]  # [B, N, M]
        fa_last_ego = fa[:, 0:1, -1, :]    # [B, 1, M]

        # validity
        valid_agents = (batch["agents_hist_mask"][:, :, -1] > 0.5)
        valid_ego = torch.ones((valid_agents.shape[0], 1), device=device, dtype=torch.bool)

        # labels (NOTE: these are computed from future trajectories; use only as offline sanity checks!)
        y_agents = batch.get("agents_maneuver", None)
        y_ego = batch.get("ego_maneuver", None)
        if y_agents is None:
            raise KeyError("Batch missing agents_maneuver")
        if y_ego is None:
            raise KeyError("Batch missing ego_maneuver")
        y_agents = y_agents.long()
        y_ego = y_ego.long()

        # any feasible?
        any_feas = fa_last_agents.any(dim=-1)
        total_no_feas += int((valid_agents & (~any_feas)).sum().item())

        # mismatch: true label not in feasible set
        idx = y_agents.unsqueeze(-1).clamp(min=0, max=fa_last_agents.shape[-1] - 1)
        ok = fa_last_agents.gather(-1, idx).squeeze(-1)
        mismatch = valid_agents & (~ok)

        total_valid_agents += int(valid_agents.sum().item())
        total_mismatch_agents += int(mismatch.sum().item())

        # per label
        for m in range(fa_last_agents.shape[-1]):
            m_mask = valid_agents & (y_agents == m)
            per_label[m]["valid"] += int(m_mask.sum().item())
            per_label[m]["mismatch"] += int((mismatch & (y_agents == m)).sum().item())

        # ego
        idx_e = y_ego.unsqueeze(-1).clamp(min=0, max=fa_last_ego.shape[-1] - 1)
        ok_e = fa_last_ego.gather(-1, idx_e).squeeze(-1)
        mismatch_e = valid_ego & (~ok_e)
        total_valid_ego += int(valid_ego.sum().item())
        total_mismatch_ego += int(mismatch_e.sum().item())

    # report
    print("\n=== Feasible-actions sanity check (offline) ===")
    if total_valid_agents > 0:
        print(f"Agents: mismatch rate = {total_mismatch_agents / total_valid_agents:.4%} "
              f"({total_mismatch_agents}/{total_valid_agents})")
    else:
        print("Agents: no valid samples")

    if total_valid_ego > 0:
        print(f"Ego:    mismatch rate = {total_mismatch_ego / total_valid_ego:.4%} "
              f"({total_mismatch_ego}/{total_valid_ego})")

    print(f"Agents with NO feasible action (should be ~0): {total_no_feas}")
    print("\nPer-maneuver mismatch (agents):")
    for m in sorted(per_label.keys()):
        v = per_label[m]["valid"]
        mm = per_label[m]["mismatch"]
        rate = float(mm) / float(v) if v > 0 else float("nan")
        print(f"  m={m}: mismatch={rate:.4%} ({mm}/{v})")

    print("\nInterpretation:")
    print("  - If mismatch is high for lane-change/turn maneuvers, feasible_actions is likely too strict or misaligned")
    print("    with the maneuver label extraction. In that case, consider increasing feasible_action_penalty (soft)")
    print("    or improving map polyline extraction / polyline types.")
    print("  - Note: labels are computed from future trajectories, so this script is ONLY for offline debugging.")


if __name__ == "__main__":
    main()
