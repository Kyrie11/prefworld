from __future__ import annotations

"""(Req-5) Compute oracle structure edit-distance statistics.

We measure how far the oracle next structure A_{t+Δ} is from the current A_t.
This helps decide whether the EB-STM candidate set needs multi-edit expansions.

Edit distance here is defined as the number of unordered pairs (i<j) whose
3-state relation differs:
  0 = no edge
  1 = i -> j
  2 = j -> i

We compute the distance over valid agents only (agent present at the current step).

Example:
  python -m prefworld.scripts.compute_edit_distance_stats \
    --cache_dir /path/to/cache \
    --split val \
    --max_samples 5000
"""

import argparse
from collections import Counter
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch


def relation_state3(A: torch.Tensor) -> torch.Tensor:
    """Encode adjacency into 3-state relation type tensor [B,K,K]."""
    a = (A > 0.5)
    at = a.transpose(-1, -2)
    typ = torch.zeros_like(A, dtype=torch.int64)
    typ = typ + (a & (~at)).to(torch.int64) * 1
    typ = typ + ((~a) & at).to(torch.int64) * 2
    # both-direction edges are rare; we treat them as state 0 in the distance.
    return typ


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

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

    dist_list = []
    hist: Counter = Counter()

    for batch in tqdm(loader, desc="edit-distance"):
        A_t = batch.get("structure_t_rule", None)
        A_t1 = batch.get("structure_t1", None)
        if A_t is None or A_t1 is None:
            raise KeyError("Batch must contain structure_t_rule and structure_t1")

        # Valid agents at current time step.
        agents_valid = (batch["agents_hist_mask"][:, :, -1] > 0.5)  # [B,N]
        B, N = agents_valid.shape
        # Include ego (index 0)
        node_mask = torch.cat([torch.ones((B, 1), dtype=torch.bool), agents_valid.to(torch.bool)], dim=1)  # [B,K]

        typ0 = relation_state3(A_t)
        typ1 = relation_state3(A_t1)

        K = typ0.shape[-1]
        triu = torch.triu(torch.ones((K, K), dtype=torch.bool), diagonal=1)
        mask_pair = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2) & triu.unsqueeze(0)

        diff = (typ0 != typ1) & mask_pair
        d = diff.sum(dim=(-1, -2)).cpu().numpy().astype(np.int64)
        dist_list.append(d)
        for x in d.tolist():
            hist[int(x)] += 1

    if not dist_list:
        print("No samples processed.")
        return
    dist = np.concatenate(dist_list, axis=0)

    # Summary
    mean = float(dist.mean())
    med = float(np.median(dist))
    p90 = float(np.quantile(dist, 0.9))
    frac_gt1 = float((dist > 1).mean())
    frac_gt2 = float((dist > 2).mean())

    print("==== Oracle edit-distance stats ====")
    print(f"N: {dist.shape[0]}")
    print(f"mean: {mean:.3f}")
    print(f"median: {med:.3f}")
    print(f"p90: {p90:.3f}")
    print(f"P(dist>1): {frac_gt1:.3%}")
    print(f"P(dist>2): {frac_gt2:.3%}")

    print("\nHistogram (distance: count):")
    for k in sorted(hist.keys())[:50]:
        print(f"  {k}: {hist[k]}")


if __name__ == "__main__":
    main()
