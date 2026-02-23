from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch
from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.trainer import evaluate
from prefworld.training.utils import load_checkpoint
from prefworld.utils.config import load_config, make_argparser, parse_overrides


def main() -> None:
    parser = make_argparser("Validate PrefWorld on cached nuPlan samples")
    args, unknown = parser.parse_known_args()
    overrides = parse_overrides(unknown)
    cfg = load_config(args.config, overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = CachedNuPlanDataset(
        cache_dir=str(cfg.dataset.cache_dir),
        split=str(cfg.dataset.split),
        max_samples=cfg.dataset.get("max_samples", None),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.eval.num_workers),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    model = PrefWorldModel(**cfg.model).to(device)
    load_checkpoint(str(cfg.eval.checkpoint), model)
    stats = evaluate(model, loader, device)
    print("Validation results:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
