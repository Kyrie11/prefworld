from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch
from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.trainer import TrainConfig, train
from prefworld.training.utils import set_seed
from prefworld.utils.config import ensure_dir, load_config, make_argparser, parse_overrides


def main() -> None:
    parser = make_argparser("Train PrefWorld on cached nuPlan samples")
    args, unknown = parser.parse_known_args()
    overrides = parse_overrides(unknown)
    cfg = load_config(args.config, overrides)

    set_seed(int(cfg.train.seed))
    ensure_dir(str(cfg.train.output_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    train_ds = CachedNuPlanDataset(
        cache_dir=str(cfg.dataset.cache_dir),
        split=str(cfg.dataset.train_split),
        max_samples=cfg.dataset.get("max_train_samples", None),
    )
    val_ds = CachedNuPlanDataset(
        cache_dir=str(cfg.dataset.cache_dir),
        split=str(cfg.dataset.val_split),
        max_samples=cfg.dataset.get("max_val_samples", None),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    # Model
    model = PrefWorldModel(**cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))

    # Train config
    tcfg = TrainConfig(**cfg.train)

    # Save config copy
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.train.output_dir, "config.yaml").write_text(str(cfg), encoding="utf-8")

    train(model, train_loader, val_loader, optimizer, tcfg, device)


if __name__ == "__main__":
    main()
