from __future__ import annotations

import argparse
from pathlib import Path

import torch

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch
from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.trainer import TrainConfig, train
from prefworld.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Seed
    seed = int(getattr(cfg, "seed", 0))
    torch.manual_seed(seed)

    device = torch.device(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Dataset
    train_ds = CachedNuPlanDataset(cfg.dataset.cache_dir, cfg.dataset.split, max_samples=cfg.dataset.max_samples)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if hasattr(cfg, "eval") and getattr(cfg.eval, "cache_dir", None):
        val_ds = CachedNuPlanDataset(cfg.eval.cache_dir, split="val", max_samples=None)
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=int(getattr(cfg.eval, "batch_size", 64)),
            shuffle=False,
            num_workers=int(getattr(cfg.eval, "num_workers", 4)),
            collate_fn=collate_batch,
            pin_memory=True,
        )

    # Model
    model = PrefWorldModel(
        agent_feat_dim=int(cfg.model.agent_feat_dim),
        z_dim=int(cfg.model.z_dim),
        tau_dim=int(cfg.model.tau_dim),
        pc_hidden=int(getattr(cfg.model, "pc_hidden", 128)),
        template_hidden=int(getattr(cfg.model, "template_hidden", 128)),
        energy_hidden=int(getattr(cfg.model, "energy_hidden", 128)),
        eb_temperature=float(getattr(cfg.model, "eb_temperature", 1.0)),
        eb_max_candidates=int(getattr(cfg.model, "eb_max_candidates", 64)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.optim.lr), weight_decay=float(cfg.optim.weight_decay))

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map legacy config keys
    pc_query_ratio = float(getattr(cfg.train, "pc_query_ratio", getattr(cfg.train, "pc_drop_prob", 0.2)))

    tcfg = TrainConfig(
        output_dir=str(out_dir),
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        max_epochs=int(cfg.train.max_epochs),
        grad_clip_norm=float(cfg.train.grad_clip_norm),
        pc_split_mode=str(getattr(cfg.train, "pc_split_mode", "random")),
        pc_query_ratio=pc_query_ratio,
        # Loss mix
        w_pc=float(getattr(cfg.train, "w_pc", 1.0)),
        w_intent=float(getattr(cfg.train, "w_intent", 0.0)),
        w_eb=float(getattr(cfg.train, "w_eb", 1.0)),
        # Old w_kl mapped to λ_prior (roughly)
        lambda_prior=float(getattr(cfg.train, "lambda_prior", getattr(cfg.train, "w_kl", 0.1))),
    )

    train(model, train_loader, val_loader, optimizer, tcfg, device)


if __name__ == "__main__":
    main()
