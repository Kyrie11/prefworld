from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch
from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.trainer import TrainConfig, evaluate
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

    # ------------------------------------------------------------------
    # Robust checkpoint loading
    #
    # Bug-fix: evaluation configs may not match the training-time model
    # hyper-parameters (e.g. pc_hidden). New checkpoints store model_hparams
    # so we can re-instantiate the correct architecture automatically.
    # ------------------------------------------------------------------
    ckpt_path = str(cfg.eval.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    use_ckpt_hparams = bool(getattr(cfg.eval, "use_ckpt_hparams", True))
    if use_ckpt_hparams and isinstance(ckpt, dict) and "model_hparams" in ckpt:
        model_kwargs = dict(ckpt["model_hparams"])
        print("Instantiating model from checkpoint hparams.")
    else:
        model_kwargs = dict(cfg.model)

    model = PrefWorldModel(**model_kwargs).to(device)
    load_checkpoint(ckpt_path, model)

    # validate.py uses the trainer's evaluation loop; provide a minimal TrainConfig so
    # PC query NLL (and any EB regularizers) are computed consistently.
    train_cfg = TrainConfig(
        output_dir=str(cfg.dataset.cache_dir),
        pc_split_mode=str(getattr(cfg.eval, "pc_split_mode", "random")),
        pc_query_ratio=float(getattr(cfg.eval, "pc_query_ratio", 0.3)),
        lambda_con=float(getattr(cfg.eval, "lambda_con", 0.0)),
        eb_smooth_scale=float(getattr(cfg.eval, "lambda_smooth", 0.0)),
        eb_phys_penalty_scale=float(getattr(cfg.eval, "lambda_phys", 0.0)),
    )
    stats = evaluate(model, loader, device, train_cfg)
    print("Validation results:")
    for k, v in stats.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
