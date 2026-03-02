from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.utils import save_checkpoint


@dataclass
class TrainConfig:
    """Simple joint trainer config.

    For paper-faithful training we recommend using the stage scripts in `prefworld/scripts/`.
    This trainer is kept as a convenient baseline.
    """

    output_dir: str

    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 20
    grad_clip_norm: float = 5.0

    # PC split
    pc_split_mode: str = "random"
    pc_query_ratio: float = 0.3

    # PC weights (Eq.18)
    lambda_distill_mu: float = 1.0
    lambda_distill_cov: float = 0.05
    lambda_prior: float = 0.1
    lambda_con: float = 0.05
    lambda_overlap: float = 1e-3
    lambda_mod: float = 1e-3

    n_z_samples: int = 1
    free_bits: float = 0.0

    # EB penalties
    eb_smooth_scale: float = 0.0
    eb_phys_dist_threshold_m: float = 1e9
    eb_phys_penalty_scale: float = 0.0

    # loss mixing
    w_pc: float = 1.0
    w_intent: float = 0.0
    w_eb: float = 1.0


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(model: PrefWorldModel, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    counts = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        batch = _move_to_device(batch, device)
        out = model(
            batch,
            run_pc=True,
            run_eb=True,
            pc_split_mode=cfg.pc_split_mode,
            # Use the configured query ratio so PC query NLL is meaningful during validation.
            pc_query_ratio=float(cfg.pc_query_ratio),
            lambda_distill_mu=cfg.lambda_distill_mu,
            lambda_distill_cov=cfg.lambda_distill_cov,
            lambda_prior=cfg.lambda_prior,
            lambda_con=float(cfg.lambda_con),
            lambda_overlap=cfg.lambda_overlap,
            lambda_mod=cfg.lambda_mod,
            n_z_samples=1,
            free_bits=cfg.free_bits,
            eb_smooth_scale=cfg.eb_smooth_scale,
            eb_phys_dist_threshold_m=cfg.eb_phys_dist_threshold_m,
            eb_phys_penalty_scale=cfg.eb_phys_penalty_scale,
            use_pseudo_intent=cfg.w_intent > 0.0,
        )

        keys = list(out.losses.keys()) + list(out.metrics.keys())
        for k in keys:
            v = out.losses.get(k, out.metrics.get(k))
            sums[k] = sums.get(k, 0.0) + float(v.item())
        counts += 1

    if counts == 0:
        return {k: 0.0 for k in sums}
    return {k: v / counts for k, v in sums.items()}


def train(
    model: PrefWorldModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    best_path = ckpt_dir / "best.pt"

    for epoch in range(int(cfg.max_epochs)):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")

        for batch in pbar:
            batch = _move_to_device(batch, device)
            out = model(
                batch,
                run_pc=True,
                run_eb=True,
                pc_split_mode=cfg.pc_split_mode,
                pc_query_ratio=float(cfg.pc_query_ratio),
                lambda_distill_mu=float(cfg.lambda_distill_mu),
                lambda_distill_cov=float(cfg.lambda_distill_cov),
                lambda_prior=float(cfg.lambda_prior),
                lambda_con=float(cfg.lambda_con),
                lambda_overlap=float(cfg.lambda_overlap),
                lambda_mod=float(cfg.lambda_mod),
                n_z_samples=int(cfg.n_z_samples),
                free_bits=float(cfg.free_bits),
                eb_smooth_scale=float(cfg.eb_smooth_scale),
                eb_phys_dist_threshold_m=float(cfg.eb_phys_dist_threshold_m),
                eb_phys_penalty_scale=float(cfg.eb_phys_penalty_scale),
                use_pseudo_intent=cfg.w_intent > 0.0,
                intent_weight=1.0,
            )

            loss = (
                float(cfg.w_pc) * out.losses["loss_pc"]
                + float(cfg.w_intent) * out.losses["loss_intent"]
                + float(cfg.w_eb) * out.losses["loss_eb"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
            optimizer.step()

            pbar.set_postfix(
                loss=float(loss.item()),
                pc=float(out.losses["loss_pc"].item()),
                eb=float(out.losses["loss_eb"].item()),
                intent_acc=float(out.metrics.get("intent_acc", torch.tensor(0.0)).item()),
                struct=float(out.metrics.get("struct_exact", torch.tensor(0.0)).item()),
            )

        if val_loader is not None:
            stats = evaluate(model, val_loader, device, cfg)
            score = stats.get("struct_exact", 0.0) + stats.get("intent_acc", 0.0)
            print(
                f"Epoch {epoch}: val struct={stats.get('struct_exact',0):.4f}, "
                f"intent_acc={stats.get('intent_acc',0):.4f}, score={score:.4f}"
            )

            save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, epoch, best_score)
            if score > best_score:
                best_score = score
                save_checkpoint(str(best_path), model, optimizer, epoch, best_score)
                print(f"  New best checkpoint -> {best_path} (score={best_score:.4f})")
        else:
            save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, epoch, best_score)

    print(f"Training done. Best score: {best_score:.4f}")
    print(f"Best checkpoint: {best_path}")
