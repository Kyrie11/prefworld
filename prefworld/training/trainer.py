from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.utils import save_checkpoint


@dataclass
class TrainConfig:
    output_dir: str
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 20
    grad_clip_norm: float = 5.0
    pc_drop_prob: float = 0.2
    w_pc: float = 1.0
    w_kl: float = 0.1
    w_intent: float = 1.0
    w_eb: float = 1.0


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model: PrefWorldModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums: Dict[str, float] = {}
    counts: int = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        batch = _move_to_device(batch, device)
        out = model(batch, pc_drop_prob=0.0)
        # Aggregate losses and metrics
        keys = list(out.losses.keys()) + list(out.metrics.keys())
        for k in keys:
            val = out.losses.get(k, out.metrics.get(k))
            sums[k] = sums.get(k, 0.0) + float(val.item())
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

    for epoch in range(cfg.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        for batch in pbar:
            batch = _move_to_device(batch, device)
            out = model(batch, pc_drop_prob=float(cfg.pc_drop_prob))

            loss = (
                cfg.w_pc * out.losses["loss_pc"]
                + cfg.w_intent * out.losses["loss_intent"]
                + cfg.w_eb * out.losses["loss_eb"]
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            pbar.set_postfix(
                loss=float(loss.item()),
                intent_acc=float(out.metrics["intent_acc"].item()),
                struct=float(out.metrics["struct_exact"].item()),
            )

        # Evaluate
        if val_loader is not None:
            stats = evaluate(model, val_loader, device)
            # Choose a selection metric (intent + structure)
            score = stats.get("intent_acc", 0.0) + 0.5 * stats.get("struct_exact", 0.0)
            print(f"Epoch {epoch}: val intent_acc={stats.get('intent_acc',0):.4f}, struct={stats.get('struct_exact',0):.4f}, score={score:.4f}")
            # Save last
            save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, epoch, best_score)
            if score > best_score:
                best_score = score
                save_checkpoint(str(best_path), model, optimizer, epoch, best_score)
                print(f"  New best checkpoint -> {best_path} (score={best_score:.4f})")
        else:
            # Save last always
            save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, epoch, best_score)

    print(f"Training done. Best score: {best_score:.4f}")
    print(f"Best checkpoint: {best_path}")
