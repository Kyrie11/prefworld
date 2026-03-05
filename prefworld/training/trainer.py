from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.utils import save_checkpoint
from prefworld.utils.calibration import finalize_bins, init_bins, save_reliability_artifacts, update_bins


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

    # EB regularizers
    eb_pref_sens_weight: float = 0.0
    eb_pref_sens_margin: float = 0.2
    eb_base_l2_weight: float = 0.0

    # loss mixing
    w_pc: float = 1.0
    w_intent: float = 0.0
    w_eb: float = 1.0

    # Guardrail: maneuver labels are derived from future trajectories and leak information.
    # Only set True if you intentionally run the leaky ablation.
    allow_future_label_leakage: bool = False

    # (Req-8) Maneuver posterior calibration diagnostics
    maneuver_calibration_bins: int = 15
    save_maneuver_reliability_diagram: bool = True


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

    # (Req-8) Calibration accumulators for maneuver posterior
    calib = init_bins(int(getattr(cfg, "maneuver_calibration_bins", 15)), device=device)
    total_correct = 0.0
    total_nll = 0.0
    total_count = 0.0
    # Per-maneuver stats
    num_m = 6  # Maneuver enum size
    per_count = torch.zeros((num_m,), device=device, dtype=torch.float32)
    per_correct = torch.zeros((num_m,), device=device, dtype=torch.float32)
    per_nll = torch.zeros((num_m,), device=device, dtype=torch.float32)

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
            eb_pref_sens_weight=float(cfg.eb_pref_sens_weight),
            eb_pref_sens_margin=float(cfg.eb_pref_sens_margin),
            eb_base_l2_weight=float(cfg.eb_base_l2_weight),
            use_pseudo_intent=cfg.w_intent > 0.0,
            allow_future_label_leakage=bool(cfg.allow_future_label_leakage),
        )

        keys = list(out.losses.keys()) + list(out.metrics.keys())
        for k in keys:
            v = out.losses.get(k, out.metrics.get(k))
            sums[k] = sums.get(k, 0.0) + float(v.item())

        # Diagnostics: maneuver entropy & z posterior std (non-ego agents)
        with torch.no_grad():
            agents_valid_last = (batch["agents_hist_mask"][:, :, -1] > 0.5)
            m_logits = out.aux.get("maneuver_logits_last", None)
            if m_logits is not None:
                p = torch.softmax(m_logits, dim=-1)
                H = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
                Hv = H[agents_valid_last]
                if Hv.numel() > 0:
                    sums["maneuver_entropy"] = sums.get("maneuver_entropy", 0.0) + float(Hv.mean().item())

                # (Req-8) Calibration / per-maneuver metrics (uses future-derived pseudo labels)
                labels = batch.get("agents_maneuver", None)
                if labels is not None:
                    labels = labels.to(torch.int64)
                    # valid agents only
                    mask = agents_valid_last & (labels >= 0) & (labels < p.shape[-1])
                    if mask.any():
                        pv = p[mask]
                        y = labels[mask]
                        conf, pred = pv.max(dim=-1)
                        correct = (pred == y)
                        calib = update_bins(calib, conf, correct)

                        # NLL for the true class
                        nll = -torch.log(pv.gather(-1, y.view(-1, 1)).clamp_min(1e-12)).squeeze(-1)
                        total_correct += float(correct.to(torch.float32).sum().item())
                        total_nll += float(nll.sum().item())
                        total_count += float(mask.to(torch.float32).sum().item())

                        # Per-maneuver
                        for mm in range(min(num_m, p.shape[-1])):
                            m_mask = (y == mm)
                            if m_mask.any():
                                per_count[mm] += m_mask.to(torch.float32).sum()
                                per_correct[mm] += correct[m_mask].to(torch.float32).sum()
                                per_nll[mm] += nll[m_mask].sum()

            z_logvar = out.aux.get("z_logvar", None)
            if z_logvar is not None:
                zstd = torch.exp(0.5 * z_logvar).mean(dim=-1)
                zv = zstd[agents_valid_last]
                if zv.numel() > 0:
                    sums["z_posterior_std"] = sums.get("z_posterior_std", 0.0) + float(zv.mean().item())
        counts += 1

    if counts == 0:
        return {k: 0.0 for k in sums}

    out_stats = {k: v / counts for k, v in sums.items()}

    # Finalize calibration stats
    if total_count > 0.0:
        bin_conf, bin_acc, ece = finalize_bins(calib)
        out_stats["maneuver_ece"] = float(ece.item())
        out_stats["maneuver_acc"] = float(total_correct / max(1.0, total_count))
        out_stats["maneuver_nll"] = float(total_nll / max(1.0, total_count))
        for mm in range(num_m):
            if float(per_count[mm].item()) > 0.0:
                out_stats[f"maneuver_acc_m{mm}"] = float((per_correct[mm] / per_count[mm]).item())
                out_stats[f"maneuver_nll_m{mm}"] = float((per_nll[mm] / per_count[mm]).item())

        # Save reliability diagram data / image (optional)
        if bool(getattr(cfg, "save_maneuver_reliability_diagram", True)):
            try:
                save_reliability_artifacts(
                    bin_conf,
                    bin_acc,
                    calib.count,
                    Path(cfg.output_dir) / "calibration",
                    tag="maneuver",
                )
            except Exception:
                pass

    return out_stats


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
                eb_pref_sens_weight=float(cfg.eb_pref_sens_weight),
                eb_pref_sens_margin=float(cfg.eb_pref_sens_margin),
                eb_base_l2_weight=float(cfg.eb_base_l2_weight),
                use_pseudo_intent=cfg.w_intent > 0.0,
                intent_weight=1.0,
                allow_future_label_leakage=bool(cfg.allow_future_label_leakage),
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

            # diagnostics
            with torch.no_grad():
                agents_valid_last = (batch["agents_hist_mask"][:, :, -1] > 0.5)
                H_mean = float("nan")
                zstd_mean = float("nan")

                m_logits = out.aux.get("maneuver_logits_last", None)
                if m_logits is not None:
                    p = torch.softmax(m_logits, dim=-1)
                    H = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
                    Hv = H[agents_valid_last]
                    if Hv.numel() > 0:
                        H_mean = float(Hv.mean().item())

                z_logvar = out.aux.get("z_logvar", None)
                if z_logvar is not None:
                    zstd = torch.exp(0.5 * z_logvar).mean(dim=-1)
                    zv = zstd[agents_valid_last]
                    if zv.numel() > 0:
                        zstd_mean = float(zv.mean().item())

            pbar.set_postfix(
                loss=float(loss.item()),
                pc=float(out.losses.get("loss_pc", torch.tensor(0.0, device=device)).item()),
                eb=float(out.losses.get("loss_eb", torch.tensor(0.0, device=device)).item()),
                intent_acc=float(out.metrics.get("intent_acc", torch.tensor(0.0)).item()),
                struct=float(out.metrics.get("struct_exact", torch.tensor(0.0)).item()),
                H=float(H_mean),
                zstd=float(zstd_mean),
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
