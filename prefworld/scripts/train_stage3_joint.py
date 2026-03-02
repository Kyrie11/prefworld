from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch
from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.utils import load_modules_from_checkpoint, save_checkpoint
from prefworld.utils.config import load_config


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to stage3_joint.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    ds_cfg = cfg["dataset"]
    dataset = CachedNuPlanDataset(ds_cfg["cache_dir"], ds_cfg["split"], max_samples=ds_cfg.get("max_samples"))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True,
    )

    mcfg = cfg["model"]
    model = PrefWorldModel(
        agent_feat_dim=int(mcfg.get("agent_feat_dim", 7)),
        z_dim=int(mcfg.get("z_dim", 8)),
        tau_dim=int(mcfg.get("tau_dim", 64)),
        pc_hidden=int(mcfg.get("pc_hidden", 128)),
        template_hidden=int(mcfg.get("template_hidden", 128)),
        energy_hidden=int(mcfg.get("energy_hidden", 128)),
        eb_temperature=float(mcfg.get("eb_temperature", 1.0)),
        eb_max_candidates=int(mcfg.get("eb_max_candidates", 64)),
    ).to(device)

    tcfg = cfg["train"]

    # Initialize from stage-1 and stage-2
    stage1_ckpt = tcfg.get("stage1_checkpoint")
    if stage1_ckpt:
        info = load_modules_from_checkpoint(stage1_ckpt, model, ("template", "pc"), strict=False)
        print(f"Loaded stage1 modules from {stage1_ckpt}: {info}")
    stage2_ckpt = tcfg.get("stage2_checkpoint")
    if stage2_ckpt:
        info = load_modules_from_checkpoint(stage2_ckpt, model, ("energy_net",), strict=False)
        print(f"Loaded stage2 modules from {stage2_ckpt}: {info}")

    opt_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get("lr", 2e-4)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )

    out_dir = Path(tcfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_path = ckpt_dir / "best.pt"

    w_pc = float(tcfg.get("w_pc", 1.0))
    w_eb = float(tcfg.get("w_eb", 1.0))

    for epoch in range(int(tcfg.get("max_epochs", 10))):
        model.train()
        pbar = tqdm(loader, desc=f"Stage3-Joint epoch {epoch}")

        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in pbar:
            batch = _move_to_device(batch, device)

            out = model(
                batch,
                run_pc=True,
                run_eb=True,
                detach_pc_for_eb=False,
                pc_query_ratio=float(tcfg.get("pc_query_ratio", 0.3)),
                lambda_distill_mu=float(tcfg.get("lambda_distill_mu", 1.0)),
                lambda_distill_cov=float(tcfg.get("lambda_distill_cov", 0.05)),
                lambda_prior=float(tcfg.get("lambda_prior", 0.1)),
                lambda_con=float(tcfg.get("lambda_con", 0.05)),
                lambda_overlap=float(tcfg.get("lambda_overlap", 1e-3)),
                lambda_mod=float(tcfg.get("lambda_mod", 1e-3)),
                n_z_samples=int(tcfg.get("n_z_samples", 1)),
                free_bits=float(tcfg.get("free_bits", 0.0)),
                eb_smooth_scale=float(tcfg.get("eb_smooth_scale", 0.0)),
                eb_phys_dist_threshold_m=float(tcfg.get("eb_phys_dist_threshold_m", 1e9)),
                eb_phys_penalty_scale=float(tcfg.get("eb_phys_penalty_scale", 0.0)),
                eb_cf_weight=float(tcfg.get("eb_cf_weight", 0.0)),
                eb_cf_base_temperature=float(tcfg.get("eb_cf_base_temperature", 2.0)),
                eb_cf_actions=tuple(tcfg.get("eb_cf_actions", (0, 1, 2))),
            )

            eb_loss_key = "loss_eb_total" if "loss_eb_total" in out.losses else "loss_eb"
            loss = w_pc * out.losses["loss_pc"] + w_eb * out.losses[eb_loss_key]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip_norm", 5.0)))
            optimizer.step()

            pbar.set_postfix(
                loss=float(loss.item()),
                pc=float(out.losses["loss_pc"].item()),
                eb=float(out.losses["loss_eb"].item()),
                cf=float(out.losses.get("loss_eb_cf", torch.tensor(0.0)).item()),
                struct=float(out.metrics.get("struct_exact", torch.tensor(0.0)).item()),
            )

            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

        save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, epoch, -best_loss)
        epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        if epoch_loss < best_loss:
            best_loss = float(epoch_loss)
            save_checkpoint(str(best_path), model, optimizer, epoch, -best_loss)

    print(f"Stage-3 done. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
