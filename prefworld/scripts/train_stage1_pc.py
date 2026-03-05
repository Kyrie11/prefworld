from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prefworld.data.dataset import CachedNuPlanDataset, collate_batch
from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.training.utils import save_checkpoint
from prefworld.utils.config import load_config


def _move_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def make_prefix_split(valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefix episode: random cutoff, context is prefix, query is suffix."""
    m = valid_mask > 0.5
    B, N, T = m.shape
    lengths = m.sum(dim=-1).clamp(min=0).to(torch.int64)
    cutoff = torch.zeros((B, N), device=m.device, dtype=torch.int64)
    ok = lengths >= 2
    if ok.any():
        rnd = torch.randint(0, 10_000, (int(ok.sum().item()),), device=m.device)
        cutoff[ok] = 1 + (rnd % (lengths[ok] - 1))
    t_idx = torch.arange(T, device=m.device).view(1, 1, T)
    ctx = (t_idx < cutoff.unsqueeze(-1)) & m
    qry = m & (~ctx)
    return ctx.to(torch.float32), qry.to(torch.float32)


def make_cross_template_split(
    tau: torch.Tensor,         # [B,N,T,Dt]
    valid_mask: torch.Tensor,  # [B,N,T]
    *,
    query_ratio: float,
    topk_frac: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-template episode split (paper Sec.4.1).

    We score token novelty by its distance to the *nearest other* valid token in τ-space
    (a cheap proxy for paper novelty: min_{t'\in C} ||h_\tau(t)-h_\tau(t')||).
    Tokens with larger novelty are more likely to be selected as queries.

    Returns ctx_mask, query_mask (float32).
    """
    m = valid_mask > 0.5
    B, N, T, Dt = tau.shape
    ctx = torch.zeros((B, N, T), device=tau.device, dtype=torch.float32)
    qry = torch.zeros_like(ctx)

    tau_det = tau.detach()

    for b in range(B):
        for n in range(N):
            idx = torch.nonzero(m[b, n], as_tuple=False).squeeze(-1)
            L = int(idx.numel())
            if L < 2:
                ctx[b, n, idx] = 1.0
                continue

            num_q = max(1, int(round(float(query_ratio) * float(L))))
            tau_seq = tau_det[b, n, idx]  # [L,Dt]

            # novelty(t) = min_{t'!=t} ||tau[t] - tau[t']||
            # Use cdist (L is small) and mask the diagonal.
            dist_mat = torch.cdist(tau_seq, tau_seq, p=2)  # [L,L]
            dist_mat = dist_mat + torch.eye(L, device=dist_mat.device, dtype=dist_mat.dtype) * 1e6
            novelty = dist_mat.min(dim=-1).values  # [L]

            # pick queries from top-k novelty pool
            k = max(1, int(round(float(topk_frac) * float(L))))
            top = torch.topk(novelty, k=k, largest=True).indices
            pool = idx[top]

            # if pool smaller than num_q, fall back to all
            if pool.numel() < num_q:
                pool = idx

            # take the top novelty values within pool for determinism
            novelty_pool = novelty[top]  # [k]
            q_sel = pool[torch.topk(novelty_pool, k=num_q, largest=True).indices]

            qry[b, n, q_sel] = 1.0
            ctx[b, n, idx] = 1.0
            ctx[b, n, q_sel] = 0.0

    return ctx, qry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to stage1_pc.yaml")
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

    # Stage-1: freeze EB modules
    for p in model.energy_net.parameters():
        p.requires_grad_(False)

    opt_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(opt_cfg.get("lr", 1e-3)),
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )

    out_dir = Path(cfg["train"]["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_path = ckpt_dir / "best.pt"

    tcfg = cfg["train"]

    for epoch in range(int(tcfg.get("max_epochs", 20))):
        model.train()
        pbar = tqdm(loader, desc=f"Stage1-PC epoch {epoch}")

        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, batch in enumerate(pbar):
            batch = _move_to_device(batch, device)
            # 在 move_to_device 之前检查（CPU 更快定位）


            # Encode templates once so we can build cross-template splits.
            with torch.set_grad_enabled(True):
                template_out, state_all, _ = model.encode_templates(batch)
                tau_all = template_out.tau

            agents_hist = batch["agents_hist"]
            agents_hist_mask = batch["agents_hist_mask"]

            # action tokens
            x = model._action_tokens_from_hist(agents_hist)              # [B,N,T-1,3]
            valid = (agents_hist_mask[:, :, 1:] * agents_hist_mask[:, :, :-1]).to(torch.float32)
            tau_agents = tau_all[:, 1:]                                   # [B,N,T,Dt]
            tau_action = tau_agents[:, :, :-1]
            ctx_action = state_all[:, 1:, :-1]

            # Episode type mixture
            mix_p = float(tcfg.get("pc_episode_mix_prefix_prob", 0.5))
            use_prefix = torch.rand(()) < mix_p

            if epoch < int(tcfg.get("pc_cross_template_warmup_epochs", 1)):
                use_prefix = True

            if bool(use_prefix):
                ctx_mask, query_mask = make_prefix_split(valid)
            else:
                ctx_mask, query_mask = make_cross_template_split(
                    tau_action,
                    valid,
                    query_ratio=float(tcfg.get("pc_query_ratio", 0.3)),
                    topk_frac=float(tcfg.get("pc_cross_template_topk_frac", 0.5)),
                )

            out = model(
                batch,
                run_pc=True,
                run_eb=False,
                pc_query_ratio=float(tcfg.get("pc_query_ratio", 0.3)),
                pc_ctx_mask_override=ctx_mask,
                pc_query_mask_override=query_mask,
                lambda_distill_mu=float(tcfg.get("lambda_distill_mu", 1.0)),
                lambda_distill_cov=float(tcfg.get("lambda_distill_cov", 0.05)),
                lambda_prior=float(tcfg.get("lambda_prior", 0.1)),
                lambda_con=float(tcfg.get("lambda_con", 0.05)),
                lambda_overlap=float(tcfg.get("lambda_overlap", 1e-3)),
                lambda_mod=float(tcfg.get("lambda_mod", 1e-3)),
                n_z_samples=int(tcfg.get("n_z_samples", 1)),
                free_bits=float(tcfg.get("free_bits", 0.0)),
            )

            loss = out.losses["loss_pc"]

            if not torch.isfinite(loss):
                print("Skip step due to non-finite loss. meta=", batch.get("_meta", None))
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip_norm", 5.0)))
            optimizer.step()

            # --------------------------------------------------------------
            # Diagnostics (Req-4): maneuver posterior entropy & z posterior std
            # --------------------------------------------------------------
            with torch.no_grad():
                m_logits = out.aux.get("maneuver_logits_last", None)
                z_logvar = out.aux.get("z_logvar", None)
                agents_valid_last = (agents_hist_mask[:, :, -1] > 0.5)

                H_mean = float("nan")
                H_p50 = float("nan")
                zstd_mean = float("nan")
                zstd_p50 = float("nan")

                if m_logits is not None:
                    p = torch.softmax(m_logits, dim=-1)
                    H = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)  # [B,N]
                    Hv = H[agents_valid_last]
                    if Hv.numel() > 0:
                        H_mean = float(Hv.mean().item())
                        H_p50 = float(torch.quantile(Hv, 0.5).item())

                if z_logvar is not None:
                    zstd = torch.exp(0.5 * z_logvar).mean(dim=-1)  # [B,N]
                    zv = zstd[agents_valid_last]
                    if zv.numel() > 0:
                        zstd_mean = float(zv.mean().item())
                        zstd_p50 = float(torch.quantile(zv, 0.5).item())

            pbar.set_postfix(
                loss=float(loss.item()),
                q_nll=float(out.losses.get("loss_pc_query_nll", torch.tensor(0.0)).item()),
                kl=float(out.losses.get("loss_pc_kl_ctx_prior", torch.tensor(0.0)).item()),
                con=float(out.losses.get("loss_pc_contrastive", torch.tensor(0.0)).item()),
                H=float(H_mean),
                H50=float(H_p50),
                zstd=float(zstd_mean),
                z50=float(zstd_p50),
            )

            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

        # checkpoint
        save_checkpoint(str(ckpt_dir / "last.pt"), model, optimizer, epoch, -best_loss)
        epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        if epoch_loss < best_loss:
            best_loss = float(epoch_loss)
            save_checkpoint(str(best_path), model, optimizer, epoch, -best_loss)

    print(f"Stage-1 done. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
