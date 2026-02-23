from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.data.labels import NUM_MANEUVERS
from prefworld.models.eb_stm import EBSTM
from prefworld.models.efen import EditFactorizedEnergyNet
from prefworld.models.intention_net import IntentionNet
from prefworld.models.preference_completion import PreferenceCompletion


@dataclass
class ModelOutput:
    losses: Dict[str, torch.Tensor]
    metrics: Dict[str, torch.Tensor]
    aux: Dict[str, torch.Tensor]


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / (mask.sum() + eps)
    return (x * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)


class PrefWorldModel(nn.Module):
    """Full model: preference completion + intention + EB-STM."""

    def __init__(
        self,
        agent_feat_dim: int = 7,
        z_dim: int = 8,
        pc_hidden: int = 64,
        intent_hidden: int = 128,
        energy_hidden: int = 128,
        eb_temperature: float = 1.0,
        eb_max_candidates: int = 64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.agent_feat_dim = agent_feat_dim

        self.pc = PreferenceCompletion(agent_input_dim=agent_feat_dim, z_dim=z_dim, hidden_dim=pc_hidden)
        # Intention net consumes agent history features and z, plus map and ego future plan
        self.intent = IntentionNet(agent_input_dim=agent_feat_dim, z_dim=z_dim, hidden_dim=intent_hidden, ctx_dim=intent_hidden)

        # EB-STM energy net uses agent features + ego action one-hot appended (optional)
        self.energy_net = EditFactorizedEnergyNet(agent_feat_dim=agent_feat_dim + NUM_MANEUVERS, z_dim=z_dim, hidden_dim=energy_hidden)
        self.ebstm = EBSTM(self.energy_net, temperature=eb_temperature, max_candidates=eb_max_candidates)

        # constants for ego vehicle size (approx)
        self.ego_length = 4.8
        self.ego_width = 2.0

    def _build_agent_state_for_eb(
        self,
        ego_dyn: torch.Tensor,            # [B,4] vx,vy,ax,ay in global? here we store in dataset as global but we can use local approx
        ego_maneuver: torch.Tensor,       # [B] int
        agents_curr: torch.Tensor,        # [B,N,7]
        agents_valid: torch.Tensor,       # [B,N] float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build [B,K,Da+NUM_MANEUVERS] features and mask for EB-STM (K=1+N)."""
        B, N, D = agents_curr.shape
        K = 1 + N
        feat = torch.zeros((B, K, D + NUM_MANEUVERS), device=agents_curr.device, dtype=agents_curr.dtype)
        mask = torch.zeros((B, K), device=agents_curr.device, dtype=agents_curr.dtype)
        mask[:, 0] = 1.0
        mask[:, 1:] = agents_valid

        # Ego current pose is origin in ego frame: x=y=yaw=0
        # Use ego velocity from ego_dyn_hist last step (already in dataset as vx,vy,ax,ay in global frame).
        # We approximate local == global here; for better accuracy, rotate by ego yaw before caching.
        vx = ego_dyn[:, 0]
        vy = ego_dyn[:, 1]
        ego_base = torch.stack([torch.zeros_like(vx), torch.zeros_like(vx), torch.zeros_like(vx), vx, vy, torch.full_like(vx, self.ego_length), torch.full_like(vx, self.ego_width)], dim=-1)
        feat[:, 0, :D] = ego_base

        # Ego action one-hot
        onehot = F.one_hot(ego_maneuver.long().clamp(min=0, max=NUM_MANEUVERS - 1), num_classes=NUM_MANEUVERS).to(feat.dtype)
        feat[:, 0, D:] = onehot

        # Agents
        feat[:, 1:, :D] = agents_curr
        # action part for non-ego agents is zero
        return feat, mask

    def forward(self, batch: Dict[str, torch.Tensor], *, pc_drop_prob: float = 0.2) -> ModelOutput:
        # Unpack
        agents_hist = batch["agents_hist"]          # [B,N,T,7]
        agents_hist_mask = batch["agents_hist_mask"]# [B,N,T]
        agents_future = batch["agents_future"]      # [B,N,Tf,3] (for labels)
        agents_m = batch["agents_maneuver"]         # [B,N]
        map_polylines = batch["map_polylines"]      # [B,M,L,2]
        map_poly_mask = batch["map_poly_mask"]      # [B,M]
        ego_future = batch["ego_future"]            # [B,Tf,3]
        ego_dyn_hist = batch["ego_dyn_hist"]        # [B,Tp,4]
        ego_m = batch["ego_maneuver"].squeeze(-1)   # [B]
        A_t = batch["structure_t"]                  # [B,K,K]
        A_t1 = batch["structure_t1"]                # [B,K,K]

        B, N, T, D = agents_hist.shape
        agents_valid = (agents_hist_mask[:, :, -1] > 0.5).float()  # [B,N]

        # Preference completion: full and partial posteriors
        post_full = self.pc(agents_hist, agents_hist_mask, drop_prob=0.0)
        post_part = self.pc(agents_hist, agents_hist_mask, drop_prob=float(pc_drop_prob))

        # KL(q_full || q_part) + KL(q_full || prior)
        qf = post_full.q
        qp = post_part.q
        # compute per-agent KLs
        kl_fp = qf.kl_to(qp)  # [B,N]
        kl_prior = qf.kl_to_standard_normal()  # [B,N]
        loss_pc = masked_mean(kl_fp, agents_valid)
        loss_kl = masked_mean(kl_prior, agents_valid)

        # Intention prediction (condition on ego future plan)
        z = post_part.q.rsample()  # [B,N,Dz]
        logits = self.intent(
            agents_hist=agents_hist,
            agents_hist_mask=agents_hist_mask,
            z=z,
            map_polylines=map_polylines,
            map_poly_mask=map_poly_mask,
            ego_future=ego_future,
        )
        # Cross entropy per agent
        ce = F.cross_entropy(logits.reshape(B * N, -1), agents_m.reshape(B * N), reduction="none").reshape(B, N)
        loss_intent = masked_mean(ce, agents_valid)

        # EB-STM structure prediction
        # Current agent state for EB uses last history step
        agents_curr = agents_hist[:, :, -1, :]  # [B,N,7]
        ego_dyn_curr = ego_dyn_hist[:, -1, :]   # [B,4]
        eb_feat, eb_mask = self._build_agent_state_for_eb(ego_dyn_curr, ego_m, agents_curr, agents_valid)

        # Preferences for EB include ego at index 0 as standard normal (mean=0, logvar=0)
        z_mean = torch.zeros((B, 1 + N, self.z_dim), device=agents_hist.device, dtype=agents_hist.dtype)
        z_logvar = torch.zeros_like(z_mean)
        z_mean[:, 1:] = post_full.q.mean
        z_logvar[:, 1:] = post_full.q.logvar

        out_eb = self.ebstm(A_t, eb_feat, z_mean, z_logvar, eb_mask, oracle_next=A_t1)
        oracle_idx = self.ebstm.oracle_index(out_eb.candidate_A, A_t1)
        logp_oracle = torch.log(out_eb.probs[torch.arange(B, device=agents_hist.device), oracle_idx] + 1e-8)
        loss_eb = -logp_oracle.mean()

        # Metrics
        with torch.no_grad():
            pred_m = logits.argmax(dim=-1)  # [B,N]
            acc = (pred_m == agents_m).float()
            acc = masked_mean(acc, agents_valid)

            # Structure accuracy: compare argmin energy candidate with oracle
            pred_idx = out_eb.energies.argmin(dim=1)
            A_pred = out_eb.candidate_A[torch.arange(B, device=agents_hist.device), pred_idx]
            # exact match rate
            match = (A_pred == A_t1).all(dim=-1).all(dim=-1).float()
            struct_acc = match.mean()

        losses = {
            "loss_pc": loss_pc,
            "loss_kl": loss_kl,
            "loss_intent": loss_intent,
            "loss_eb": loss_eb,
        }
        metrics = {
            "intent_acc": acc,
            "struct_exact": struct_acc,
        }
        aux = {
            "logits": logits.detach(),
            "z_mean": post_part.q.mean.detach(),
            "z_logvar": post_part.q.logvar.detach(),
        }
        return ModelOutput(losses=losses, metrics=metrics, aux=aux)
