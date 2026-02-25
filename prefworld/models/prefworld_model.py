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
from prefworld.models.template_encoder import TemplateEncoder


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
        tau_dim: int = 64,
        pc_hidden: int = 64,
        intent_hidden: int = 128,
        energy_hidden: int = 128,
        eb_temperature: float = 1.0,
        eb_max_candidates: int = 64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.tau_dim = int(tau_dim)
        self.agent_feat_dim = agent_feat_dim

        # Template encoder τ (HD-map polylines + neighbor graph)
        self.template = TemplateEncoder(tau_dim=self.tau_dim, agent_state_dim=5, hidden_dim=intent_hidden)

        # Preference completion (paper-aligned): evidence tokens are action increments X and template τ
        # We use Dx=3 (Δx,Δy,Δyaw) and context Dc=5 (x,y,yaw,vx,vy).
        self.pc = PreferenceCompletion(x_dim=3, tau_dim=self.tau_dim, ctx_dim=5, z_dim=z_dim, hidden_dim=pc_hidden)
        # Intention net consumes agent history features and z, plus map and ego future plan
        self.intent = IntentionNet(
            agent_input_dim=agent_feat_dim,
            z_dim=z_dim,
            hidden_dim=intent_hidden,
            ctx_dim=intent_hidden,
            tau_dim=self.tau_dim,
        )

        # EB-STM energy net uses agent features + ego action one-hot appended (optional)
        # EB-STM energy net uses agent features + τ + ego action one-hot
        self.energy_net = EditFactorizedEnergyNet(agent_feat_dim=agent_feat_dim + self.tau_dim + NUM_MANEUVERS, z_dim=z_dim, hidden_dim=energy_hidden)
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
        tau_curr: torch.Tensor,           # [B,1+N,Dt]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build [B,K,(Da+Dt)+NUM_MANEUVERS] features and mask for EB-STM (K=1+N)."""
        B, N, D = agents_curr.shape
        K = 1 + N
        feat = torch.zeros((B, K, D + self.tau_dim + NUM_MANEUVERS), device=agents_curr.device, dtype=agents_curr.dtype)
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
        feat[:, 0, D : D + self.tau_dim] = tau_curr[:, 0]

        # Ego action one-hot
        onehot = F.one_hot(ego_maneuver.long().clamp(min=0, max=NUM_MANEUVERS - 1), num_classes=NUM_MANEUVERS).to(feat.dtype)
        feat[:, 0, D + self.tau_dim :] = onehot

        # Agents
        feat[:, 1:, :D] = agents_curr
        feat[:, 1:, D : D + self.tau_dim] = tau_curr[:, 1:]
        # action part for non-ego agents is zero
        return feat, mask

    @staticmethod
    def _action_tokens_from_hist(agents_hist: torch.Tensor) -> torch.Tensor:
        """Compute per-step action tokens X from agent history.

        X_t = [Δx, Δy, Δyaw] between consecutive history steps.
        agents_hist: [B,N,T,7] with (x,y,yaw,...) in ego frame.
        Returns: [B,N,T-1,3]
        """
        pos = agents_hist[..., 0:2]
        yaw = agents_hist[..., 2]
        dpos = pos[..., 1:, :] - pos[..., :-1, :]
        dyaw = torch.atan2(torch.sin(yaw[..., 1:] - yaw[..., :-1]), torch.cos(yaw[..., 1:] - yaw[..., :-1]))
        return torch.cat([dpos, dyaw.unsqueeze(-1)], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor], *, pc_drop_prob: float = 0.2) -> ModelOutput:
        # Unpack
        agents_hist = batch["agents_hist"]          # [B,N,T,7]
        agents_hist_mask = batch["agents_hist_mask"]# [B,N,T]
        agents_future = batch["agents_future"]      # [B,N,Tf,3] (for labels)
        agents_m = batch["agents_maneuver"]         # [B,N]
        map_polylines = batch["map_polylines"]      # [B,M,L,2]
        map_poly_mask = batch["map_poly_mask"]      # [B,M]
        map_poly_type = batch.get("map_poly_type", None)
        map_tl_status = batch.get("map_tl_status", None)
        map_on_route = batch.get("map_on_route", None)
        ego_future = batch["ego_future"]            # [B,Tf,3]
        ego_dyn_hist = batch["ego_dyn_hist"]        # [B,Tp,4]
        ego_hist = batch["ego_hist"]                # [B,Tp,3]
        ego_m = batch["ego_maneuver"].squeeze(-1)   # [B]
        A_t = batch["structure_t"]                  # [B,K,K]
        A_t1 = batch["structure_t1"]                # [B,K,K]

        B, N, T, D = agents_hist.shape
        agents_valid = (agents_hist_mask[:, :, -1] > 0.5).float()  # [B,N]

        # -----------------------------
        # Template encoding τ for ego + agents over history
        # -----------------------------
        ego_state_seq = torch.cat([ego_hist, ego_dyn_hist[..., 0:2]], dim=-1).unsqueeze(1)  # [B,1,T,5]
        agents_state_seq = agents_hist[..., :5]  # [B,N,T,5]
        state_all = torch.cat([ego_state_seq, agents_state_seq], dim=1)  # [B,1+N,T,5]
        mask_all = torch.cat([torch.ones((B, 1, T), device=agents_hist.device, dtype=agents_hist.dtype), agents_hist_mask], dim=1)
        tau_all = self.template(
            agents_state=state_all,
            agents_mask=mask_all,
            map_polylines=map_polylines,
            map_poly_mask=map_poly_mask,
            map_poly_type=map_poly_type,
            map_tl_status=map_tl_status,
            map_on_route=map_on_route,
        ).tau  # [B,1+N,T,Dt]
        tau_agents = tau_all[:, 1:]  # [B,N,T,Dt]

        # -----------------------------
        # Preference completion (paper objective)
        # Evidence tokens are action increments X_t and templates τ_t.
        # -----------------------------
        x_action = self._action_tokens_from_hist(agents_hist)               # [B,N,T-1,3]
        mask_action = (agents_hist_mask[:, :, 1:] * agents_hist_mask[:, :, :-1]).to(dtype=agents_hist.dtype)  # [B,N,T-1]
        tau_action = tau_agents[:, :, :-1, :]                               # [B,N,T-1,Dt]
        ctx_action = agents_state_seq[:, :, :-1, :]                         # [B,N,T-1,5]

        pc_out = self.pc(
            x=x_action,
            tau=tau_action,
            ctx=ctx_action,
            mask=mask_action,
            split_mode="random",
            query_ratio=float(pc_drop_prob),
            lambda_kl=1.0,
            lambda_prior=1.0,
            lambda_prec=1e-3,
            n_z_samples=1,
            free_bits=0.0,
        )

        post_full = pc_out.post_full
        post_part = pc_out.post_ctx  # context posterior for downstream
        loss_pc = pc_out.loss_total

        # Intention prediction (condition on ego future plan)
        z = post_part.q.rsample()  # [B,N,Dz]
        logits = self.intent(
            agents_hist=agents_hist,
            agents_hist_mask=agents_hist_mask,
            z=z,
            map_polylines=map_polylines,
            map_poly_mask=map_poly_mask,
            ego_future=ego_future,
            tau_curr=tau_agents[:, :, -1, :],
        )
        # Cross entropy per agent
        ce = F.cross_entropy(logits.reshape(B * N, -1), agents_m.reshape(B * N), reduction="none").reshape(B, N)
        loss_intent = masked_mean(ce, agents_valid)

        # EB-STM structure prediction
        # Current agent state for EB uses last history step
        agents_curr = agents_hist[:, :, -1, :]  # [B,N,7]
        ego_dyn_curr = ego_dyn_hist[:, -1, :]   # [B,4]
        tau_curr = tau_all[:, :, -1, :]  # [B,1+N,Dt]
        eb_feat, eb_mask = self._build_agent_state_for_eb(ego_dyn_curr, ego_m, agents_curr, agents_valid, tau_curr)

        # Preferences for EB include ego at index 0 as standard normal (mean=0, logvar=0)
        z_mean = torch.zeros((B, 1 + N, self.z_dim), device=agents_hist.device, dtype=agents_hist.dtype)
        z_logvar = torch.zeros_like(z_mean)
        z_mean[:, 1:] = post_part.q.mean
        z_logvar[:, 1:] = post_part.q.logvar

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
            "loss_pc_query_nll": pc_out.loss_query_nll,
            "loss_pc_kl_full_ctx": pc_out.loss_kl_full_ctx,
            "loss_pc_kl_ctx_prior": pc_out.loss_kl_ctx_prior,
            "loss_pc_prec_reg": pc_out.loss_prec_reg,
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
