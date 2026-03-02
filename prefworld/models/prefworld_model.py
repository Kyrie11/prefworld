from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefworld.data.labels import NUM_MANEUVERS
from prefworld.models.eb_stm import EBSTM
from prefworld.models.efen_edit import EditFactorizedEnergyNetEdit
from prefworld.models.preference_completion import PreferenceCompletion
from prefworld.models.template_encoder import TemplateEncoder, TemplateEncoding


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
    """PrefWorld end-to-end model (Template encoder + Preference Completion + EB-STM).

    This implementation follows the paper structure:
      1) Preference completion (PC): evidence accumulation in natural parameters and
         query-likelihood under q(z|C).
      2) Edit-factorized energy model (EFEN) and EB-STM for structural rollouts.

    The repo keeps maneuver labels *optional* (only for metrics / ablations) and treats
    maneuvers as latent by default.

    Stage training (recommended):
      - Stage1: train template+PC
      - Stage2: freeze template+PC, train EFEN+EBSTM
      - Stage3: joint fine-tune
    """

    def __init__(
        self,
        agent_feat_dim: int = 7,
        z_dim: int = 8,
        tau_dim: int = 64,
        pc_hidden: int = 128,
        template_hidden: int = 128,
        energy_hidden: int = 128,
        eb_temperature: float = 1.0,
        eb_max_candidates: int = 64,
        num_maneuvers: int = NUM_MANEUVERS,
        # EFEN constants
        lambda_u: float = 1.0,
        return_token_params: bool = True,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.tau_dim = int(tau_dim)
        self.agent_feat_dim = int(agent_feat_dim)
        self.num_maneuvers = int(num_maneuvers)

        # Template encoder τ (HD-map polylines + neighbor graph)
        self.template = TemplateEncoder(tau_dim=self.tau_dim, agent_state_dim=5, hidden_dim=int(template_hidden))

        # Preference completion on action tokens
        self.pc = PreferenceCompletion(
            x_dim=3,
            tau_dim=self.tau_dim,
            ctx_dim=5,
            z_dim=self.z_dim,
            hidden_dim=int(pc_hidden),
            num_maneuvers=self.num_maneuvers,
        )

        # EFEN / EB-STM (paper-faithful edit-token transition energy)
        self.energy_net = EditFactorizedEnergyNetEdit(
            agent_feat_dim=self.agent_feat_dim + self.tau_dim + self.num_maneuvers,
            z_dim=self.z_dim,
            hidden_dim=int(energy_hidden),
            lambda_u=float(lambda_u),
            return_token_params=bool(return_token_params),
        )
        self.ebstm = EBSTM(self.energy_net, temperature=float(eb_temperature), max_candidates=int(eb_max_candidates))

        # constants for ego vehicle size (approx)
        self.ego_length = 4.8
        self.ego_width = 2.0

        # Save constructor hyper-parameters in checkpoints for robust evaluation.
        # This prevents config/checkpoint mismatches (e.g., pc_hidden) from breaking loading.
        self.hparams: Dict[str, object] = {
            "agent_feat_dim": int(agent_feat_dim),
            "z_dim": int(z_dim),
            "tau_dim": int(tau_dim),
            "pc_hidden": int(pc_hidden),
            "template_hidden": int(template_hidden),
            "energy_hidden": int(energy_hidden),
            "eb_temperature": float(eb_temperature),
            "eb_max_candidates": int(eb_max_candidates),
            "num_maneuvers": int(num_maneuvers),
            "lambda_u": float(lambda_u),
            "return_token_params": bool(return_token_params),
        }

    def encode_templates(self, batch: Dict[str, torch.Tensor]) -> Tuple[TemplateEncoding, torch.Tensor, torch.Tensor]:
        """Compute template encodings for ego+agents over history."""
        agents_hist = batch["agents_hist"]            # [B,N,Tp,7]
        agents_hist_mask = batch["agents_hist_mask"]  # [B,N,Tp]
        map_polylines = batch["map_polylines"]        # [B,M,L,2]
        map_poly_mask = batch["map_poly_mask"]        # [B,M]
        map_poly_type = batch.get("map_poly_type", None)
        map_tl_status = batch.get("map_tl_status", None)
        map_on_route = batch.get("map_on_route", None)
        ego_dyn_hist = batch["ego_dyn_hist"]          # [B,Tp,4]
        ego_hist = batch["ego_hist"]                  # [B,Tp,3]

        B, N, T, _ = agents_hist.shape

        ego_state_seq = torch.cat([ego_hist, ego_dyn_hist[..., 0:2]], dim=-1).unsqueeze(1)  # [B,1,T,5]
        agents_state_seq = agents_hist[..., :5]                                             # [B,N,T,5]
        state_all = torch.cat([ego_state_seq, agents_state_seq], dim=1)                      # [B,1+N,T,5]
        mask_all = torch.cat(
            [torch.ones((B, 1, T), device=agents_hist.device, dtype=agents_hist.dtype), agents_hist_mask],
            dim=1,
        )

        template_out = self.template(
            agents_state=state_all,
            agents_mask=mask_all,
            map_polylines=map_polylines,
            map_poly_mask=map_poly_mask,
            map_poly_type=map_poly_type,
            map_tl_status=map_tl_status,
            map_on_route=map_on_route,
        )

        return template_out, state_all, mask_all

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

    def _build_agent_state_for_eb(
        self,
        ego_dyn: torch.Tensor,            # [B,4] vx,vy,ax,ay (ego-local)
        ego_maneuver: torch.Tensor,       # [B] int
        agents_curr: torch.Tensor,        # [B,N,7]
        agents_valid: torch.Tensor,       # [B,N] float
        tau_curr: torch.Tensor,           # [B,1+N,Dt]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build [B,K,(Da+Dt)+M] features and mask for EB-STM (K=1+N)."""
        B, N, D = agents_curr.shape
        K = 1 + N
        feat = torch.zeros((B, K, D + self.tau_dim + self.num_maneuvers), device=agents_curr.device, dtype=agents_curr.dtype)
        mask = torch.zeros((B, K), device=agents_curr.device, dtype=agents_curr.dtype)
        mask[:, 0] = 1.0
        mask[:, 1:] = agents_valid

        # Ego current pose is origin in ego frame: x=y=yaw=0.
        vx = ego_dyn[:, 0]
        vy = ego_dyn[:, 1]
        ego_base = torch.stack(
            [
                torch.zeros_like(vx),
                torch.zeros_like(vx),
                torch.zeros_like(vx),
                vx,
                vy,
                torch.full_like(vx, self.ego_length),
                torch.full_like(vx, self.ego_width),
            ],
            dim=-1,
        )
        feat[:, 0, :D] = ego_base
        feat[:, 0, D : D + self.tau_dim] = tau_curr[:, 0]

        # Ego action one-hot
        onehot = F.one_hot(ego_maneuver.long().clamp(min=0, max=self.num_maneuvers - 1), num_classes=self.num_maneuvers).to(feat.dtype)
        feat[:, 0, D + self.tau_dim :] = onehot

        # Agents
        feat[:, 1:, :D] = agents_curr
        feat[:, 1:, D : D + self.tau_dim] = tau_curr[:, 1:]
        # action part for non-ego agents is zero
        return feat, mask

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        # toggles
        run_pc: bool = True,
        run_eb: bool = True,
        detach_pc_for_eb: bool = True,
        # PC episode split (if not overridden)
        pc_split_mode: str = "random",
        pc_query_ratio: float = 0.3,
        pc_ctx_mask_override: Optional[torch.Tensor] = None,
        pc_query_mask_override: Optional[torch.Tensor] = None,
        # PC weights
        lambda_distill_mu: float = 1.0,
        lambda_distill_cov: float = 0.05,
        lambda_prior: float = 0.1,
        lambda_con: float = 0.05,
        lambda_overlap: float = 1e-3,
        lambda_mod: float = 1e-3,
        n_z_samples: int = 1,
        free_bits: float = 0.0,
        # EB penalties (optional)
        eb_smooth_scale: float = 0.0,
        eb_phys_dist_threshold_m: float = 1e9,
        eb_phys_penalty_scale: float = 0.0,
        # EB counterfactual conservatism regularizer (paper Eq.(26))
        eb_cf_weight: float = 0.0,
        eb_cf_base_temperature: float = 2.0,
        eb_cf_actions: Optional[Tuple[int, ...]] = None,
        # Optional weak supervision for maneuver metrics
        use_pseudo_intent: bool = False,
        intent_weight: float = 1.0,
    ) -> ModelOutput:
        # Unpack
        agents_hist = batch["agents_hist"]            # [B,N,Tp,7]
        agents_hist_mask = batch["agents_hist_mask"]  # [B,N,Tp]
        agents_m = batch.get("agents_maneuver", None)  # [B,N]
        ego_dyn_hist = batch["ego_dyn_hist"]          # [B,Tp,4]
        ego_hist = batch["ego_hist"]                  # [B,Tp,3]
        ego_m = batch["ego_maneuver"].squeeze(-1)      # [B]

        # Prefer rule-based A_t when available to avoid future leakage in online usage.
        A_t = batch.get("structure_t_rule", batch.get("structure_t", None))  # [B,K,K]
        A_t1 = batch.get("structure_t1", None)                                   # [B,K,K]

        B, N, T, _ = agents_hist.shape
        agents_valid = (agents_hist_mask[:, :, -1] > 0.5).float()  # [B,N]

        # -----------------------------
        # Template encoding τ
        # -----------------------------
        template_out, state_all, mask_all = self.encode_templates(batch)
        tau_all = template_out.tau
        tau_agents = tau_all[:, 1:]  # [B,N,T,Dt]
        feasible_all = template_out.feasible_actions

        # -----------------------------
        # Preference completion
        # -----------------------------
        x_action = self._action_tokens_from_hist(agents_hist)  # [B,N,T-1,3]
        mask_action = (agents_hist_mask[:, :, 1:] * agents_hist_mask[:, :, :-1]).to(dtype=agents_hist.dtype)  # [B,N,T-1]
        tau_action = tau_agents[:, :, :-1, :]                   # [B,N,T-1,Dt]
        ctx_action = state_all[:, 1:, :-1, :]                    # [B,N,T-1,5]
        feasible_action = None
        if feasible_all is not None:
            feasible_action = feasible_all[:, 1:, :-1, :]

        if run_pc:
            pc_out = self.pc(
                x=x_action,
                tau=tau_action,
                ctx=ctx_action,
                mask=mask_action,
                feasible_actions=feasible_action,
                split_mode=str(pc_split_mode),
                query_ratio=float(pc_query_ratio),
                lambda_distill_mu=float(lambda_distill_mu),
                lambda_distill_cov=float(lambda_distill_cov),
                lambda_prior=float(lambda_prior),
                lambda_con=float(lambda_con),
                lambda_overlap=float(lambda_overlap),
                lambda_mod=float(lambda_mod),
                n_z_samples=int(n_z_samples),
                free_bits=float(free_bits),
                ctx_mask_override=pc_ctx_mask_override,
                query_mask_override=pc_query_mask_override,
                ensure_query_nonempty=float(pc_query_ratio) > 0.0,
            )
            post_part = pc_out.post_ctx
            loss_pc = pc_out.loss_total
            maneuver_logits_last = pc_out.maneuver_logits_last
        else:
            pc_out = None
            # standard normal for non-ego agents
            post_part = None
            loss_pc = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)
            maneuver_logits_last = torch.zeros((B, N, self.num_maneuvers), device=agents_hist.device, dtype=agents_hist.dtype)

        # -------------------------------------------------
        # Preference belief tensors (ego + other agents)
        #
        # Bug-fix: These must be populated whenever run_pc=True, even if run_eb=False.
        # Planner/PCI inference consumes aux["z_mean"/"z_logvar"]. Previously they were
        # only filled inside the run_eb branch, which caused zero beliefs during planning.
        # -------------------------------------------------
        z_mean = torch.zeros((B, 1 + N, self.z_dim), device=agents_hist.device, dtype=agents_hist.dtype)
        z_logvar = torch.zeros_like(z_mean)
        if post_part is not None:
            z_mean[:, 1:] = post_part.q.mean
            z_logvar[:, 1:] = post_part.q.logvar

        # maneuver metric (optional)
        with torch.no_grad():
            if agents_m is not None:
                pred_m = maneuver_logits_last.argmax(dim=-1)
                acc = masked_mean((pred_m == agents_m).float(), agents_valid)
            else:
                acc = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)

        # Optional weak supervision for maneuver (ablations)
        if use_pseudo_intent and agents_m is not None:
            ce = F.cross_entropy(maneuver_logits_last.reshape(B * N, -1), agents_m.reshape(B * N), reduction="none").reshape(B, N)
            loss_intent = intent_weight * masked_mean(ce, agents_valid)
        else:
            loss_intent = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)

        # -----------------------------
        # EB-STM structure prediction (one-step)
        # -----------------------------
        if run_eb:
            if A_t is None:
                raise KeyError("structure_t_rule/structure_t missing in batch but run_eb=True")

            agents_curr = agents_hist[:, :, -1, :]  # [B,N,7]
            ego_dyn_curr = ego_dyn_hist[:, -1, :]   # [B,4]
            tau_curr = tau_all[:, :, -1, :]         # [B,1+N,Dt]
            eb_feat, eb_mask = self._build_agent_state_for_eb(ego_dyn_curr, ego_m, agents_curr, agents_valid, tau_curr)

            # Detach preference belief if we don't want EB gradients flowing into PC.
            z_mean_eb = z_mean.detach() if detach_pc_for_eb else z_mean
            z_logvar_eb = z_logvar.detach() if detach_pc_for_eb else z_logvar

            out_eb = self.ebstm(
                A_t,
                eb_feat,
                z_mean_eb,
                z_logvar_eb,
                eb_mask,
                oracle_next=A_t1,
                smooth_scale=float(eb_smooth_scale),
                phys_dist_threshold_m=float(eb_phys_dist_threshold_m),
                phys_penalty_scale=float(eb_phys_penalty_scale),
            )
            if A_t1 is not None:
                oracle_idx = self.ebstm.oracle_index(out_eb.candidate_A, A_t1)
                logp_oracle = torch.log(out_eb.probs[torch.arange(B, device=agents_hist.device), oracle_idx] + 1e-8)
                loss_eb = -logp_oracle.mean()

                with torch.no_grad():
                    pred_idx = out_eb.energies.argmin(dim=1)
                    A_pred = out_eb.candidate_A[torch.arange(B, device=agents_hist.device), pred_idx]
                    struct_acc = (A_pred == A_t1).all(dim=-1).all(dim=-1).float().mean()
            else:
                loss_eb = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)
                struct_acc = torch.tensor(float("nan"), device=agents_hist.device, dtype=agents_hist.dtype)

            # -------------------------------------------------
            # Counterfactual conservatism regularizer (optional)
            # -------------------------------------------------
            loss_cf = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            if float(eb_cf_weight) > 0.0:
                # Small coarse action set by default (KEEP / LCL / LCR)
                if eb_cf_actions is None:
                    eb_cf_actions = (0, 1, 2)

                T_base = float(eb_cf_base_temperature)
                T_base = max(1e-6, T_base)

                for a in eb_cf_actions:
                    a = int(a)

                    # Skip per-batch items where this equals the logged action.
                    cf_mask = (ego_m.long() != int(a)).to(torch.float32)  # [B]
                    if cf_mask.sum() < 0.5:
                        continue

                    eb_feat_a, eb_mask_a = self._build_agent_state_for_eb(
                        ego_dyn_curr,
                        torch.full_like(ego_m, a),
                        agents_curr,
                        agents_valid,
                        tau_curr,
                    )
                    out_a = self.ebstm(
                        A_t,
                        eb_feat_a,
                        z_mean_eb,
                        z_logvar_eb,
                        eb_mask_a,
                        oracle_next=None,
                        smooth_scale=float(eb_smooth_scale),
                        phys_dist_threshold_m=float(eb_phys_dist_threshold_m),
                        phys_penalty_scale=float(eb_phys_penalty_scale),
                    )

                    # Base distribution: only smooth + phys energies (paper Eq.(25)).
                    smooth_E = self.ebstm._smooth_penalty(A_t, out_a.candidate_A)  # [B,C]
                    phys_E = self.ebstm._phys_penalty(
                        eb_feat_a,
                        out_a.candidate_A,
                        out_a.pair_energies.pair_mask,
                        float(eb_phys_dist_threshold_m),
                    )
                    base_E = float(eb_smooth_scale) * smooth_E + float(eb_phys_penalty_scale) * phys_E

                    base_logits = -(base_E / T_base)
                    base_logits = base_logits.masked_fill(~out_a.candidate_mask, -1e9)
                    base_probs = torch.softmax(base_logits, dim=-1)

                    p = out_a.probs.clamp_min(1e-12)
                    q = base_probs.clamp_min(1e-12)
                    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)  # [B]
                    # average only over valid cf items
                    loss_cf = loss_cf + (kl * cf_mask).sum() / (cf_mask.sum() + 1e-6)

            loss_eb_total = loss_eb + float(eb_cf_weight) * loss_cf
        else:
            loss_eb = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)
            struct_acc = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)
            out_eb = None
            loss_cf = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            loss_eb_total = loss_eb

        # -----------------------------
        # Collect
        # -----------------------------
        losses: Dict[str, torch.Tensor] = {
            "loss_pc": loss_pc,
            "loss_intent": loss_intent,
            "loss_eb": loss_eb,
            "loss_eb_cf": loss_cf,
            "loss_eb_total": loss_eb_total,
        }
        metrics: Dict[str, torch.Tensor] = {
            "intent_acc": acc,
            "struct_exact": struct_acc,
        }
        aux: Dict[str, torch.Tensor] = {
            "maneuver_logits_last": maneuver_logits_last.detach(),
            "z_mean": z_mean[:, 1:].detach() if z_mean.shape[1] > 1 else z_mean.detach(),
            "z_logvar": z_logvar[:, 1:].detach() if z_logvar.shape[1] > 1 else z_logvar.detach(),
        }

        if pc_out is not None:
            losses.update(
                {
                    "loss_pc_query_nll": pc_out.loss_query_nll,
                    "loss_pc_distill_mu": pc_out.loss_distill_mu,
                    "loss_pc_distill_cov": pc_out.loss_distill_cov,
                    "loss_pc_kl_ctx_prior": pc_out.loss_kl_ctx_prior,
                    "loss_pc_contrastive": pc_out.loss_contrastive,
                    "loss_pc_overlap": pc_out.loss_overlap,
                    "loss_pc_mod": pc_out.loss_modulation,
                }
            )

        return ModelOutput(losses=losses, metrics=metrics, aux=aux)
