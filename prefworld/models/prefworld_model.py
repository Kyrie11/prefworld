from __future__ import annotations

import math
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
        # PC stability / expressiveness knobs
        pc_context_dependent_prior: bool = True,
        pc_prior_hidden: Optional[int] = None,
        maneuver_beta: float = 1.0,
        feasible_action_penalty: float = 5.0,
        feasible_action_soft_penalty_train: bool = True,
        feasible_action_hard_mask_eval: bool = True,
        template_hidden: int = 128,
        energy_hidden: int = 128,
        eb_temperature: float = 1.0,
        eb_max_candidates: int = 64,
        # (Req-5) Candidate expansion: allow 2-step edit combinations in EB-STM
        eb_two_step_topk_pairs: int = 0,
        eb_two_step_beam_size: int = 24,
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

        # Template encoder τ (HD-map polylines + neighbor graph + structured action slots)
        self.template = TemplateEncoder(tau_dim=self.tau_dim, agent_state_dim=5, hidden_dim=int(template_hidden))
        self.action_feature_dim = int(self.template.action_feature_dim)

        # Preference completion on structured action tokens
        self.pc = PreferenceCompletion(
            x_dim=3,
            tau_dim=self.tau_dim,
            ctx_dim=5,
            z_dim=self.z_dim,
            hidden_dim=int(pc_hidden),
            num_maneuvers=self.num_maneuvers,
            action_feature_dim=self.action_feature_dim,
            context_dependent_prior=bool(pc_context_dependent_prior),
            prior_hidden_dim=pc_prior_hidden,
            maneuver_beta=float(maneuver_beta),
            feasible_action_penalty=float(feasible_action_penalty),
            feasible_action_soft_penalty_train=bool(feasible_action_soft_penalty_train),
            feasible_action_hard_mask_eval=bool(feasible_action_hard_mask_eval),
        )

        # EFEN / EB-STM now conditions on a structured ego-action descriptor instead of a
        # pure legacy maneuver one-hot.
        self.energy_net = EditFactorizedEnergyNetEdit(
            agent_feat_dim=self.agent_feat_dim + self.tau_dim + self.action_feature_dim,
            z_dim=self.z_dim,
            hidden_dim=int(energy_hidden),
            lambda_u=float(lambda_u),
            return_token_params=bool(return_token_params),
        )
        self.ebstm = EBSTM(
            self.energy_net,
            temperature=float(eb_temperature),
            max_candidates=int(eb_max_candidates),
            two_step_topk_pairs=int(eb_two_step_topk_pairs),
            two_step_beam_size=int(eb_two_step_beam_size),
        )

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
            "pc_context_dependent_prior": bool(pc_context_dependent_prior),
            "pc_prior_hidden": None if pc_prior_hidden is None else int(pc_prior_hidden),
            "maneuver_beta": float(maneuver_beta),
            "feasible_action_penalty": float(feasible_action_penalty),
            "feasible_action_soft_penalty_train": bool(feasible_action_soft_penalty_train),
            "feasible_action_hard_mask_eval": bool(feasible_action_hard_mask_eval),
            "template_hidden": int(template_hidden),
            "energy_hidden": int(energy_hidden),
            "eb_temperature": float(eb_temperature),
            "eb_max_candidates": int(eb_max_candidates),
            "eb_two_step_topk_pairs": int(eb_two_step_topk_pairs),
            "eb_two_step_beam_size": int(eb_two_step_beam_size),
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

    @staticmethod
    def _select_action_feature_for_family(
        desired_family: torch.Tensor,          # [B]
        action_features: torch.Tensor,         # [B,A,D]
        action_family: torch.Tensor,           # [B,A]
        feasible_actions: torch.Tensor,        # [B,A]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick a representative structured action slot for each desired coarse family."""
        B, A, D = action_features.shape
        score = torch.arange(A, device=action_features.device, dtype=torch.float32).view(1, A).expand(B, A)
        fam = desired_family.long().unsqueeze(-1)
        feas = feasible_actions > 0.5
        match = feas & (action_family.long() == fam)
        any_match = match.any(dim=-1, keepdim=True)
        sel = torch.where(any_match, match, feas)
        sel = torch.where(sel.any(dim=-1, keepdim=True), sel, torch.ones_like(sel))
        idx = score.masked_fill(~sel, 1e9).argmin(dim=-1).long()
        batch_idx = torch.arange(B, device=action_features.device)
        feat = action_features[batch_idx, idx]
        return feat, idx

    def _build_agent_state_for_eb(
        self,
        ego_dyn: torch.Tensor,            # [B,4] vx,vy,ax,ay (ego-local)
        ego_maneuver: torch.Tensor,       # [B] int (legacy family, used to select a structured slot)
        agents_curr: torch.Tensor,        # [B,N,7]
        agents_valid: torch.Tensor,       # [B,N] float
        tau_curr: torch.Tensor,           # [B,1+N,Dt]
        ego_action_features_curr: Optional[torch.Tensor] = None,  # [B,A,D_a]
        ego_action_family_curr: Optional[torch.Tensor] = None,    # [B,A]
        ego_feasible_curr: Optional[torch.Tensor] = None,         # [B,A]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build [B,K,(Da+Dt)+D_a] features and mask for EB-STM (K=1+N)."""
        B, N, D = agents_curr.shape
        K = 1 + N
        feat = torch.zeros((B, K, D + self.tau_dim + self.action_feature_dim), device=agents_curr.device, dtype=agents_curr.dtype)
        mask = torch.zeros((B, K), device=agents_curr.device, dtype=agents_curr.dtype)
        mask[:, 0] = 1.0
        mask[:, 1:] = agents_valid

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

        if ego_action_features_curr is not None and ego_action_family_curr is not None and ego_feasible_curr is not None:
            ego_action_feat, _ = self._select_action_feature_for_family(
                ego_maneuver,
                ego_action_features_curr,
                ego_action_family_curr,
                ego_feasible_curr,
            )
        else:
            ego_action_feat = torch.zeros((B, self.action_feature_dim), device=agents_curr.device, dtype=agents_curr.dtype)
            n = min(self.num_maneuvers, self.action_feature_dim)
            ego_onehot = F.one_hot(ego_maneuver.long().clamp(min=0, max=self.num_maneuvers - 1), num_classes=self.num_maneuvers).to(ego_action_feat.dtype)
            ego_action_feat[:, :n] = ego_onehot[:, :n]

        feat[:, 0, D + self.tau_dim :] = ego_action_feat
        feat[:, 1:, :D] = agents_curr
        feat[:, 1:, D : D + self.tau_dim] = tau_curr[:, 1:]
        return feat, mask

    @staticmethod
    def _shuffle_non_ego_beliefs(
        z_mean: torch.Tensor,
        z_logvar: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shuffle non-ego preference beliefs within each batch item.

        This is used to regularize EB-STM so that it cannot ignore preference inputs.
        Ego (index 0) is never shuffled. Invalid agents (mask=0) are kept as-is.

        Returns:
          z_mean_shuf, z_logvar_shuf, shuffle_mask [B] (True if shuffled)
        """
        B, K, Dz = z_mean.shape
        z_m = z_mean.clone()
        z_v = z_logvar.clone()
        did = torch.zeros((B,), device=z_mean.device, dtype=torch.bool)
        for b in range(B):
            valid = (agent_mask[b, 1:] > 0.5).nonzero(as_tuple=False).view(-1)
            if valid.numel() < 2:
                continue
            idx = valid + 1  # shift to account for ego
            perm = idx[torch.randperm(idx.numel(), device=z_mean.device)]
            z_m[b, idx] = z_mean[b, perm]
            z_v[b, idx] = z_logvar[b, perm]
            did[b] = True
        return z_m, z_v, did

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
        # EB noisy-observation supervision (paper Sec.6 / App.)
        eb_use_noisy_obs: bool = True,
        eb_noisy_epsilon0: float = 0.2,
        eb_noisy_w_min: float = 0.2,
        eb_noisy_eps_min: float = 1e-4,
        # EB counterfactual conservatism regularizer (paper Eq.(26))
        eb_cf_weight: float = 0.0,
        eb_cf_base_temperature: float = 2.0,
        eb_cf_actions: Optional[Tuple[int, ...]] = None,
        # EB regularizers to prevent the energy model from ignoring preferences
        eb_pref_sens_weight: float = 0.0,
        eb_pref_sens_margin: float = 0.2,
        eb_base_l2_weight: float = 0.0,
        # Optional weak supervision for maneuver metrics
        use_pseudo_intent: bool = False,
        intent_weight: float = 1.0,
        # Guardrail: maneuver labels come from future trajectories and leak information.
        # Only enable if you *explicitly* want leaky supervision/metrics.
        allow_future_label_leakage: bool = False,
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
        action_features_all = template_out.action_features
        action_family_all = template_out.action_family

        # -----------------------------
        # Preference completion
        # -----------------------------
        x_action = self._action_tokens_from_hist(agents_hist)  # [B,N,T-1,3]
        mask_action = (agents_hist_mask[:, :, 1:] * agents_hist_mask[:, :, :-1]).to(dtype=agents_hist.dtype)  # [B,N,T-1]
        tau_action = tau_agents[:, :, :-1, :]                   # [B,N,T-1,Dt]
        ctx_action = state_all[:, 1:, :-1, :]                    # [B,N,T-1,5]
        feasible_action = None
        action_features_action = None
        action_family_action = None
        if feasible_all is not None:
            feasible_action = feasible_all[:, 1:, :-1, :]
        if action_features_all is not None:
            action_features_action = action_features_all[:, 1:, :-1, :, :]
        if action_family_all is not None:
            action_family_action = action_family_all[:, 1:, :-1, :]

        comparable_metrics_action = template_out.comparable_metrics[:, 1:, :-1, :, :]
        dynamic_metrics_action = template_out.dynamic_metrics[:, 1:, :-1, :, :]
        path_polyline_idx_action = template_out.path_polyline_idx[:, 1:, :-1, :]

        action_path_type_action = template_out.action_path_type[:, 1:, :-1, :]
        action_constraint_type_action = template_out.action_constraint_type[:, 1:, :-1, :]
        if run_pc:
            pc_out = self.pc(
                x=x_action,
                tau=tau_action,
                ctx=ctx_action,
                mask=mask_action,
                feasible_actions=feasible_action,
                action_features=action_features_action,
                action_family=action_family_action,
                comparable_metrics=comparable_metrics_action,
                dynamic_metrics=dynamic_metrics_action,
                path_polyline_idx=path_polyline_idx_action,
                action_path_type=action_path_type_action,
                action_constraint_type = action_constraint_type_action,
                map_polylines=batch["map_polylines"],
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
            action_logits_last = pc_out.action_logits_last
            action_family_last = pc_out.action_family_last
        else:
            pc_out = None
            post_part = None
            loss_pc = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)
            maneuver_logits_last = torch.zeros((B, N, self.num_maneuvers), device=agents_hist.device, dtype=agents_hist.dtype)
            action_logits_last = None
            action_family_last = None

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
            if not bool(allow_future_label_leakage):
                raise ValueError(
                    "use_pseudo_intent=True uses maneuver labels computed from future trajectories (information leakage). "
                    "If you intentionally want this ablation, pass allow_future_label_leakage=True."
                )
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

            # Noisy observation at t+1 (preferred, avoids future leakage).
            # If not available, we fall back to the oracle structure_t1 (leaky; offline-only).
            A_obs_t1 = batch.get("structure_t1_rule", None)
            w_t = batch.get("structure_conf_t", None)
            w_t1 = batch.get("structure_conf_t1", None)
            if w_t is not None and torch.is_tensor(w_t):
                w_t = w_t.view(-1).to(dtype=agents_hist.dtype)
            if w_t1 is not None and torch.is_tensor(w_t1):
                w_t1 = w_t1.view(-1).to(dtype=agents_hist.dtype)

            agents_curr = agents_hist[:, :, -1, :]  # [B,N,7]
            ego_dyn_curr = ego_dyn_hist[:, -1, :]   # [B,4]
            tau_curr = tau_all[:, :, -1, :]         # [B,1+N,Dt]
            ego_action_features_curr = action_features_all[:, 0, -1] if action_features_all is not None else None
            ego_action_family_curr = action_family_all[:, 0, -1] if action_family_all is not None else None
            ego_feasible_curr = feasible_all[:, 0, -1] if feasible_all is not None else None
            eb_feat, eb_mask = self._build_agent_state_for_eb(
                ego_dyn_curr,
                ego_m,
                agents_curr,
                agents_valid,
                tau_curr,
                ego_action_features_curr,
                ego_action_family_curr,
                ego_feasible_curr,
            )

            # Detach preference belief if we don't want EB gradients flowing into PC.
            z_mean_eb = z_mean.detach() if detach_pc_for_eb else z_mean
            z_logvar_eb = z_logvar.detach() if detach_pc_for_eb else z_logvar

            # Candidate energies and transition distribution.
            # For noisy-observation training we must NOT force-insert the oracle.
            out_eb = self.ebstm(
                A_t,
                eb_feat,
                z_mean_eb,
                z_logvar_eb,
                eb_mask,
                oracle_next=None if bool(eb_use_noisy_obs) else A_t1,
                smooth_scale=float(eb_smooth_scale),
                phys_dist_threshold_m=float(eb_phys_dist_threshold_m),
                phys_penalty_scale=float(eb_phys_penalty_scale),
            )

            # -------------------------------------------------
            # EB supervision
            #   - Preferred (paper-faithful): soft targets from noisy observation 
            #     \tilde s_{t+1} (no future leakage).
            #   - Fallback: hard oracle label structure_t1 (offline-only).
            # -------------------------------------------------
            loss_eb = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            if bool(eb_use_noisy_obs) and A_obs_t1 is not None:
                # confidence weights
                if w_t is None:
                    w_t = torch.ones((B,), device=agents_hist.device, dtype=agents_hist.dtype)
                if w_t1 is None:
                    w_t1 = torch.ones((B,), device=agents_hist.device, dtype=agents_hist.dtype)
                w_tt1 = torch.minimum(w_t, w_t1)

                # Drop low-confidence transitions (paper: skip if w < w_min).
                w_min = float(eb_noisy_w_min)
                keep = (w_tt1 >= w_min).to(dtype=agents_hist.dtype)

                # Observation noise rate eps_t = eps0 (1 - w_{t+1}).
                eps0 = float(eb_noisy_epsilon0)
                eps = eps0 * (1.0 - w_t1).clamp(min=0.0, max=1.0)
                eps = eps.clamp(min=float(eb_noisy_eps_min), max=1.0 - float(eb_noisy_eps_min))  # [B]

                # Build per-candidate soft targets q_{t+1}(\tau') \propto p_\eta(\tilde s | s(\tau')).
                cand_A = out_eb.candidate_A  # [B,C,K,K]
                cand_mask = out_eb.candidate_mask  # [B,C] bool

                # 3-state relation type per unordered pair (upper triangle only).
                cand_typ = self.ebstm._relation_state3(cand_A)  # [B,C,K,K]
                obs_typ = self.ebstm._relation_state3(A_obs_t1.to(device=cand_A.device)).unsqueeze(1)  # [B,1,K,K]

                # valid unordered pairs (i<j) among valid nodes
                pair_mask = out_eb.pair_energies.pair_mask  # [B,K,K] bool
                Kk = pair_mask.shape[-1]
                triu = torch.triu(torch.ones((Kk, Kk), device=pair_mask.device, dtype=torch.bool), diagonal=1)
                valid_pairs = (pair_mask & triu).unsqueeze(1)  # [B,1,K,K]

                match = (cand_typ == obs_typ) & valid_pairs
                num_match = match.to(torch.float32).sum(dim=(-1, -2))  # [B,C]
                num_pairs = valid_pairs.to(torch.float32).sum(dim=(-1, -2)).clamp_min(1.0)  # [B,1]
                num_mis = num_pairs - num_match

                log_p_match = torch.log1p(-eps).unsqueeze(1)  # [B,1]
                # 3-state categorical flip: mismatch prob is eps/2
                log_p_mis = (torch.log(eps) - math.log(2.0)).unsqueeze(1)  # [B,1]
                loglik = num_match * log_p_match + num_mis * log_p_mis

                # mask invalid candidates
                loglik = loglik.masked_fill(~cand_mask, -1e9)
                q = torch.softmax(loglik, dim=-1)  # [B,C]

                logp = torch.log(out_eb.probs.clamp_min(1e-12))
                per = -(q * logp).sum(dim=-1)  # [B]
                w_eff = (keep * w_tt1).to(torch.float32)
                loss_eb = (per.to(torch.float32) * w_eff).sum() / (w_eff.sum() + 1e-6)

            elif A_t1 is not None:
                # Hard oracle cross-entropy (leaky; kept for backwards compatibility).
                oracle_idx = self.ebstm.oracle_index(out_eb.candidate_A, A_t1)
                logp_oracle = torch.log(out_eb.probs[torch.arange(B, device=agents_hist.device), oracle_idx] + 1e-8)
                loss_eb = -logp_oracle.mean().to(torch.float32)

            # ------------------------------------------------------------------
            # Structure metrics (oracle only; computed from future labels)
            # ------------------------------------------------------------------
            if A_t1 is not None:
                with torch.no_grad():
                    pred_idx = out_eb.energies.argmin(dim=1)
                    A_pred = out_eb.candidate_A[torch.arange(B, device=agents_hist.device), pred_idx]

                    # Exact match
                    struct_acc = (A_pred == A_t1).all(dim=-1).all(dim=-1).float().mean()

                    # (Req-9) Directed edge precision/recall/F1
                    node_mask = eb_mask.to(torch.bool)  # [B,K]
                    eye = torch.eye(A_pred.shape[-1], device=A_pred.device, dtype=torch.bool).unsqueeze(0)
                    mask_dir = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2) & (~eye)

                    pred_edge = (A_pred > 0.5)
                    true_edge = (A_t1 > 0.5)
                    tp = (pred_edge & true_edge & mask_dir).sum().to(torch.float32)
                    fp = (pred_edge & (~true_edge) & mask_dir).sum().to(torch.float32)
                    fn = ((~pred_edge) & true_edge & mask_dir).sum().to(torch.float32)
                    struct_edge_prec = tp / (tp + fp + 1e-9)
                    struct_edge_rec = tp / (tp + fn + 1e-9)
                    struct_edge_f1 = 2.0 * struct_edge_prec * struct_edge_rec / (struct_edge_prec + struct_edge_rec + 1e-9)

                    # (Req-9) Pairwise 3-state accuracy over unordered pairs (i<j)
                    typ_pred = self.ebstm._relation_state3(A_pred)  # [B,K,K]
                    typ_true = self.ebstm._relation_state3(A_t1)
                    triu = torch.triu(torch.ones_like(typ_pred, dtype=torch.bool), diagonal=1)
                    mask_pair = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2) & triu
                    struct_pair_acc = ((typ_pred == typ_true) & mask_pair).sum().to(torch.float32) / mask_pair.sum().clamp_min(1).to(torch.float32)

                    # (Req-5 helper) Oracle edit distance between current A_t and oracle A_{t+Δ}
                    typ0 = self.ebstm._relation_state3(A_t)
                    oracle_edit_distance = ((typ0 != typ_true) & mask_pair).sum(dim=(-1, -2)).to(torch.float32).mean()
            else:
                struct_acc = torch.tensor(float("nan"), device=agents_hist.device, dtype=agents_hist.dtype)
                struct_edge_prec = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
                struct_edge_rec = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
                struct_edge_f1 = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
                struct_pair_acc = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
                oracle_edit_distance = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)

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
                        ego_action_features_curr,
                        ego_action_family_curr,
                        ego_feasible_curr,
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

            # -------------------------------------------------
            # Preference-sensitivity regularizer (prevents ignoring z)
            # -------------------------------------------------
            loss_eb_pref_sens = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            if float(eb_pref_sens_weight) > 0.0 and A_t1 is not None:
                # Shuffle non-ego beliefs; if EB-STM is using preferences, the oracle should become less likely.
                z_mean_shuf, z_logvar_shuf, did_shuf = self._shuffle_non_ego_beliefs(z_mean_eb, z_logvar_eb, eb_mask)
                if did_shuf.any():
                    out_shuf = self.ebstm(
                        A_t,
                        eb_feat,
                        z_mean_shuf,
                        z_logvar_shuf,
                        eb_mask,
                        oracle_next=A_t1,
                        smooth_scale=float(eb_smooth_scale),
                        phys_dist_threshold_m=float(eb_phys_dist_threshold_m),
                        phys_penalty_scale=float(eb_phys_penalty_scale),
                    )
                    lp_shuf = torch.log(out_shuf.probs[torch.arange(B, device=agents_hist.device), oracle_idx] + 1e-8)
                    # Maximize drop in oracle log-prob (margin ranking)
                    margin = float(eb_pref_sens_margin)
                    delta_lp = logp_oracle - lp_shuf
                    per = F.relu(margin - delta_lp)
                    w = did_shuf.to(per.dtype)
                    loss_eb_pref_sens = (per * w).sum() / (w.sum() + 1e-6)

            # -------------------------------------------------
            # Base-energy regularizer (encourage relying on preference terms)
            # -------------------------------------------------
            loss_eb_base_l2 = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            if float(eb_base_l2_weight) > 0.0:
                base = getattr(out_eb.pair_energies, 'base_edit', None)
                if base is not None:
                    Kk = base.shape[1]
                    triu = torch.triu(torch.ones((Kk, Kk), device=base.device, dtype=torch.bool), diagonal=1)
                    valid = (out_eb.pair_energies.pair_mask & triu).unsqueeze(-1)
                    denom = valid.to(torch.float32).sum().clamp_min(1e-6)
                    loss_eb_base_l2 = ((base.to(torch.float32) ** 2) * valid.to(torch.float32)).sum() / denom

            loss_eb_total = (
                loss_eb
                + float(eb_cf_weight) * loss_cf
                + float(eb_pref_sens_weight) * loss_eb_pref_sens
                + float(eb_base_l2_weight) * loss_eb_base_l2
            )
        else:
            loss_eb = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)
            struct_acc = torch.zeros((), device=agents_hist.device, dtype=agents_hist.dtype)
            struct_edge_prec = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
            struct_edge_rec = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
            struct_edge_f1 = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
            struct_pair_acc = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
            oracle_edit_distance = torch.tensor(float("nan"), device=agents_hist.device, dtype=torch.float32)
            out_eb = None
            loss_cf = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            loss_eb_pref_sens = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            loss_eb_base_l2 = torch.zeros((), device=agents_hist.device, dtype=torch.float32)
            loss_eb_total = loss_eb

        # -----------------------------
        # Collect
        # -----------------------------
        losses: Dict[str, torch.Tensor] = {
            "loss_pc": loss_pc,
            "loss_intent": loss_intent,
            "loss_eb": loss_eb,
            "loss_eb_cf": loss_cf,
            "loss_eb_pref_sens": loss_eb_pref_sens,
            "loss_eb_base_l2": loss_eb_base_l2,
            "loss_eb_total": loss_eb_total,
        }
        metrics: Dict[str, torch.Tensor] = {
            "intent_acc": acc,
            "struct_exact": struct_acc,
            # (Req-9)
            "struct_edge_precision": struct_edge_prec,
            "struct_edge_recall": struct_edge_rec,
            "struct_edge_f1": struct_edge_f1,
            "struct_pair_acc": struct_pair_acc,
            # (Req-5 helper)
            "oracle_edit_distance": oracle_edit_distance,
        }
        aux: Dict[str, torch.Tensor] = {
            "maneuver_logits_last": maneuver_logits_last.detach(),
            "z_mean": z_mean[:, 1:].detach() if z_mean.shape[1] > 1 else z_mean.detach(),
            "z_logvar": z_logvar[:, 1:].detach() if z_logvar.shape[1] > 1 else z_logvar.detach(),
        }
        if action_logits_last is not None:
            aux["action_logits_last"] = action_logits_last.detach()
        if action_family_last is not None:
            aux["action_family_last"] = action_family_last.detach()
        if feasible_all is not None:
            aux["ego_feasible_actions_last"] = feasible_all[:, 0, -1].detach()
        if action_features_all is not None:
            aux["ego_action_features_last"] = action_features_all[:, 0, -1].detach()
        if action_family_all is not None:
            aux["ego_action_family_last"] = action_family_all[:, 0, -1].detach()

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
