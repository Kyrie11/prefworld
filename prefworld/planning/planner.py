from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from prefworld.data.labels import Maneuver, NUM_MANEUVERS
from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.planning.critical_agents import select_topk_with_tiebreak
from prefworld.planning.pci import compute_pci_scores
from prefworld.planning.risk import collision_risk
from prefworld.planning.mpc import BicycleState, mpc_track, mpc_track_robust, rollout_bicycle


@dataclass
class PlannerConfig:
    horizon_steps: int = 80
    dt: float = 0.1
    # Structure time step (EB-STM) Δt_s; typically 0.5s for nuPlan cached sampling.
    structure_dt: float = 0.5
    # If set, overrides derived structure horizon steps (otherwise derived from horizon_steps*dt/structure_dt).
    structure_horizon_steps: Optional[int] = None
    # Optional predictability-adaptive horizon stopping (transition entropy threshold).
    structure_entropy_stop_threshold: Optional[float] = None
    structure_min_horizon_steps: int = 1

    # (Req-6) Predictability-adaptive horizon refinement:
    # Stop expanding the structure horizon only when BOTH
    #   (entropy > structure_entropy_stop_threshold) AND (risk_proxy < structure_risk_stop_threshold).
    # Set to None to disable the risk term (entropy-only truncation).
    structure_risk_stop_threshold: Optional[float] = None
    structure_risk_horizon_s: float = 3.0
    structure_risk_sigma_dist: float = 6.0

    num_structure_samples: int = 32
    use_beam: bool = False
    beam_size: int = 16

    # Feasible action set
    respect_feasible_actions: bool = True

    # Chance constraints (paper Eq.(33))
    enforce_chance_constraints: bool = True
    chance_collision_epsilon: float = 0.05
    chance_cvar_rho: float = 0.1
    infeasible_penalty: float = 1e3

    # PCI (critical-agent selection)
    pci_enabled: bool = True
    pci_topk: int = 6
    pci_tiebreak_eps: float = 1e-3
    pci_smooth_eps: float = 1e-6
    pci_coarse_actions: Tuple[int, ...] = (int(Maneuver.KEEP), int(Maneuver.LANE_CHANGE_LEFT), int(Maneuver.LANE_CHANGE_RIGHT))
    pci_horizon_steps: int = 12
    pci_num_structure_samples: int = 32
    pci_mix_baseline_support: bool = True
    pci_num_rollouts_baseline: int = 16
    pci_use_beam: bool = False
    pci_beam_size: int = 16
    # Whether to weight shared rollout support by the model rollout probabilities.
    # Setting this to False uses a uniform distribution over the sampled/beam rollouts,
    # which is often more robust when logp estimates are noisy.
    pci_use_model_probs: bool = False
    pci_fallback_k: int = 2

    # Risk aggregation
    cvar_alpha: float = 0.9
    w_exp_risk: float = 1.0
    w_cvar_risk: float = 1.0

    # (Req-7) nuPlan-competition-style secondary objectives (lightweight proxies).
    # Defaults are 0.0 (disabled) so existing behavior is unchanged.
    w_progress: float = 0.0
    w_drivable: float = 0.0
    w_direction: float = 0.0
    w_speed_limit: float = 0.0
    w_comfort: float = 0.0
    drivable_threshold_m: float = 2.0
    direction_cos_threshold: float = 0.0
    speed_limit_mps: float = 13.9  # ~50 km/h

    # Agent rollout model (planner-side approximation)
    # If enabled, we roll out simple maneuver-conditioned primitives for other agents
    # (keep/lane-change/turn/stop) instead of pure constant-velocity motion.
    use_agent_primitives: bool = True
    lane_width_m: float = 3.5
    turn_yaw_rad: float = 0.6
    agent_stop_time_s: float = 3.0
    yield_slowdown: float = 0.4

    # MPC
    mpc_horizon_steps: int = 20
    mpc_iters: int = 40
    mpc_lr: float = 0.3

    # Robust MPC
    use_robust_mpc: bool = True
    robust_mpc_w_chance: float = 50.0
    robust_mpc_w_cvar: float = 50.0
    robust_mpc_tighten_margin: float = 0.3
    robust_mpc_softmin_beta: float = 20.0
    robust_mpc_sigmoid_temp: float = 0.25


@dataclass
class PlannerOutput:
    best_maneuver: torch.Tensor     # [B]
    best_score: torch.Tensor        # [B]
    per_maneuver_score: torch.Tensor  # [B,M]
    per_maneuver_risk: torch.Tensor   # [B,M]
    per_maneuver_cvar: torch.Tensor   # [B,M]
    controls: Optional[torch.Tensor]  # [B,H,2]
    ego_traj_xy: Optional[torch.Tensor] = None   # [B,H(+1),2]
    ego_traj_yaw: Optional[torch.Tensor] = None  # [B,H(+1)]


def _ego_reference_primitive(
    ego_speed: torch.Tensor,  # [B]
    maneuver: int,
    horizon_steps: int,
    dt: float,
    lane_width: float = 3.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a simple ego reference trajectory in ego frame."""
    B = ego_speed.shape[0]
    H = int(horizon_steps)
    t = torch.arange(H, device=ego_speed.device, dtype=ego_speed.dtype) * float(dt)

    # longitudinal distance
    if maneuver == int(Maneuver.STOP):
        # linear decel to 0 over 3s
        T_stop = 3.0
        a = -ego_speed / T_stop
        v_t = torch.clamp(ego_speed.unsqueeze(-1) + a.unsqueeze(-1) * t, min=0.0)
        s = torch.cumsum(v_t * dt, dim=-1)
    else:
        s = ego_speed.unsqueeze(-1) * t  # constant speed

    y = torch.zeros((B, H), device=ego_speed.device, dtype=ego_speed.dtype)
    yaw = torch.zeros((B, H), device=ego_speed.device, dtype=ego_speed.dtype)

    if maneuver == int(Maneuver.LANE_CHANGE_LEFT):
        y = lane_width * torch.sigmoid((t - t.mean()) / (0.6)) - 0.5 * lane_width
    elif maneuver == int(Maneuver.LANE_CHANGE_RIGHT):
        y = -lane_width * torch.sigmoid((t - t.mean()) / (0.6)) + 0.5 * lane_width
    elif maneuver == int(Maneuver.TURN_LEFT):
        yaw = 0.6 * torch.sigmoid((t - t.mean()) / 0.8)
    elif maneuver == int(Maneuver.TURN_RIGHT):
        yaw = -0.6 * torch.sigmoid((t - t.mean()) / 0.8)

    x = s
    ref_xy = torch.stack([x, y], dim=-1)  # [B,H,2]
    return ref_xy, yaw


@torch.no_grad()
def plan_with_structures(
    model: PrefWorldModel,
    batch: Dict[str, torch.Tensor],
    cfg: PlannerConfig,
    candidate_maneuvers: Optional[Tuple[int, ...]] = None,
) -> PlannerOutput:
    """A lightweight planner consistent with PrefWorld's design:

      1) Sample / beam-rollout interaction structures with EB-STM conditioned on ego maneuver.
      2) Score each ego maneuver by a risk metric derived from structure rollouts.
      3) Run an MPC tracker on the selected maneuver reference (optional).

    NOTE: This is a simplified planner that uses structure interaction frequency as a proxy for risk.
    It is intended to make the repo runnable and to match the paper's logic flow, even without
    a full closed-loop simulator integration.
    """
    if candidate_maneuvers is None:
        candidate_maneuvers = tuple(range(NUM_MANEUVERS))

    device = next(model.parameters()).device
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Time-scale separation (paper-aligned):
    #   - motion dt (MPC / trajectories)
    #   - structure dt (EB-STM transitions / PCI)
    # ------------------------------------------------------------------
    motion_dt = float(cfg.dt)
    H_motion = int(cfg.horizon_steps)

    structure_dt = float(getattr(cfg, 'structure_dt', motion_dt) or motion_dt)
    if structure_dt <= 0.0:
        structure_dt = motion_dt

    ratio = structure_dt / max(1e-6, motion_dt)
    K_m = max(1, int(round(ratio)))
    if abs(ratio - K_m) > 1e-3:
        # Snap to an integer multiple for deterministic indexing.
        K_m = max(1, int(math.floor(ratio + 1e-6)))
        structure_dt = motion_dt * K_m

    H_structure = getattr(cfg, 'structure_horizon_steps', None)
    if H_structure is None:
        # cover the motion horizon in structure steps
        H_structure = int(math.ceil((H_motion * motion_dt) / max(1e-6, structure_dt)))
    H_structure = max(1, int(H_structure))

    # Run the model once to get posterior preferences.
    # NOTE: PrefWorldModel.forward does not take pc_drop_prob; we use pc_query_ratio=0 to
    # condition on the full observed history when planning.
    out = model(batch, run_pc=True, run_eb=False, pc_query_ratio=0.0)
    z_mean = out.aux["z_mean"]  # [B,N,Dz]
    z_logvar = out.aux["z_logvar"]  # [B,N,Dz]
    maneuver_logits_last = out.aux.get("maneuver_logits_last", None)  # [B,N,M]

    # Rebuild EB features (needs τ and current agent state)
    agents_hist = batch["agents_hist"]
    agents_hist_mask = batch["agents_hist_mask"]
    ego_dyn_hist = batch["ego_dyn_hist"]
    ego_hist = batch["ego_hist"]
    map_polylines = batch["map_polylines"]
    map_poly_mask = batch["map_poly_mask"]
    map_poly_type = batch.get("map_poly_type", None)
    map_tl_status = batch.get("map_tl_status", None)
    map_on_route = batch.get("map_on_route", None)

    B, N, T, D = agents_hist.shape
    agents_valid = (agents_hist_mask[:, :, -1] > 0.5).float()

    ego_dyn_curr = ego_dyn_hist[:, -1, :]
    ego_speed = torch.norm(ego_dyn_curr[:, 0:2], dim=-1)

    # τ encoding via the model helper (also provides feasible actions)
    template_out, state_all, _ = model.encode_templates(batch)
    tau_all = template_out.tau
    feasible_all = template_out.feasible_actions
    tau_curr = tau_all[:, :, -1, :]

    agents_curr = agents_hist[:, :, -1, :]

    # Predicted maneuvers for other agents (used for a more realistic motion rollout).
    # If logits are unavailable, fall back to KEEP.
    if maneuver_logits_last is not None and maneuver_logits_last.numel() > 0:
        agents_maneuver = maneuver_logits_last.argmax(dim=-1).to(torch.int64)  # [B,N]
    else:
        agents_maneuver = torch.zeros((B, N), device=device, dtype=torch.int64)

    # Optionally respect per-agent feasible action masks.
    if feasible_all is not None:
        feas_agents = feasible_all[:, 1:, -1, :].to(torch.bool)  # [B,N,M]
        keep_id = int(Maneuver.KEEP)
        chosen_ok = feas_agents.gather(dim=-1, index=agents_maneuver.unsqueeze(-1)).squeeze(-1)
        agents_maneuver = torch.where(chosen_ok, agents_maneuver, torch.full_like(agents_maneuver, keep_id))

    # Build z tensors for EB (include ego at index 0 as standard normal)
    z_mean_eb = torch.zeros((B, 1 + N, model.z_dim), device=device, dtype=agents_hist.dtype)
    z_logvar_eb = torch.zeros_like(z_mean_eb)
    z_mean_eb[:, 1:] = z_mean
    z_logvar_eb[:, 1:] = z_logvar

    # Base (non-individualized) belief for mixed summaries (paper Eq.(31))
    base_mean = model.pc.prior_mean.to(device=device, dtype=agents_hist.dtype).view(1, 1, model.z_dim).expand(B, 1 + N, model.z_dim)
    base_logvar = model.pc.prior_logvar.to(device=device, dtype=agents_hist.dtype).view(1, 1, model.z_dim).expand(B, 1 + N, model.z_dim)

    # IMPORTANT: avoid future leakage. Use rule-based structure if available.
    A_t = batch.get("structure_t_rule", batch.get("structure_t"))
    if A_t is None:
        raise KeyError("Batch missing structure_t_rule/structure_t")

    # --------------------------------------------------------------
    # PCI: select a small set of critical agents and mix summaries.
    # --------------------------------------------------------------
    eb_mask_full = torch.cat([torch.ones((B, 1), device=device, dtype=agents_valid.dtype), agents_valid], dim=1)
    sel_mask = eb_mask_full.clone()  # default: keep individualized for all valid agents

    if bool(cfg.pci_enabled):
        K = 1 + N
        pci_max = torch.zeros((B, K), device=device, dtype=torch.float32)
        risk_at_max = torch.zeros_like(pci_max)

        for a in cfg.pci_coarse_actions:
            ego_a = torch.full((B,), int(a), device=device, dtype=torch.int64)
            eb_feat_a, eb_mask_a = model._build_agent_state_for_eb(ego_dyn_curr, ego_a, agents_curr, agents_valid, tau_curr)

            pci_res = compute_pci_scores(
                ebstm=model.ebstm,
                A_t=A_t,
                agent_feat=eb_feat_a,
                z_mean=z_mean_eb,
                z_logvar=z_logvar_eb,
                agent_mask=eb_mask_a,
                baseline_mean=0.0,
                baseline_logvar=float(model.pc.prior_logvar.mean().item()),
                include_ego=False,
                horizon_steps=int(cfg.pci_horizon_steps),
                num_rollouts=int(cfg.pci_num_structure_samples),
                use_beam_rollout=bool(cfg.pci_use_beam),
                beam_size=int(cfg.pci_beam_size),
                use_model_probs=bool(getattr(cfg, "pci_use_model_probs", False)),
                cvar_alpha=float(cfg.cvar_alpha),
                smooth_eps=float(cfg.pci_smooth_eps),
                dt=float(structure_dt),
                mix_baseline_support=bool(getattr(cfg, "pci_mix_baseline_support", True)),
                num_rollouts_baseline=int(getattr(cfg, "pci_num_rollouts_baseline", 16)),
            )

            better = pci_res.pci > pci_max
            pci_max = torch.where(better, pci_res.pci, pci_max)
            risk_at_max = torch.where(better, pci_res.tiebreak_risk, risk_at_max)

        # select top-k agents (exclude ego)
        sel = select_topk_with_tiebreak(
            pci_scores=pci_max,
            risk_scores=risk_at_max,
            agent_mask=eb_mask_full,
            k=int(cfg.pci_topk),
            include_ego=False,
            tiebreak_eps=float(cfg.pci_tiebreak_eps),
        )

        sel_mask = sel.mask

        # deterministic fallback: always keep the nearest few agents
        if int(cfg.pci_fallback_k) > 0:
            dist = torch.norm(agents_curr[:, :, 0:2], dim=-1)  # [B,N]
            dist = dist.masked_fill(agents_valid <= 0.5, float("inf"))
            k_fb = min(int(cfg.pci_fallback_k), N)
            if k_fb > 0:
                _, idx = torch.topk(-dist, k=k_fb, dim=-1)  # smallest dist
                idx_eb = idx + 1
                fb_mask = torch.zeros((B, 1 + N), device=device, dtype=sel_mask.dtype)
                fb_mask.scatter_(1, idx_eb, 1.0)
                sel_mask = torch.maximum(sel_mask, fb_mask)

        # Always exclude ego from "critical" set (ego is always individualized by construction)
        sel_mask[:, 0] = 1.0

    # Mixed summaries: replace non-critical agents with baseline belief.
    z_mean_mixed = z_mean_eb.clone()
    z_logvar_mixed = z_logvar_eb.clone()
    noncrit = (sel_mask <= 0.5) & (eb_mask_full > 0.5)
    noncrit[:, 0] = False
    z_mean_mixed[noncrit] = base_mean[noncrit]
    z_logvar_mixed[noncrit] = base_logvar[noncrit]
    per_score = torch.full((B, NUM_MANEUVERS), float("inf"), device=device, dtype=torch.float32)
    per_risk = torch.zeros((B, NUM_MANEUVERS), device=device, dtype=torch.float32)
    per_cvar = torch.zeros((B, NUM_MANEUVERS), device=device, dtype=torch.float32)

    # Ego feasible maneuvers (if available)
    ego_feasible = None
    if feasible_all is not None:
        ego_feasible = feasible_all[:, 0, -1, :].to(torch.bool)  # [B,M]

    stored_worlds: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for m in candidate_maneuvers:
        ego_m = torch.full((B,), int(m), device=device, dtype=torch.int64)
        eb_feat, eb_mask = model._build_agent_state_for_eb(ego_dyn_curr, ego_m, agents_curr, agents_valid, tau_curr)

        if cfg.use_beam:
            roll, logp = model.ebstm.beam_rollout(
                A_t,
                eb_feat,
                z_mean_mixed,
                z_logvar_mixed,
                eb_mask,
                max_horizon_steps=int(H_structure),
                beam_size=int(cfg.beam_size),
                entropy_stop_threshold=(None if getattr(cfg, "structure_entropy_stop_threshold", None) is None else float(getattr(cfg, "structure_entropy_stop_threshold"))),
                min_horizon_steps=int(getattr(cfg, "structure_min_horizon_steps", 1)),
                risk_stop_threshold=(None if getattr(cfg, "structure_risk_stop_threshold", None) is None else float(getattr(cfg, "structure_risk_stop_threshold"))),
                risk_horizon_s=float(getattr(cfg, "structure_risk_horizon_s", 3.0)),
                risk_sigma_dist=float(getattr(cfg, "structure_risk_sigma_dist", 6.0)),
            )
            # [B,S,H,K,K]
            S = roll.shape[1]
            p = torch.softmax(logp, dim=-1)  # [B,S]
        else:
            roll, logp, _ = model.ebstm.rollout(
                A_t,
                eb_feat,
                z_mean_mixed,
                z_logvar_mixed,
                eb_mask,
                horizon_steps=int(H_structure),
                num_samples=int(cfg.num_structure_samples),
                return_indices=False,
                entropy_stop_threshold=(None if getattr(cfg, "structure_entropy_stop_threshold", None) is None else float(getattr(cfg, "structure_entropy_stop_threshold"))),
                min_horizon_steps=int(getattr(cfg, "structure_min_horizon_steps", 1)),
                risk_stop_threshold=(None if getattr(cfg, "structure_risk_stop_threshold", None) is None else float(getattr(cfg, "structure_risk_stop_threshold"))),
                risk_horizon_s=float(getattr(cfg, "structure_risk_horizon_s", 3.0)),
                risk_sigma_dist=float(getattr(cfg, "structure_risk_sigma_dist", 6.0)),
            )
            S = roll.shape[1]
            p = torch.softmax(logp, dim=-1)

        # ------------------------------------------------------------------
        # Risk: collision probability (exp + CVaR) via constant-velocity rollouts
        # conditioned on sampled interaction structures.
        # ------------------------------------------------------------------
        ego_ref_xy, ego_ref_yaw = _ego_reference_primitive(
            ego_speed,
            int(m),
            horizon_steps=int(cfg.horizon_steps),
            dt=float(cfg.dt),
        )  # [B,H,2]

        # ------------------------------------------------------------------
        # Other agents initial state
        #
        # Improvement (paper-alignment): roll out simple maneuver-conditioned
        # motion primitives (lane change / turn / stop) instead of pure
        # constant-velocity propagation. This stays lightweight while producing
        # more realistic risk estimates.
        # ------------------------------------------------------------------
        pos0 = agents_curr[:, :, 0:2]          # [B,N,2]
        yaw0 = agents_curr[:, :, 2]            # [B,N]
        vel0 = agents_curr[:, :, 3:5]          # [B,N,2]
        speed0 = torch.norm(vel0, dim=-1).clamp_min(0.0)  # [B,N]

        # Use velocity direction as a robust heading estimate when speed is non-trivial.
        yaw_vel = torch.atan2(vel0[..., 1], vel0[..., 0])
        yaw0 = torch.where(speed0 > 0.5, yaw_vel, yaw0)

        # mask out invalid agents to avoid spurious collisions at the origin
        invalid = (agents_valid < 0.5).unsqueeze(-1)
        pos0 = torch.where(invalid, torch.full_like(pos0, 1e6), pos0)
        yaw0 = torch.where(agents_valid < 0.5, torch.zeros_like(yaw0), yaw0)
        speed0 = torch.where(agents_valid < 0.5, torch.zeros_like(speed0), speed0)
        # radius proxy from size if available
        if agents_curr.shape[-1] >= 7:
            length = agents_curr[:, :, 5].clamp_min(0.1)
            width = agents_curr[:, :, 6].clamp_min(0.1)
            radius = 0.5 * torch.maximum(length, width)
        else:
            radius = torch.full((B, N), 1.5, device=device, dtype=ego_ref_xy.dtype)
        radius = radius * agents_valid

        Hh = int(cfg.horizon_steps)
        S = roll.shape[1]
        Hs = roll.shape[2]  # structure horizon
        dtype = ego_ref_xy.dtype

        # Primitive shaping functions over time
        t_idx = torch.arange(Hh, device=device, dtype=dtype) + 1.0  # 1..H
        s_norm = (t_idx / float(max(1, Hh))).clamp(0.0, 1.0)
        smooth = s_norm * s_norm * (3.0 - 2.0 * s_norm)  # smoothstep

        lane_w = float(getattr(cfg, "lane_width_m", 3.5))
        turn_yaw = float(getattr(cfg, "turn_yaw_rad", 0.6))
        stop_time = float(getattr(cfg, "agent_stop_time_s", 3.0))

        # Per-agent primitive profiles [B,N,H]
        y_off_base = lane_w * smooth  # [H]
        yaw_off_base = turn_yaw * smooth  # [H]

        m_id = agents_maneuver  # [B,N]
        m_lcl = int(Maneuver.LANE_CHANGE_LEFT)
        m_lcr = int(Maneuver.LANE_CHANGE_RIGHT)
        m_tl = int(Maneuver.TURN_LEFT)
        m_tr = int(Maneuver.TURN_RIGHT)
        m_stop = int(Maneuver.STOP)

        is_lcl = (m_id == m_lcl).unsqueeze(-1)
        is_lcr = (m_id == m_lcr).unsqueeze(-1)
        is_tl = (m_id == m_tl).unsqueeze(-1)
        is_tr = (m_id == m_tr).unsqueeze(-1)
        is_stop = (m_id == m_stop).unsqueeze(-1)

        y_off = (is_lcl.to(dtype) * y_off_base) + (is_lcr.to(dtype) * (-y_off_base))  # [B,N,H]
        yaw_off = (is_tl.to(dtype) * yaw_off_base) + (is_tr.to(dtype) * (-yaw_off_base))
        yaw_t = yaw0.unsqueeze(-1) + yaw_off

        # speed profile
        t_sec = t_idx * float(cfg.dt)
        v_const = speed0.unsqueeze(-1).expand(B, N, Hh)
        v_stop = speed0.unsqueeze(-1) * (1.0 - (t_sec / max(stop_time, 1e-3))).clamp(min=0.0)
        v_t = torch.where(is_stop, v_stop, v_const)  # [B,N,H]

        # lane-change incremental lateral displacement
        dy = torch.zeros_like(y_off)
        dy[:, :, 0] = y_off[:, :, 0]
        if Hh > 1:
            dy[:, :, 1:] = y_off[:, :, 1:] - y_off[:, :, :-1]

        # direction vectors
        fwd = torch.stack([torch.cos(yaw_t), torch.sin(yaw_t)], dim=-1)  # [B,N,H,2]
        left = torch.stack([-torch.sin(yaw0), torch.cos(yaw0)], dim=-1)  # [B,N,2]

        # [B,S,N,2]
        pos = pos0.unsqueeze(1).expand(B, S, N, 2).clone()
        scale = torch.ones((B, S, N), device=device, dtype=dtype)
        slowdown = float(getattr(cfg, "yield_slowdown", 0.4))

        traj = torch.zeros((B, S, N, Hh, 2), device=device, dtype=dtype)
        for h in range(Hh):
            # If agent yields to ego at this step, it slows down (keeps the minimum scale).
            s_idx = min(int(h // K_m), int(Hs) - 1)
            yield_mask = (roll[:, :, s_idx, 1:, 0] > 0.5)  # [B,S,N] (structure step)
            scale = torch.minimum(scale, torch.where(yield_mask, torch.full_like(scale, slowdown), torch.ones_like(scale)))

            v_eff = v_t[:, :, h].unsqueeze(1) * scale  # [B,S,N]
            disp_fwd = fwd[:, :, h, :].unsqueeze(1) * v_eff.unsqueeze(-1) * float(cfg.dt)  # [B,S,N,2]
            disp_lat = left.unsqueeze(1) * dy[:, :, h].unsqueeze(1).unsqueeze(-1)  # [B,S,N,2]

            pos = pos + disp_fwd + disp_lat
            traj[:, :, :, h, :] = pos

        # Optional fallback to a simpler constant-velocity rollout (debug / ablations)
        if not bool(getattr(cfg, "use_agent_primitives", True)):
            pos = pos0.unsqueeze(1).expand(B, S, N, 2).clone()
            vel = vel0.unsqueeze(1).expand(B, S, N, 2).clone()
            scale = torch.ones((B, S, N), device=device, dtype=dtype)
            traj = torch.zeros((B, S, N, Hh, 2), device=device, dtype=dtype)
            for h in range(Hh):
                s_idx = min(int(h // K_m), int(Hs) - 1)
                yield_mask = (roll[:, :, s_idx, 1:, 0] > 0.5)
                scale = torch.minimum(scale, torch.where(yield_mask, torch.full_like(scale, slowdown), torch.ones_like(scale)))
                pos = pos + vel * scale.unsqueeze(-1) * float(cfg.dt)
                traj[:, :, :, h, :] = pos

        exp_risk, cvar_r = collision_risk(
            ego_traj=ego_ref_xy,
            agent_traj_samples=traj,
            agent_radius=radius,
            weights=p,
            cvar_alpha=float(cfg.cvar_alpha),
        )

        score = float(cfg.w_exp_risk) * exp_risk + float(cfg.w_cvar_risk) * cvar_r

        # ------------------------------------------------------------------
        # (Req-7) nuPlan-competition-style secondary objectives (lightweight proxies)
        # ------------------------------------------------------------------
        if any(float(getattr(cfg, k, 0.0)) > 0.0 for k in ("w_progress", "w_drivable", "w_direction", "w_speed_limit", "w_comfort")):
            # Progress proxy: maximize forward displacement in ego frame.
            if float(getattr(cfg, "w_progress", 0.0)) > 0.0:
                progress = ego_ref_xy[:, -1, 0]  # [B]
                score = score + float(cfg.w_progress) * (-progress)

            # Speed limit compliance proxy
            if float(getattr(cfg, "w_speed_limit", 0.0)) > 0.0:
                vel = (ego_ref_xy[:, 1:, :] - ego_ref_xy[:, :-1, :]) / float(cfg.dt)
                speed = torch.norm(vel, dim=-1)
                exceed = F.relu(speed - float(getattr(cfg, "speed_limit_mps", 13.9)))
                score = score + float(cfg.w_speed_limit) * exceed.mean(dim=-1)

            # Comfort proxy: penalize large accel + jerk magnitudes
            if float(getattr(cfg, "w_comfort", 0.0)) > 0.0:
                vel = (ego_ref_xy[:, 1:, :] - ego_ref_xy[:, :-1, :]) / float(cfg.dt)
                speed = torch.norm(vel, dim=-1)
                if speed.shape[1] >= 3:
                    accel = (speed[:, 1:] - speed[:, :-1]) / float(cfg.dt)
                    jerk = (accel[:, 1:] - accel[:, :-1]) / float(cfg.dt)
                    comfort = accel.abs().mean(dim=-1) + 0.5 * jerk.abs().mean(dim=-1)
                elif speed.shape[1] >= 2:
                    accel = (speed[:, 1:] - speed[:, :-1]) / float(cfg.dt)
                    comfort = accel.abs().mean(dim=-1)
                else:
                    comfort = torch.zeros((B,), device=device, dtype=torch.float32)
                score = score + float(cfg.w_comfort) * comfort

            # Drivable-area / driving-direction proxies using map polylines in ego frame.
            if float(getattr(cfg, "w_drivable", 0.0)) > 0.0 or float(getattr(cfg, "w_direction", 0.0)) > 0.0:
                # Sample at a coarse stride for efficiency.
                stride = 4
                ego_pts = ego_ref_xy[:, ::stride, :]  # [B,He,2]
                ego_yaw_s = ego_ref_yaw[:, ::stride]

                Bm, M, L, _ = map_polylines.shape
                if Bm != B:
                    raise ValueError("Batch size mismatch between ego_ref_xy and map_polylines")

                # Flatten map points.
                map_pts = map_polylines.reshape(B, M * L, 2)
                map_valid = map_poly_mask.unsqueeze(-1).expand(B, M, L).reshape(B, M * L) > 0.5

                if map_valid.any():
                    # Distance from ego samples to map points.
                    dist_mat = torch.cdist(ego_pts, map_pts)  # [B,He,P]
                    dist_mat = dist_mat.masked_fill(~map_valid.unsqueeze(1), float("inf"))
                    min_dist, min_idx = dist_mat.min(dim=-1)  # [B,He]

                    if float(getattr(cfg, "w_drivable", 0.0)) > 0.0:
                        thr = float(getattr(cfg, "drivable_threshold_m", 2.0))
                        drivable_violation = F.relu(min_dist - thr)
                        score = score + float(cfg.w_drivable) * drivable_violation.mean(dim=-1)

                    if float(getattr(cfg, "w_direction", 0.0)) > 0.0:
                        # Compute per-point map direction (approx) by finite differences along polyline.
                        d = map_polylines[:, :, 1:, :] - map_polylines[:, :, :-1, :]
                        dir_pts = torch.zeros_like(map_polylines)
                        dir_pts[:, :, :-1, :] = d
                        if L >= 2:
                            dir_pts[:, :, -1, :] = d[:, :, -1, :]
                        dir_flat = dir_pts.reshape(B, M * L, 2)
                        dir_flat = F.normalize(dir_flat, dim=-1, eps=1e-6)

                        # Gather nearest map direction for each ego sample.
                        nn_dir = dir_flat.gather(1, min_idx.unsqueeze(-1).expand(-1, -1, 2))  # [B,He,2]
                        ego_dir = torch.stack([torch.cos(ego_yaw_s), torch.sin(ego_yaw_s)], dim=-1)
                        dot = (nn_dir * ego_dir).sum(dim=-1)  # [B,He]
                        thr_cos = float(getattr(cfg, "direction_cos_threshold", 0.0))
                        dir_violation = F.relu(thr_cos - dot)
                        score = score + float(cfg.w_direction) * dir_violation.mean(dim=-1)

        # Enforce feasible maneuver set (if provided by the template encoder)
        if bool(cfg.respect_feasible_actions) and ego_feasible is not None:
            infeas = ~ego_feasible[:, int(m)]
            score = torch.where(infeas, torch.full_like(score, float("inf")), score)

        # Chance constraints (paper Eq.(33)): treat violations as infeasible (soft penalty)
        if bool(cfg.enforce_chance_constraints):
            v1 = torch.relu(exp_risk - float(cfg.chance_collision_epsilon))
            v2 = torch.relu(cvar_r - float(cfg.chance_cvar_rho))
            violation = v1 + v2
            score = score + float(cfg.infeasible_penalty) * violation
        per_score[:, m] = score
        per_risk[:, m] = exp_risk
        per_cvar[:, m] = cvar_r

        # Keep worlds for robust MPC (store agent trajectories + weights + radius)
        stored_worlds[int(m)] = (ego_ref_xy, traj, p, radius)

    best_m = per_score.argmin(dim=-1)  # [B]

    # Safety: if all candidate maneuvers were infeasible (score=inf), argmin may return
    # an uncomputed maneuver. Fall back to the first provided candidate.
    if len(candidate_maneuvers) > 0:
        fallback = int(candidate_maneuvers[0])
        for b in range(B):
            if int(best_m[b].item()) not in stored_worlds:
                best_m[b] = fallback

    best_score = per_score.gather(dim=-1, index=best_m.view(B, 1)).squeeze(-1)

    # MPC tracking for the chosen maneuver

    # ------------------------------------------------------------------
    # Build per-batch *full-horizon* reference trajectories.
    #
    # nuPlan closed-loop expects ~8s@10Hz => 80 future steps + current.
    # We therefore output H_out = horizon_steps + 1 points (t=0 included).
    # ------------------------------------------------------------------
    H_out = int(cfg.horizon_steps) + 1
    ref_xy_full_list = []
    ref_yaw_full_list = []
    for b in range(B):
        ref_xy_b, ref_yaw_b = _ego_reference_primitive(
            ego_speed[b : b + 1],
            int(best_m[b].item()),
            int(H_out),
            float(cfg.dt),
        )
        ref_xy_full_list.append(ref_xy_b)
        ref_yaw_full_list.append(ref_yaw_b)
    ref_xy_full = torch.cat(ref_xy_full_list, dim=0)    # [B,H_out,2]
    ref_yaw_full = torch.cat(ref_yaw_full_list, dim=0)  # [B,H_out]

    # MPC reference: align with rollout_bicycle outputs which start at t=dt.
    H_mpc = int(cfg.mpc_horizon_steps)
    ref_xy = ref_xy_full[:, 1 : 1 + H_mpc]
    ref_yaw = ref_yaw_full[:, 1 : 1 + H_mpc]

    state0 = BicycleState(
        x=torch.zeros((B,), device=device, dtype=ref_xy.dtype),
        y=torch.zeros((B,), device=device, dtype=ref_xy.dtype),
        yaw=torch.zeros((B,), device=device, dtype=ref_xy.dtype),
        v=ego_speed,
    )
    # MPC needs gradients even though planning uses no_grad
    with torch.enable_grad():
        if bool(cfg.use_robust_mpc):
            # Fetch stored world rollouts for each batch item and build tensors
            # We use the sampled agent trajectories for the selected maneuver as scenarios.
            ego_ref_xy_best, traj_best, p_best, radius_best = stored_worlds[int(best_m[0].item())]
            # If best maneuvers differ across batch, gather per-batch
            if not torch.all(best_m == best_m[0]):
                ego_ref_xy_best = torch.stack([stored_worlds[int(best_m[b].item())][0][b] for b in range(B)], dim=0)
                traj_best = torch.stack([stored_worlds[int(best_m[b].item())][1][b] for b in range(B)], dim=0)
                p_best = torch.stack([stored_worlds[int(best_m[b].item())][2][b] for b in range(B)], dim=0)
                radius_best = torch.stack([stored_worlds[int(best_m[b].item())][3][b] for b in range(B)], dim=0)

            traj_best = traj_best[:, :, :, : int(cfg.mpc_horizon_steps)]

            controls = mpc_track_robust(
                state0,
                ref_xy=ref_xy,
                ref_yaw=ref_yaw,
                agent_traj_samples=traj_best,
                weights=p_best,
                agent_radius=radius_best,
                dt=float(cfg.dt),
                horizon_steps=int(cfg.mpc_horizon_steps),
                iters=int(cfg.mpc_iters),
                lr=float(cfg.mpc_lr),
                chance_collision_epsilon=float(cfg.chance_collision_epsilon),
                cvar_alpha=float(cfg.cvar_alpha),
                chance_cvar_rho=float(cfg.chance_cvar_rho),
                w_chance=float(cfg.robust_mpc_w_chance),
                w_cvar=float(cfg.robust_mpc_w_cvar),
                tighten_margin=float(cfg.robust_mpc_tighten_margin),
                softmin_beta=float(cfg.robust_mpc_softmin_beta),
                sigmoid_temp=float(cfg.robust_mpc_sigmoid_temp),
            )
        else:
            controls = mpc_track(
                state0,
                ref_xy=ref_xy,
                ref_yaw=ref_yaw,
                dt=float(cfg.dt),
                horizon_steps=int(cfg.mpc_horizon_steps),
                iters=int(cfg.mpc_iters),
                lr=float(cfg.mpc_lr),
            )

    # Roll out the selected controls to produce an ego trajectory.
    # We prepend the initial state (t=0) for downstream consumers (e.g., nuPlan).
    ego_traj_xy: Optional[torch.Tensor]
    ego_traj_yaw: Optional[torch.Tensor]
    if controls is not None:
        st = rollout_bicycle(state0, controls, dt=float(cfg.dt))
        xy_exec = torch.stack([st.x, st.y], dim=-1)  # [B,H_mpc,2] at t=dt..H_mpc*dt
        yaw_exec = st.yaw                             # [B,H_mpc]
        zero_xy = torch.zeros((B, 1, 2), device=xy_exec.device, dtype=xy_exec.dtype)
        zero_yaw = torch.zeros((B, 1), device=yaw_exec.device, dtype=yaw_exec.dtype)
        xy_exec = torch.cat([zero_xy, xy_exec], dim=1)   # [B,H_mpc+1]
        yaw_exec = torch.cat([zero_yaw, yaw_exec], dim=1)

        # Extend to full horizon by appending a shifted reference tail.
        if xy_exec.shape[1] >= H_out:
            ego_traj_xy = xy_exec[:, :H_out]
            ego_traj_yaw = yaw_exec[:, :H_out]
        else:
            anchor_idx = xy_exec.shape[1] - 1
            ref_anchor_xy = ref_xy_full[:, anchor_idx, :]
            ref_anchor_yaw = ref_yaw_full[:, anchor_idx]
            exec_anchor_xy = xy_exec[:, anchor_idx, :]
            exec_anchor_yaw = yaw_exec[:, anchor_idx]

            delta_xy = ref_xy_full[:, anchor_idx:, :] - ref_anchor_xy.unsqueeze(1)
            delta_yaw = ref_yaw_full[:, anchor_idx:] - ref_anchor_yaw.unsqueeze(1)

            tail_xy = exec_anchor_xy.unsqueeze(1) + delta_xy
            tail_yaw = exec_anchor_yaw.unsqueeze(1) + delta_yaw

            # Remove duplicated anchor point.
            ego_traj_xy = torch.cat([xy_exec, tail_xy[:, 1:]], dim=1)
            ego_traj_yaw = torch.cat([yaw_exec, tail_yaw[:, 1:]], dim=1)
    else:
        # Fallback: output the reference trajectory (already full-horizon).
        ego_traj_xy = ref_xy_full
        ego_traj_yaw = ref_yaw_full

    return PlannerOutput(
        best_maneuver=best_m,
        best_score=best_score,
        per_maneuver_score=per_score,
        per_maneuver_risk=per_risk,
        per_maneuver_cvar=per_cvar,
        controls=controls,
        ego_traj_xy=ego_traj_xy,
        ego_traj_yaw=ego_traj_yaw,
    )
