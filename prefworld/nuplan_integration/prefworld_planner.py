"""nuPlan integration.

This module provides a minimal planner wrapper that can be plugged into the
nuPlan simulation stack. It is written to be import-safe when the nuPlan devkit
is not installed.

The planner logic is delegated to :func:`prefworld.planning.planner.plan_with_structures`.

IMPORTANT:
  nuPlan APIs evolve; this file intentionally uses defensive imports and
  avoids hard-coding too many internal details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from prefworld.models.prefworld_model import PrefWorldModel
from prefworld.planning.planner import PlannerConfig, plan_with_structures


# Defensive nuPlan imports (module must remain import-safe without nuPlan installed)
try:  # pragma: no cover
    from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner  # type: ignore
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput  # type: ignore
except Exception:  # pragma: no cover
    AbstractPlanner = object  # type: ignore
    PlannerInitialization = Any  # type: ignore
    PlannerInput = Any  # type: ignore


@dataclass
class NuPlanAdapterConfig:
    """Configuration for converting nuPlan inputs into PrefWorld batches."""

    max_agents: int = 20
    agent_radius_m: float = 50.0
    keep_only_vehicles: bool = True
    past_horizon_s: float = 2.0
    past_num_samples: int = 20
    future_horizon_s: float = 8.0
    future_num_samples: int = 80
    # Map sampling
    max_map_polylines: int = 200
    points_per_polyline: int = 20
    map_radius_m: float = 60.0
    include_route: bool = True
    # If True, raise when feature extraction fails (prevents silent fallbacks to all-zeros).
    strict_features: bool = True
    # Structure extraction (rule-based, no future leakage)
    structure_conflict_dist_m: float = 3.0
    structure_conflict_radius_m: float = 2.0


class PrefWorldNuPlanPlanner(AbstractPlanner):
    """A nuPlan-compatible planner that wraps a trained PrefWorldModel.

    The planner:
      1) builds a PrefWorld batch from nuPlan simulation input
      2) infers preferences z via preference completion
      3) samples interaction structures via EB-STM
      4) scores ego maneuvers by expected/CVaR collision risk
      5) returns the best ego trajectory

    If nuPlan is not installed, instantiation will raise ImportError.
    """

    def __init__(
        self,
        *,
        model: PrefWorldModel,
        device: torch.device | str = "cuda",
        planner_cfg: Optional[PlannerConfig] = None,
        adapter_cfg: Optional[NuPlanAdapterConfig] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.planner_cfg = planner_cfg or PlannerConfig()
        self.adapter_cfg = adapter_cfg or NuPlanAdapterConfig()

        self.model.to(self.device)
        self.model.eval()

        # Defer nuPlan imports until init to keep the package import-safe.
        try:
            import nuplan  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "nuPlan devkit is not installed. Install nuplan-devkit to use PrefWorldNuPlanPlanner."
            ) from e

    # ---------------------------------------------------------------------
    # nuPlan interface (duck-typed)
    # ---------------------------------------------------------------------
    def initialize(self, initialization: Any) -> None:  # pragma: no cover
        """Called once at the beginning of simulation."""
        self._initialization = initialization

    def compute_planner_trajectory(self, current_input: Any) -> Any:  # pragma: no cover
        """Compute and return the next ego trajectory in nuPlan format."""
        batch = self._build_batch_from_nuplan(current_input)

        with torch.no_grad():
            # plan_with_structures internally moves tensors to the model device.
            res = plan_with_structures(self.model, batch, cfg=self.planner_cfg)

        # Convert to nuPlan trajectory. PrefWorld planner outputs in ego-local frame
        # (current ego pose is the origin). Transform back to nuPlan's global frame.
        ego_xy_local = res.ego_traj_xy[0].detach().cpu().numpy()  # [H,2] local
        ego_yaw_local = res.ego_traj_yaw[0].detach().cpu().numpy()  # [H] local

        ego_pose = batch.get("ego_global_pose", None)
        if ego_pose is not None:
            x0, y0, yaw0 = ego_pose[0].detach().cpu().numpy().tolist()
        else:
            # Fallback: treat local as global
            x0, y0, yaw0 = 0.0, 0.0, 0.0

        cy, sy = float(np.cos(yaw0)), float(np.sin(yaw0))
        ego_xy = np.empty_like(ego_xy_local)
        ego_xy[:, 0] = x0 + cy * ego_xy_local[:, 0] - sy * ego_xy_local[:, 1]
        ego_xy[:, 1] = y0 + sy * ego_xy_local[:, 0] + cy * ego_xy_local[:, 1]
        ego_yaw = (ego_yaw_local + yaw0 + np.pi) % (2 * np.pi) - np.pi

        try:
            from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
            from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
        except Exception:
            # Fallback: return raw arrays
            return {"ego_xy": ego_xy, "ego_yaw": ego_yaw}

        # Build a list of StateSE2 with monotonically increasing TimePoints.
        dt = float(self.planner_cfg.dt)

        # Prefer the timestamp from the ego state vector used during feature extraction.
        start_time_us = 0
        try:
            if "ego_time_us" in batch:
                start_time_us = int(batch["ego_time_us"][0].detach().cpu().item())
        except Exception:
            start_time_us = 0

        # Fallback: attempt to read from nuPlan's history/current_state
        if start_time_us == 0 and hasattr(current_input, "history") and hasattr(current_input.history, "current_state"):
            # nuPlan stores timestamps in microseconds on EgoState
            try:
                start_time_us = int(current_input.history.current_state.time_point.time_us)
            except Exception:
                start_time_us = 0
        states = []
        for k in range(ego_xy.shape[0]):
            tp = TimePoint(start_time_us + int(round(1e6 * dt * k)))
            states.append(StateSE2(ego_xy[k, 0], ego_xy[k, 1], ego_yaw[k], tp))
        return InterpolatedTrajectory(states)

    # ---------------------------------------------------------------------
    # Batch building
    # ---------------------------------------------------------------------
    def _build_batch_from_nuplan(self, current_input: Any) -> Dict[str, torch.Tensor]:  # pragma: no cover
        """Convert nuPlan PlannerInput into a PrefWorld batch.

        This is necessarily approximate because nuPlan can be configured with
        different history buffers and map APIs.

        The returned dict mirrors keys expected by PrefWorldModel.forward and
        plan_with_structures.
        """

        cfg = self.adapter_cfg

        # Import PrefWorld extractor helpers lazily (keeps module import-safe).
        from prefworld.data.extractor import (
            ExtractionConfig,
            _agent_pose_vel_size,
            _ego_state_to_vector,
            _get_track_token,
            _traffic_light_dict,
            _extract_map_polylines,
            _compute_structure_and_confidence_from_futures,
        )
        from prefworld.utils.geometry import global_to_local_pose, global_to_local_xy

        # ------------------------------------------------------------------
        # Ego history
        # ------------------------------------------------------------------
        try:
            ego_states = list(current_input.history.ego_states)  # type: ignore
        except Exception as e:
            raise RuntimeError("Unable to access ego history from nuPlan PlannerInput") from e
        if len(ego_states) == 0:
            raise RuntimeError("nuPlan PlannerInput contains empty ego history")

        Tp = int(cfg.past_num_samples)
        if len(ego_states) < Tp:
            ego_states = [ego_states[0]] * (Tp - len(ego_states)) + ego_states
        ego_states = ego_states[-Tp:]

        # current ego pose (global)
        # Use the same conversion helper as the offline extractor for consistency.
        ego_vec = _ego_state_to_vector(ego_states[-1])
        ego_time_us = int(ego_vec[0])
        ego_xy = np.array([float(ego_vec[1]), float(ego_vec[2])], dtype=np.float32)
        ego_yaw = float(ego_vec[3])

        # ------------------------------------------------------------------
        # Observations (tracked objects) history
        # ------------------------------------------------------------------
        hist = current_input.history
        obs_list = None
        for attr in ("observations", "observation_buffer", "observations_buffer"):
            if hasattr(hist, attr):
                try:
                    obs_list = list(getattr(hist, attr))
                    break
                except Exception:
                    obs_list = None

        if obs_list is None:
            # Fallback: replicate the current tracked objects snapshot.
            try:
                obs_list = [hist.current_state] * Tp
            except Exception:
                obs_list = [getattr(hist, "current_state", None)] * Tp

        if len(obs_list) < Tp:
            obs_list = [obs_list[0]] * (Tp - len(obs_list)) + obs_list
        obs_list = obs_list[-Tp:]

        def _iter_objects(obs: Any):
            if obs is None:
                return []
            # typical: Observation has .tracked_objects
            cont = getattr(obs, "tracked_objects", obs)
            objs = getattr(cont, "tracked_objects", None)
            if objs is not None:
                try:
                    return list(objs)
                except Exception:
                    return []
            try:
                return list(cont)
            except Exception:
                return []

        # ------------------------------------------------------------------
        # Select agents from the current snapshot
        # ------------------------------------------------------------------
        curr_objs = _iter_objects(obs_list[-1])
        cand = []
        for o in curr_objs:
            if bool(cfg.keep_only_vehicles):
                t = getattr(o, "tracked_object_type", None)
                if t is None:
                    t = getattr(o, "object_type", None)
                if t is not None and ("vehicle" not in str(t).lower()):
                    continue
            try:
                token = _get_track_token(o)
                pose, vel, length, width = _agent_pose_vel_size(o)
            except Exception:
                continue
            pose_local = global_to_local_pose(pose[None, :], ego_xy, ego_yaw)[0]
            d = float(np.linalg.norm(pose_local[:2]))
            if d > float(cfg.agent_radius_m):
                continue
            cand.append((d, token))
        cand.sort(key=lambda x: x[0])
        track_tokens = [t for _, t in cand[: int(cfg.max_agents)]]

        # ------------------------------------------------------------------
        # Build agents history arrays in ego-local frame of current ego
        # ------------------------------------------------------------------
        N = int(cfg.max_agents)
        agents_hist = np.zeros((N, Tp, 7), dtype=np.float32)
        agents_hist_mask = np.zeros((N, Tp), dtype=np.float32)

        for k in range(Tp):
            objs_k = _iter_objects(obs_list[k])
            token_to_obj = {}
            for o in objs_k:
                try:
                    token_to_obj[_get_track_token(o)] = o
                except Exception:
                    continue

            for i, token in enumerate(track_tokens):
                if i >= N:
                    break
                o = token_to_obj.get(token, None)
                if o is None:
                    continue
                try:
                    pose, vel, length, width = _agent_pose_vel_size(o)
                except Exception:
                    continue
                pose_local = global_to_local_pose(pose[None, :], ego_xy, ego_yaw)[0]
                vel_local = global_to_local_xy(vel[None, :], np.zeros(2, dtype=np.float32), ego_yaw)[0]
                agents_hist[i, k] = np.array([pose_local[0], pose_local[1], pose_local[2], vel_local[0], vel_local[1], length, width], dtype=np.float32)
                agents_hist_mask[i, k] = 1.0

        # ------------------------------------------------------------------
        # Ego history in ego-local frame (current ego is origin)
        # ------------------------------------------------------------------
        ego_hist_local = np.zeros((Tp, 3), dtype=np.float32)
        ego_dyn_local = np.zeros((Tp, 4), dtype=np.float32)
        for k, st in enumerate(ego_states):
            ev = _ego_state_to_vector(st)
            pose_g = np.array([float(ev[1]), float(ev[2]), float(ev[3])], dtype=np.float32)
            ego_hist_local[k] = global_to_local_pose(pose_g[None, :], ego_xy, ego_yaw)[0]

            vxg, vyg = float(ev[4]), float(ev[5])
            axg, ayg = float(ev[6]), float(ev[7])
            vel_local = global_to_local_xy(np.array([[vxg, vyg]], dtype=np.float32), np.zeros(2, dtype=np.float32), ego_yaw)[0]
            acc_local = global_to_local_xy(np.array([[axg, ayg]], dtype=np.float32), np.zeros(2, dtype=np.float32), ego_yaw)[0]
            ego_dyn_local[k] = np.array([vel_local[0], vel_local[1], acc_local[0], acc_local[1]], dtype=np.float32)

        # ------------------------------------------------------------------
        # Map extraction (match offline extractor as closely as possible)
        # ------------------------------------------------------------------
        ex_cfg = ExtractionConfig(
            past_horizon_s=float(cfg.past_horizon_s),
            past_num_samples=int(cfg.past_num_samples),
            future_horizon_s=float(cfg.future_horizon_s),
            future_num_samples=int(cfg.future_num_samples),
            max_agents=int(cfg.max_agents),
            agent_radius_m=float(cfg.agent_radius_m),
            keep_only_vehicles=bool(cfg.keep_only_vehicles),
            map_radius_m=float(cfg.map_radius_m),
            max_map_polylines=int(cfg.max_map_polylines),
            polyline_points=int(cfg.points_per_polyline),
            include_route=bool(cfg.include_route),
            structure_conflict_dist_m=float(cfg.structure_conflict_dist_m),
            structure_conflict_radius_m=float(cfg.structure_conflict_radius_m),
        )

        map_polylines = np.zeros((int(cfg.max_map_polylines), int(cfg.points_per_polyline), 2), dtype=np.float32)
        map_mask = np.zeros((int(cfg.max_map_polylines),), dtype=np.float32)
        map_type = np.zeros((int(cfg.max_map_polylines),), dtype=np.int64)
        map_tl = np.zeros((int(cfg.max_map_polylines),), dtype=np.int64)
        map_on_route = np.zeros((int(cfg.max_map_polylines),), dtype=np.int64)

        try:
            map_api = getattr(self._initialization, "map_api", None)
            route_ids = getattr(self._initialization, "route_roadblock_ids", []) or []

            # traffic lights
            tl_statuses = None
            for name in (
                "traffic_light_data",
                "traffic_light_status_data",
                "traffic_light_statuses",
                "traffic_light_status",
            ):
                if hasattr(current_input, name):
                    tl_statuses = getattr(current_input, name)
                    break
                if hasattr(hist, "current_state") and hasattr(hist.current_state, name):
                    tl_statuses = getattr(hist.current_state, name)
                    break
            tl_dict = _traffic_light_dict(tl_statuses) if tl_statuses is not None else {}

            if map_api is not None:
                mf = _extract_map_polylines(
                    map_api=map_api,
                    ego_xy=ego_xy,
                    ego_yaw=float(ego_yaw),
                    route_roadblock_ids=list(route_ids),
                    tl_dict=tl_dict,
                    cfg=ex_cfg,
                )
                map_polylines = mf["map_polylines"].astype(np.float32)
                map_mask = mf["map_poly_mask"].astype(np.float32)
                map_type = mf["map_poly_type"].astype(np.int64)
                map_tl = mf.get("map_tl_status", np.zeros_like(map_type)).astype(np.int64)
                map_on_route = mf.get("map_on_route", np.zeros_like(map_type)).astype(np.int64)
            else:
                if bool(getattr(cfg, "strict_features", True)):
                    raise RuntimeError("nuPlan map_api not available (strict_features=True)")
        except Exception as e:
            if bool(getattr(cfg, "strict_features", True)):
                raise RuntimeError(f"nuPlan map extraction failed (strict_features=True): {e}")
            # Non-strict fallback: keep safe zero map
            pass

        # ------------------------------------------------------------------
        # Rule-based structure at current time (no future leakage)
        # ------------------------------------------------------------------
        K = 1 + int(cfg.max_agents)
        dt_fut = float(cfg.future_horizon_s) / max(1, int(cfg.future_num_samples))
        horizon_rule = min(3.0, float(cfg.future_horizon_s))
        T_rule = int(round(horizon_rule / dt_fut)) + 1

        traj_xy_rule = np.zeros((K, T_rule, 2), dtype=np.float32)
        vel_rule = np.zeros((K, 2), dtype=np.float32)
        traj_xy_rule[0, 0] = np.array([0.0, 0.0], dtype=np.float32)
        # ego velocity in ego frame
        ego_vel_local = ego_dyn_local[-1, :2]
        vel_rule[0] = ego_vel_local
        # agents
        for i in range(int(cfg.max_agents)):
            if agents_hist_mask[i, -1] < 0.5:
                continue
            traj_xy_rule[1 + i, 0] = agents_hist[i, -1, :2]
            vel_rule[1 + i] = agents_hist[i, -1, 3:5]
        for t in range(1, T_rule):
            tt = float(t) * dt_fut
            traj_xy_rule[:, t] = traj_xy_rule[:, 0] + vel_rule * tt
        valid = np.zeros((K,), dtype=np.float32)
        valid[0] = 1.0
        valid[1:] = agents_hist_mask[:, -1]
        structure_t_rule, structure_conf_t = _compute_structure_and_confidence_from_futures(traj_xy_rule, valid, ex_cfg)

        batch_np: Dict[str, Any] = {
            "ego_hist": ego_hist_local[None, :, :],
            "ego_dyn_hist": ego_dyn_local[None, :, :],
            "agents_hist": agents_hist[None, :, :, :],
            "agents_hist_mask": agents_hist_mask[None, :, :],
            "map_polylines": map_polylines[None, :, :, :],
            "map_poly_mask": map_mask[None, :],
            "map_poly_type": map_type[None, :],
            "map_tl_status": map_tl[None, :],
            "map_on_route": map_on_route[None, :],
            # planning-time placeholder (not used for inference)
            "ego_maneuver": np.zeros((1, 1), dtype=np.int64),
            "structure_t_rule": structure_t_rule[None, :, :],
            "structure_conf_t": np.array([float(structure_conf_t)], dtype=np.float32),
            # local->global conversion
            "ego_global_pose": np.array([[float(ego_xy[0]), float(ego_xy[1]), float(ego_yaw)]], dtype=np.float32),
            # absolute time (microseconds) of the current ego state
            "ego_time_us": np.array([ego_time_us], dtype=np.int64),
        }

        return {k: torch.as_tensor(v, device=self.device) for k, v in batch_np.items()}
