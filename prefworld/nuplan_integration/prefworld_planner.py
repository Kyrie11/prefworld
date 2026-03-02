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


@dataclass
class NuPlanAdapterConfig:
    """Configuration for converting nuPlan inputs into PrefWorld batches."""

    max_agents: int = 20
    past_horizon_s: float = 2.0
    past_num_samples: int = 20
    future_horizon_s: float = 8.0
    future_num_samples: int = 80
    # Map sampling
    max_map_polylines: int = 200
    points_per_polyline: int = 20


class PrefWorldNuPlanPlanner:
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
        start_time_us = 0
        if hasattr(current_input, "history") and hasattr(current_input.history, "current_state"):
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

        # ------------------------------------------------------------------
        # Ego past (t,x,y,yaw,vx,vy,ax,ay,steering)
        # ------------------------------------------------------------------
        ego_hist = []
        try:
            ego_states = list(current_input.history.ego_states)  # type: ignore
        except Exception as e:
            raise RuntimeError("Unable to access ego history from nuPlan PlannerInput") from e

        # Use last cfg.past_num_samples states.
        ego_states = ego_states[-cfg.past_num_samples :]
        t0 = ego_states[-1].time_point.time_s  # type: ignore
        for st in ego_states:
            # nuPlan EgoState has: rear_axle pose, velocities, accelerations
            pose = st.rear_axle  # type: ignore
            vx = float(getattr(st.dynamic_car_state.rear_axle_velocity_2d, "x", 0.0))
            vy = float(getattr(st.dynamic_car_state.rear_axle_velocity_2d, "y", 0.0))
            ax = float(getattr(st.dynamic_car_state.rear_axle_acceleration_2d, "x", 0.0))
            ay = float(getattr(st.dynamic_car_state.rear_axle_acceleration_2d, "y", 0.0))
            steering = float(getattr(st.tire_steering_angle, "value", 0.0))
            ego_hist.append(
                [
                    float(st.time_point.time_s - t0),
                    float(pose.x),
                    float(pose.y),
                    float(pose.heading),
                    vx,
                    vy,
                    ax,
                    ay,
                    steering,
                ]
            )
        ego_hist = np.asarray(ego_hist, dtype=np.float32)

        # ------------------------------------------------------------------
        # Agents current state (x,y,yaw,vx,vy,length,width) in ego-local frame
        # ------------------------------------------------------------------
        # This is a minimal adapter: use the current tracked objects only.
        try:
            tracks = current_input.history.current_state.tracked_objects  # type: ignore
        except Exception as e:
            raise RuntimeError("Unable to access tracked objects from nuPlan PlannerInput") from e

        # Ego-local transform
        ego_x, ego_y, ego_yaw = ego_hist[-1, 1], ego_hist[-1, 2], ego_hist[-1, 3]
        cy, sy = np.cos(-ego_yaw), np.sin(-ego_yaw)

        def to_local_xy(x: float, y: float) -> tuple[float, float]:
            dx, dy = x - ego_x, y - ego_y
            return (cy * dx - sy * dy, sy * dx + cy * dy)

        def to_local_vel(vx: float, vy: float) -> tuple[float, float]:
            return (cy * vx - sy * vy, sy * vx + cy * vy)

        agents = np.zeros((cfg.max_agents, 7), dtype=np.float32)
        agents_mask = np.zeros((cfg.max_agents,), dtype=np.float32)
        # Iterate over tracked objects and keep vehicles only.
        count = 0
        for obj in getattr(tracks, "tracked_objects", tracks):  # handles both container styles
            if count >= cfg.max_agents:
                break
            try:
                # Vehicle-like objects expose oriented_box and velocity
                box = obj.box  # type: ignore
                xg, yg = float(box.center.x), float(box.center.y)
                yawg = float(box.center.heading)
                vxg = float(getattr(obj.velocity, "x", 0.0))
                vyg = float(getattr(obj.velocity, "y", 0.0))
                length = float(getattr(box, "length", 4.5))
                width = float(getattr(box, "width", 2.0))
            except Exception:
                continue

            xl, yl = to_local_xy(xg, yg)
            vxl, vyl = to_local_vel(vxg, vyg)
            yawl = float(np.arctan2(np.sin(yawg - ego_yaw), np.cos(yawg - ego_yaw)))

            agents[count] = np.array([xl, yl, yawl, vxl, vyl, length, width], dtype=np.float32)
            agents_mask[count] = 1.0
            count += 1

        # ------------------------------------------------------------------
        # Map polylines
        # ------------------------------------------------------------------
        # NOTE: Map extraction depends heavily on nuPlan map API. We provide a safe fallback:
        map_polylines = np.zeros((cfg.max_map_polylines, cfg.points_per_polyline, 2), dtype=np.float32)
        map_mask = np.zeros((cfg.max_map_polylines,), dtype=np.float32)
        map_type = np.zeros((cfg.max_map_polylines,), dtype=np.int64)

        # ------------------------------------------------------------------
        # Build tensors
        # ------------------------------------------------------------------
        # PrefWorld expects shapes:
        #  ego_hist:      [B,Tp,3]   (x,y,yaw) in ego-local frame
        #  ego_dyn_hist:  [B,Tp,4]   (vx,vy,ax,ay) in ego-local frame
        #  agents_hist:   [B,N,Tp,7]
        #  agents_hist_mask: [B,N,Tp]
        #  map_polylines: [B,M,L,2]
        #  map_poly_mask: [B,M]
        #  structure_t_rule: [B,1+N,1+N]
        #
        # For online planning, we may only have the current snapshot; replicate across Tp.
        Tp = int(ego_hist.shape[0])
        agents_hist = np.tile(agents[:, None, :], (1, Tp, 1))  # [N,Tp,7]
        agents_hist_mask = np.tile(agents_mask[:, None], (1, Tp))

        # Ego history in ego-local frame (current ego pose is origin)
        ego_hist_local = np.zeros((Tp, 3), dtype=np.float32)
        ego_dyn_local = np.zeros((Tp, 4), dtype=np.float32)
        for k in range(Tp):
            xg, yg, yawg = float(ego_hist[k, 1]), float(ego_hist[k, 2]), float(ego_hist[k, 3])
            vxg, vyg = float(ego_hist[k, 4]), float(ego_hist[k, 5])
            axg, ayg = float(ego_hist[k, 6]), float(ego_hist[k, 7])

            xl, yl = to_local_xy(xg, yg)
            vxl, vyl = to_local_vel(vxg, vyg)
            axl, ayl = to_local_vel(axg, ayg)
            yawl = float(np.arctan2(np.sin(yawg - ego_yaw), np.cos(yawg - ego_yaw)))

            ego_hist_local[k] = np.array([xl, yl, yawl], dtype=np.float32)
            ego_dyn_local[k] = np.array([vxl, vyl, axl, ayl], dtype=np.float32)

        K = 1 + int(cfg.max_agents)
        structure_t = np.zeros((1, K, K), dtype=np.float32)

        batch_np: Dict[str, Any] = {
            "ego_hist": ego_hist_local[None, :, :],
            "ego_dyn_hist": ego_dyn_local[None, :, :],
            "agents_hist": agents_hist[None, :, :, :],
            "agents_hist_mask": agents_hist_mask[None, :, :],
            "map_polylines": map_polylines[None, :, :, :],
            "map_poly_mask": map_mask[None, :],
            "map_poly_type": map_type[None, :],
            # Required by model.forward (planning uses candidate ego maneuvers)
            "ego_maneuver": np.zeros((1, 1), dtype=np.int64),
            # Safe default: assume no interaction edges at current time.
            "structure_t_rule": structure_t,
            # For converting local planned trajectory back to global nuPlan frame.
            "ego_global_pose": np.array([[ego_x, ego_y, ego_yaw]], dtype=np.float32),
        }

        batch_t = {k: torch.as_tensor(v, device=self.device) for k, v in batch_np.items()}
        return batch_t
