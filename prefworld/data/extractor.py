from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from prefworld.data.labels import Maneuver
from prefworld.utils.geometry import global_to_local_pose, global_to_local_xy, pose_to_xy_yaw, wrap_angle

try:
    # nuPlan imports (required at runtime)
    from nuplan.common.actor_state.state_representation import Point2D  # type: ignore
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # type: ignore
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType  # type: ignore
except Exception:  # pragma: no cover
    Point2D = None  # type: ignore
    SemanticMapLayer = None  # type: ignore
    TrackedObjectType = None  # type: ignore


@dataclass
class ExtractionConfig:
    # Temporal windows
    past_horizon_s: float = 2.0
    past_num_samples: int = 20
    future_horizon_s: float = 8.0
    future_num_samples: int = 80

    # Agent selection
    max_agents: int = 20
    agent_radius_m: float = 50.0
    keep_only_vehicles: bool = True

    # Map extraction
    map_radius_m: float = 60.0
    max_map_polylines: int = 256
    polyline_points: int = 20
    include_route: bool = True

    # Structure labeling
    structure_delta_s: float = 1.0
    structure_conflict_dist_m: float = 3.0
    structure_conflict_radius_m: float = 2.0

    # Maneuver labeling thresholds
    stop_dist_threshold_m: float = 1.0
    lateral_threshold_m: float = 2.0
    turn_threshold_rad: float = 0.4


def _ego_state_to_vector(ego_state) -> np.ndarray:
    """Convert nuPlan EgoState to numeric vector (9 dims)."""
    # EgoState implements __iter__ yielding:
    # (time_us, rear_axle.x, rear_axle.y, rear_axle.heading,
    #  v_x, v_y, a_x, a_y, tire_steering_angle)
    vec = np.array(list(ego_state), dtype=np.float32)
    assert vec.shape == (9,), f"Unexpected EgoState vector shape: {vec.shape}"
    return vec


def _agent_pose_vel_size(agent) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Extract (pose[x,y,yaw], velocity[vx,vy], length, width) from nuPlan Agent/AgentState."""
    # Pose
    if hasattr(agent, "box"):
        box = agent.box
    elif hasattr(agent, "oriented_box"):
        box = agent.oriented_box
    else:
        raise AttributeError("Agent does not have box/oriented_box attribute")

    if hasattr(box, "center"):
        center = box.center
        x, y, yaw = pose_to_xy_yaw(center)
    else:
        # Fallback: some versions store center x/y/heading directly
        x = float(getattr(box, "x"))
        y = float(getattr(box, "y"))
        yaw = float(getattr(box, "heading"))

    pose = np.array([x, y, yaw], dtype=np.float32)

    # Velocity
    vel = getattr(agent, "velocity", None)
    if vel is None:
        # Some objects might not have velocity; assume zero
        vx, vy = 0.0, 0.0
    else:
        vx = float(getattr(vel, "x", 0.0))
        vy = float(getattr(vel, "y", 0.0))
    vel_vec = np.array([vx, vy], dtype=np.float32)

    length = float(getattr(box, "length", 0.0))
    width = float(getattr(box, "width", 0.0))
    return pose, vel_vec, length, width


def _get_track_token(agent) -> str:
    meta = getattr(agent, "metadata", None)
    if meta is not None and hasattr(meta, "track_token"):
        return str(meta.track_token)
    # Some versions use `token`
    if meta is not None and hasattr(meta, "token"):
        return str(meta.token)
    # As a fallback, hash object id (not stable across frames!)
    return str(id(agent))


def _get_tracked_objects_list(detections_tracks) -> List:
    # DetectionsTracks wraps TrackedObjects in a `.tracked_objects` attribute
    if hasattr(detections_tracks, "tracked_objects"):
        tracked = detections_tracks.tracked_objects
        # tracked may be a TrackedObjects container with `.tracked_objects` list
        if hasattr(tracked, "tracked_objects"):
            return list(tracked.tracked_objects)
        if isinstance(tracked, list):
            return tracked
    # fallback: already a list
    if isinstance(detections_tracks, list):
        return detections_tracks
    raise AttributeError("Unable to extract tracked objects list from detections_tracks")


def _traffic_light_dict(tl_statuses) -> Dict[str, int]:
    """Map lane_connector_id -> status int (0 unknown, 1 green, 2 yellow, 3 red)."""
    d: Dict[str, int] = {}
    if tl_statuses is None:
        return d
    # TrafficLightStatuses is iterable in nuPlan
    items = list(tl_statuses) if not isinstance(tl_statuses, list) else tl_statuses
    for item in items:
        # item is TrafficLightStatusData
        lc_id = getattr(item, "lane_connector_id", None)
        if lc_id is None:
            continue
        status = getattr(item, "status", None)
        # status might be enum with name/value
        sname = str(status.name).upper() if hasattr(status, "name") else str(status).upper()
        if "GREEN" in sname:
            code = 1
        elif "YELLOW" in sname:
            code = 2
        elif "RED" in sname:
            code = 3
        else:
            code = 0
        d[str(lc_id)] = code
    return d


def _sample_polyline_xy(polyline_xy: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a polyline to a fixed number of points by uniform indexing."""
    if polyline_xy.shape[0] == 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    if polyline_xy.shape[0] == num_points:
        return polyline_xy.astype(np.float32)
    # Uniform indices
    idx = np.linspace(0, polyline_xy.shape[0] - 1, num_points).round().astype(int)
    return polyline_xy[idx].astype(np.float32)


def _extract_map_polylines(
    map_api,
    ego_xy: np.ndarray,
    ego_yaw: float,
    route_roadblock_ids: Optional[List[str]],
    tl_dict: Dict[str, int],
    cfg: ExtractionConfig,
) -> Dict[str, np.ndarray]:
    """Extract lane + lane-connector polylines around ego."""
    if Point2D is None or SemanticMapLayer is None:
        raise ImportError("nuPlan is required for map extraction")

    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    prox = map_api.get_proximal_map_objects(Point2D(float(ego_xy[0]), float(ego_xy[1])), cfg.map_radius_m, layers)

    polylines: List[np.ndarray] = []
    poly_type: List[int] = []
    tl_status: List[int] = []
    on_route: List[int] = []

    # Lanes
    for lane in prox.get(SemanticMapLayer.LANE, []):
        try:
            path = lane.baseline_path.discrete_path
            pts = np.array([[p.x, p.y] for p in path], dtype=np.float32)
        except Exception:
            continue
        pts_local = global_to_local_xy(pts, ego_xy, ego_yaw)
        pts_local = _sample_polyline_xy(pts_local, cfg.polyline_points)
        polylines.append(pts_local)
        poly_type.append(0)
        tl_status.append(0)
        if cfg.include_route and route_roadblock_ids is not None:
            try:
                rb_id = lane.get_roadblock_id()
                on_route.append(1 if str(rb_id) in set(route_roadblock_ids) else 0)
            except Exception:
                on_route.append(0)
        else:
            on_route.append(0)

    # Lane connectors (+ traffic lights)
    for conn in prox.get(SemanticMapLayer.LANE_CONNECTOR, []):
        try:
            path = conn.baseline_path.discrete_path
            pts = np.array([[p.x, p.y] for p in path], dtype=np.float32)
        except Exception:
            continue
        pts_local = global_to_local_xy(pts, ego_xy, ego_yaw)
        pts_local = _sample_polyline_xy(pts_local, cfg.polyline_points)
        polylines.append(pts_local)
        poly_type.append(1)
        conn_id = str(getattr(conn, "id", getattr(conn, "fid", "")))
        tl_status.append(int(tl_dict.get(conn_id, 0)))
        on_route.append(0)

    if len(polylines) == 0:
        polylines_arr = np.zeros((cfg.max_map_polylines, cfg.polyline_points, 2), dtype=np.float32)
        mask_arr = np.zeros((cfg.max_map_polylines,), dtype=np.float32)
        return {
            "map_polylines": polylines_arr,
            "map_poly_mask": mask_arr,
            "map_poly_type": np.zeros((cfg.max_map_polylines,), dtype=np.int64),
            "map_tl_status": np.zeros((cfg.max_map_polylines,), dtype=np.int64),
            "map_on_route": np.zeros((cfg.max_map_polylines,), dtype=np.int64),
        }

    # Sort by distance of polyline first point to ego (in local)
    dists = [float(np.linalg.norm(pl[0])) for pl in polylines]
    order = np.argsort(dists)[: cfg.max_map_polylines]
    polylines = [polylines[i] for i in order]
    poly_type = [poly_type[i] for i in order]
    tl_status = [tl_status[i] for i in order]
    on_route = [on_route[i] for i in order]

    M = len(polylines)
    polylines_arr = np.zeros((cfg.max_map_polylines, cfg.polyline_points, 2), dtype=np.float32)
    polylines_arr[:M] = np.stack(polylines, axis=0)
    mask_arr = np.zeros((cfg.max_map_polylines,), dtype=np.float32)
    mask_arr[:M] = 1.0
    return {
        "map_polylines": polylines_arr,
        "map_poly_mask": mask_arr,
        "map_poly_type": np.array(poly_type, dtype=np.int64).tolist()
        + [0] * (cfg.max_map_polylines - M),
        "map_tl_status": np.array(tl_status, dtype=np.int64).tolist()
        + [0] * (cfg.max_map_polylines - M),
        "map_on_route": np.array(on_route, dtype=np.int64).tolist()
        + [0] * (cfg.max_map_polylines - M),
    }


def _classify_maneuver(
    curr_pose: np.ndarray,
    future_pose: np.ndarray,
    cfg: ExtractionConfig,
) -> int:
    """Heuristic maneuver label from future kinematics."""
    # curr_pose: [x,y,yaw] local in ego frame
    # future_pose: [T,3] local in ego frame
    if future_pose.shape[0] < 2:
        return int(Maneuver.KEEP)
    # Use final state
    dx = float(future_pose[-1, 0] - curr_pose[0])
    dy = float(future_pose[-1, 1] - curr_pose[1])
    dist = math.hypot(dx, dy)
    if dist < cfg.stop_dist_threshold_m:
        return int(Maneuver.STOP)

    # Rotate displacement into agent's heading frame at current
    yaw = float(curr_pose[2])
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    lon = c * dx - s * dy
    lat = s * dx + c * dy

    # Estimate final heading
    dyaw = wrap_angle(float(future_pose[-1, 2] - curr_pose[2]))

    if abs(lat) > cfg.lateral_threshold_m and abs(lon) > 1.0:
        return int(Maneuver.LANE_CHANGE_LEFT if lat > 0 else Maneuver.LANE_CHANGE_RIGHT)
    if abs(dyaw) > cfg.turn_threshold_rad:
        return int(Maneuver.TURN_LEFT if dyaw > 0 else Maneuver.TURN_RIGHT)
    return int(Maneuver.KEEP)


def _compute_structure_from_futures(
    traj_xy: np.ndarray,  # [K, T, 2] in ego frame, includes ego at index 0
    valid_mask: np.ndarray,  # [K]
    cfg: ExtractionConfig,
) -> np.ndarray:
    """Compute a directed yield/precedence adjacency matrix from future trajectories.

    Edge convention: A[i,j] = 1 means i yields to j (j has precedence).
    """
    K, T, _ = traj_xy.shape
    A = np.zeros((K, K), dtype=np.int64)
    score = np.zeros((K, K), dtype=np.float32)  # confidence proxy for cycle breaking
    for i in range(K):
        if not valid_mask[i]:
            continue
        for j in range(i + 1, K):
            if not valid_mask[j]:
                continue
            # pairwise distance over time (synchronized)
            d = np.linalg.norm(traj_xy[i] - traj_xy[j], axis=-1)  # [T]
            t_min = int(np.argmin(d))
            d_min = float(d[t_min])
            if d_min > cfg.structure_conflict_dist_m:
                continue
            # define conflict point
            p_conf = 0.5 * (traj_xy[i, t_min] + traj_xy[j, t_min])
            # find earliest time each comes within radius
            r = cfg.structure_conflict_radius_m
            ti = None
            tj = None
            di = np.linalg.norm(traj_xy[i] - p_conf[None, :], axis=-1)
            dj = np.linalg.norm(traj_xy[j] - p_conf[None, :], axis=-1)
            idx_i = np.where(di < r)[0]
            idx_j = np.where(dj < r)[0]
            if len(idx_i) > 0:
                ti = int(idx_i[0])
            if len(idx_j) > 0:
                tj = int(idx_j[0])
            if ti is None or tj is None:
                continue
            if ti < tj:
                # j yields to i
                A[j, i] = 1
                score[j, i] = float(abs(tj - ti))
            elif tj < ti:
                A[i, j] = 1
                score[i, j] = float(abs(ti - tj))
            else:
                # tie: ignore
                pass

    # Enforce DAG (paper assumes acyclic precedence). Remove the weakest edge in each detected cycle.
    A = _enforce_acyclic(A, valid_mask=valid_mask, edge_score=score)
    return A


def _compute_structure_and_confidence_from_futures(
    traj_xy: np.ndarray,  # [K, T, 2] in ego frame, includes ego at index 0
    valid_mask: np.ndarray,  # [K]
    cfg: ExtractionConfig,
) -> Tuple[np.ndarray, np.float32]:
    """Compute structure adjacency and a lightweight confidence score w∈[0,1].

    This is a *practical* approximation of the paper's extractor confidence w_t.
    We measure how "decisive" each predicted precedence relation is:
      - closer encounters (smaller d_min) are more confident
      - larger time separation |t_i - t_j| at the conflict region is more confident

    The returned confidence is averaged over predicted edges. If no edges are
    predicted (no conflicts), we return w=1.0.
    """
    K, T, _ = traj_xy.shape
    A = np.zeros((K, K), dtype=np.int64)
    score = np.zeros((K, K), dtype=np.float32)  # used for cycle breaking
    edge_conf: List[float] = []

    for i in range(K):
        if not valid_mask[i]:
            continue
        for j in range(i + 1, K):
            if not valid_mask[j]:
                continue

            d = np.linalg.norm(traj_xy[i] - traj_xy[j], axis=-1)  # [T]
            t_min = int(np.argmin(d))
            d_min = float(d[t_min])
            if d_min > cfg.structure_conflict_dist_m:
                continue

            p_conf = 0.5 * (traj_xy[i, t_min] + traj_xy[j, t_min])
            r = float(cfg.structure_conflict_radius_m)
            di = np.linalg.norm(traj_xy[i] - p_conf[None, :], axis=-1)
            dj = np.linalg.norm(traj_xy[j] - p_conf[None, :], axis=-1)
            idx_i = np.where(di < r)[0]
            idx_j = np.where(dj < r)[0]
            if len(idx_i) == 0 or len(idx_j) == 0:
                continue
            ti = int(idx_i[0])
            tj = int(idx_j[0])

            # confidence proxy
            d_conf = max(0.0, min(1.0, (float(cfg.structure_conflict_dist_m) - d_min) / max(1e-6, float(cfg.structure_conflict_dist_m))))
            t_conf = max(0.0, min(1.0, float(abs(ti - tj)) / max(1.0, float(T - 1))))
            conf = 0.5 * d_conf + 0.5 * t_conf

            if ti < tj:
                A[j, i] = 1
                score[j, i] = float(abs(tj - ti))
                edge_conf.append(conf)
            elif tj < ti:
                A[i, j] = 1
                score[i, j] = float(abs(ti - tj))
                edge_conf.append(conf)
            else:
                # tie: ignore
                pass

    A = _enforce_acyclic(A, valid_mask=valid_mask, edge_score=score)
    if len(edge_conf) == 0:
        w = 1.0
    else:
        w = float(np.clip(np.mean(np.array(edge_conf, dtype=np.float32)), 0.0, 1.0))
    return A, np.float32(w)


def _find_cycle_edges(A: np.ndarray, valid_mask: np.ndarray) -> Optional[list[tuple[int, int]]]:
    """Return a list of directed edges (u,v) that form a cycle, or None if acyclic."""
    K = A.shape[0]
    valid = (valid_mask > 0.5)
    state = np.zeros((K,), dtype=np.int8)  # 0=unvisited,1=visiting,2=done
    parent = -np.ones((K,), dtype=np.int32)

    def dfs(u: int) -> Optional[list[tuple[int, int]]]:
        state[u] = 1
        for v in range(K):
            if not valid[v] or A[u, v] == 0:
                continue
            if state[v] == 0:
                parent[v] = u
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif state[v] == 1:
                # found back-edge u->v, reconstruct cycle edges
                edges: list[tuple[int, int]] = [(u, v)]
                cur = u
                while cur != v and cur != -1:
                    p = int(parent[cur])
                    if p == -1:
                        break
                    edges.append((p, cur))
                    cur = p
                return edges
        state[u] = 2
        return None

    for s in range(K):
        if not valid[s] or state[s] != 0:
            continue
        cyc = dfs(int(s))
        if cyc is not None:
            return cyc
    return None


def _enforce_acyclic(A: np.ndarray, *, valid_mask: np.ndarray, edge_score: Optional[np.ndarray] = None) -> np.ndarray:
    """Greedily remove edges until the directed graph is acyclic."""
    A = A.copy().astype(np.int64)
    if edge_score is None:
        edge_score = np.ones_like(A, dtype=np.float32)
    else:
        edge_score = edge_score.astype(np.float32)

    while True:
        cyc = _find_cycle_edges(A, valid_mask)
        if cyc is None:
            break
        # remove weakest edge in the cycle
        weakest = None
        weakest_s = float("inf")
        for (u, v) in cyc:
            s = float(edge_score[u, v])
            if s < weakest_s:
                weakest_s = s
                weakest = (u, v)
        if weakest is None:
            # fallback
            u, v = cyc[0]
        else:
            u, v = weakest
        A[u, v] = 0
    return A


def extract_sample(
    scenario,
    iteration: int,
    cfg: ExtractionConfig,
    *,
    include_future_agents: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract a single training sample from a nuPlan scenario at a given iteration."""
    ego_state = scenario.get_ego_state_at_iteration(iteration)
    ego_vec = _ego_state_to_vector(ego_state)
    # current origin pose (rear axle)
    ego_xy = ego_vec[1:3].copy()
    ego_yaw = float(ego_vec[3])

    # Ego past/future
    # NOTE: nuPlan's get_ego_past_trajectory / get_past_tracked_objects typically DO NOT include the
    # current frame at `iteration`. We explicitly append the current frame so that the last history
    # step corresponds to the current time (ego-local origin).
    # We therefore request Tp-1 past samples and append current -> Tp total.
    Tp = int(cfg.past_num_samples)
    past_num = max(0, Tp - 1)
    ego_past = list(scenario.get_ego_past_trajectory(iteration, cfg.past_horizon_s, past_num))
    ego_past.append(ego_state)
    # ensure length Tp
    if len(ego_past) != Tp:
        if len(ego_past) == 0:
            ego_past = [ego_state] * Tp
        elif len(ego_past) < Tp:
            ego_past = ego_past + [ego_past[-1]] * (Tp - len(ego_past))
        else:
            ego_past = ego_past[-Tp:]

    ego_future = list(scenario.get_ego_future_trajectory(iteration, cfg.future_horizon_s, cfg.future_num_samples))

    ego_past_vec = np.stack([_ego_state_to_vector(s) for s in ego_past], axis=0)  # [Tp, 9]
    ego_future_vec = np.stack([_ego_state_to_vector(s) for s in ego_future], axis=0)  # [Tf, 9]

    # Convert ego poses to local frame
    ego_past_pose = np.stack([ego_past_vec[:, 1], ego_past_vec[:, 2], ego_past_vec[:, 3]], axis=-1)
    ego_future_pose = np.stack([ego_future_vec[:, 1], ego_future_vec[:, 2], ego_future_vec[:, 3]], axis=-1)
    ego_past_local = global_to_local_pose(ego_past_pose, ego_xy, ego_yaw)
    ego_future_local = global_to_local_pose(ego_future_pose, ego_xy, ego_yaw)

    # Current tracked objects
    detections = scenario.get_tracked_objects_at_iteration(iteration)
    agents_list = _get_tracked_objects_list(detections)

    # Filter agents
    filtered = []
    for ag in agents_list:
        if cfg.keep_only_vehicles and TrackedObjectType is not None:
            if getattr(ag, "tracked_object_type", None) != TrackedObjectType.VEHICLE:
                continue
        pose, vel, length, width = _agent_pose_vel_size(ag)
        dist = float(np.linalg.norm(pose[:2] - ego_xy))
        if dist <= cfg.agent_radius_m:
            filtered.append((dist, ag, pose, vel, length, width))
    filtered.sort(key=lambda x: x[0])
    filtered = filtered[: cfg.max_agents]

    track_tokens = [_get_track_token(x[1]) for x in filtered]
    N = len(track_tokens)
    # Prepare arrays
    Tf = cfg.future_num_samples
    D = 7  # x,y,yaw,vx,vy,length,width

    agents_hist = np.zeros((cfg.max_agents, Tp, D), dtype=np.float32)
    agents_hist_mask = np.zeros((cfg.max_agents, Tp), dtype=np.float32)
    agents_curr = np.zeros((cfg.max_agents, D), dtype=np.float32)
    agents_future = np.zeros((cfg.max_agents, Tf, 3), dtype=np.float32)
    agents_future_mask = np.zeros((cfg.max_agents, Tf), dtype=np.float32)

    # Build dicts for past and future frames
    past_tracks = list(scenario.get_past_tracked_objects(iteration, cfg.past_horizon_s, past_num))
    past_tracks.append(detections)
    # Ensure length Tp
    if len(past_tracks) != Tp:
        # If scenario doesn't have enough, pad by repeating last available
        if len(past_tracks) == 0:
            past_tracks = [detections] * Tp
        elif len(past_tracks) < Tp:
            last = past_tracks[-1]
            past_tracks = past_tracks + [last] * (Tp - len(past_tracks))
        else:
            past_tracks = past_tracks[-Tp:]

    future_tracks = []
    if include_future_agents:
        future_tracks = list(scenario.get_future_tracked_objects(iteration, cfg.future_horizon_s, cfg.future_num_samples))
        if len(future_tracks) != Tf:
            if len(future_tracks) == 0:
                future_tracks = [scenario.get_tracked_objects_at_iteration(iteration)] * Tf
            else:
                last = future_tracks[-1]
                future_tracks = future_tracks + [last] * (Tf - len(future_tracks))
                future_tracks = future_tracks[:Tf]

    # Fill past
    for t in range(Tp):
        objs = _get_tracked_objects_list(past_tracks[t])
        token_to_obj = {_get_track_token(o): o for o in objs}
        for i, token in enumerate(track_tokens):
            if i >= cfg.max_agents:
                break
            o = token_to_obj.get(token, None)
            if o is None:
                continue
            pose, vel, length, width = _agent_pose_vel_size(o)
            # to ego-local
            pose_local = global_to_local_pose(pose[None, :], ego_xy, ego_yaw)[0]
            vel_local = global_to_local_xy(vel[None, :], np.zeros(2, dtype=np.float32), ego_yaw)[0]
            feat = np.array([pose_local[0], pose_local[1], pose_local[2], vel_local[0], vel_local[1], length, width], dtype=np.float32)
            agents_hist[i, t] = feat
            agents_hist_mask[i, t] = 1.0

    # Fill current (use t = last past sample)
    if Tp > 0:
        agents_curr[:] = agents_hist[:, -1, :]

    # Fill future
    if include_future_agents:
        for t in range(Tf):
            objs = _get_tracked_objects_list(future_tracks[t])
            token_to_obj = {_get_track_token(o): o for o in objs}
            for i, token in enumerate(track_tokens):
                if i >= cfg.max_agents:
                    break
                o = token_to_obj.get(token, None)
                if o is None:
                    continue
                pose, vel, length, width = _agent_pose_vel_size(o)
                pose_local = global_to_local_pose(pose[None, :], ego_xy, ego_yaw)[0]
                agents_future[i, t] = pose_local
                agents_future_mask[i, t] = 1.0

    # Maneuver labels for agents at this iteration (based on their future)
    maneuver = np.full((cfg.max_agents,), int(Maneuver.KEEP), dtype=np.int64)
    for i in range(N):
        curr_pose = agents_curr[i, :3]
        fut = agents_future[i]  # [Tf,3]
        maneuver[i] = _classify_maneuver(curr_pose, fut, cfg)

    # Ego maneuver label (optional)
    ego_maneuver = np.array([_classify_maneuver(np.array([0.0, 0.0, 0.0], dtype=np.float32), ego_future_local, cfg)], dtype=np.int64)

    # Structure labels at t (using ego+agents)
    K = 1 + cfg.max_agents
    traj_xy = np.zeros((K, Tf + 1, 2), dtype=np.float32)
    valid = np.zeros((K,), dtype=np.float32)
    valid[0] = 1.0
    # ego trajectory starts at origin
    traj_xy[0, 0] = np.array([0.0, 0.0], dtype=np.float32)
    traj_xy[0, 1:] = ego_future_local[:, :2]
    for i in range(cfg.max_agents):
        if i < N and agents_hist_mask[i, -1] > 0.5:
            valid[1 + i] = 1.0
            traj_xy[1 + i, 0] = agents_curr[i, :2]
            if include_future_agents:
                traj_xy[1 + i, 1:] = agents_future[i, :, :2]
    structure_t = _compute_structure_from_futures(traj_xy, valid, cfg)

    # Rule-based structure at t (NO future leakage): constant-velocity rollout from current state.
    # This is intended for online planning / evaluation.
    dt_fut = float(cfg.future_horizon_s) / max(1, int(cfg.future_num_samples))
    horizon_rule = min(3.0, float(cfg.future_horizon_s))
    T_rule = int(round(horizon_rule / max(1e-3, dt_fut))) + 1
    traj_xy_rule = np.zeros((K, T_rule, 2), dtype=np.float32)
    vel_rule = np.zeros((K, 2), dtype=np.float32)
    traj_xy_rule[0, 0] = np.array([0.0, 0.0], dtype=np.float32)
    # ego velocity in ego-local frame (vx,vy)
    ego_vel_local = global_to_local_xy(ego_vec[4:6][None, :], np.zeros(2, dtype=np.float32), ego_yaw)[0]
    vel_rule[0] = ego_vel_local
    for i in range(cfg.max_agents):
        if valid[1 + i] > 0.5:
            traj_xy_rule[1 + i, 0] = agents_curr[i, 0:2]
            vel_rule[1 + i] = agents_curr[i, 3:5]
    for t in range(1, T_rule):
        tt = float(t) * dt_fut
        traj_xy_rule[:, t] = traj_xy_rule[:, 0] + vel_rule * tt
    structure_t_rule, structure_conf_t = _compute_structure_and_confidence_from_futures(traj_xy_rule, valid, cfg)

    # Structure at t + delta
    # nuPlan scenario interval (seconds)
    db_dt = getattr(scenario, "database_interval", None)
    if db_dt is None:
        db_dt = getattr(scenario, "database_row_interval", None)
    if db_dt is None:
        # nuPlan DB logs are typically 20Hz
        db_dt = 0.05
    delta_iter = max(1, int(round(cfg.structure_delta_s / float(db_dt))))
    it2 = min(iteration + delta_iter, scenario.get_number_of_iterations() - 1)
    # Recompute future trajectories at it2 for same agent tokens
    ego_state2 = scenario.get_ego_state_at_iteration(it2)
    ego_vec2 = _ego_state_to_vector(ego_state2)
    ego_xy2 = ego_vec2[1:3].copy()
    ego_yaw2 = float(ego_vec2[3])
    ego_future2 = list(scenario.get_ego_future_trajectory(it2, cfg.future_horizon_s, cfg.future_num_samples))
    ego_future_vec2 = np.stack([_ego_state_to_vector(s) for s in ego_future2], axis=0)
    ego_future_pose2 = np.stack([ego_future_vec2[:, 1], ego_future_vec2[:, 2], ego_future_vec2[:, 3]], axis=-1)
    ego_future_local2 = global_to_local_pose(ego_future_pose2, ego_xy2, ego_yaw2)

    future_tracks2 = list(scenario.get_future_tracked_objects(it2, cfg.future_horizon_s, cfg.future_num_samples))
    if len(future_tracks2) != Tf:
        if len(future_tracks2) == 0:
            future_tracks2 = [scenario.get_tracked_objects_at_iteration(it2)] * Tf
        else:
            last = future_tracks2[-1]
            future_tracks2 = future_tracks2 + [last] * (Tf - len(future_tracks2))
            future_tracks2 = future_tracks2[:Tf]

    # Build trajs in ego2 local, but we need to compare within same frame.
    # For simplicity, compute structure_t1 in the ego2 frame (it is used as label only).
    traj_xy2 = np.zeros((K, Tf + 1, 2), dtype=np.float32)
    valid2 = np.zeros((K,), dtype=np.float32)
    valid2[0] = 1.0
    traj_xy2[0, 0] = np.array([0.0, 0.0], dtype=np.float32)
    traj_xy2[0, 1:] = ego_future_local2[:, :2]

    # current agents at it2
    det2 = scenario.get_tracked_objects_at_iteration(it2)
    objs2 = _get_tracked_objects_list(det2)
    token_to_obj2 = {_get_track_token(o): o for o in objs2}
    token_to_future2: Dict[str, np.ndarray] = {}
    for t in range(Tf):
        objs = _get_tracked_objects_list(future_tracks2[t])
        token_to_obj = {_get_track_token(o): o for o in objs}
        for token, o in token_to_obj.items():
            pose, vel, length, width = _agent_pose_vel_size(o)
            pose_local = global_to_local_pose(pose[None, :], ego_xy2, ego_yaw2)[0]
            if token not in token_to_future2:
                token_to_future2[token] = np.zeros((Tf, 3), dtype=np.float32)
            token_to_future2[token][t] = pose_local

    for i, token in enumerate(track_tokens):
        if i >= cfg.max_agents:
            break
        o = token_to_obj2.get(token, None)
        if o is None:
            continue
        pose, vel, length, width = _agent_pose_vel_size(o)
        pose_local = global_to_local_pose(pose[None, :], ego_xy2, ego_yaw2)[0]
        valid2[1 + i] = 1.0
        traj_xy2[1 + i, 0] = pose_local[:2]
        fut = token_to_future2.get(token, None)
        if fut is not None:
            traj_xy2[1 + i, 1:] = fut[:, :2]

    structure_t1 = _compute_structure_from_futures(traj_xy2, valid2, cfg)

    # Rule-based structure at t+delta (NO future leakage): constant-velocity rollout in ego2 frame.
    traj_xy2_rule = np.zeros((K, T_rule, 2), dtype=np.float32)
    vel2_rule = np.zeros((K, 2), dtype=np.float32)
    traj_xy2_rule[0, 0] = np.array([0.0, 0.0], dtype=np.float32)
    ego_vel_local2 = global_to_local_xy(ego_vec2[4:6][None, :], np.zeros(2, dtype=np.float32), ego_yaw2)[0]
    vel2_rule[0] = ego_vel_local2
    for i, token in enumerate(track_tokens):
        if i >= cfg.max_agents:
            break
        if valid2[1 + i] < 0.5:
            continue
        o = token_to_obj2.get(token, None)
        if o is None:
            continue
        pose, vel, length, width = _agent_pose_vel_size(o)
        pose_local = global_to_local_pose(pose[None, :], ego_xy2, ego_yaw2)[0]
        vel_local = global_to_local_xy(vel[None, :], np.zeros(2, dtype=np.float32), ego_yaw2)[0]
        traj_xy2_rule[1 + i, 0] = pose_local[:2]
        vel2_rule[1 + i] = vel_local
    for t in range(1, T_rule):
        tt = float(t) * dt_fut
        traj_xy2_rule[:, t] = traj_xy2_rule[:, 0] + vel2_rule * tt
    structure_t1_rule, structure_conf_t1 = _compute_structure_and_confidence_from_futures(traj_xy2_rule, valid2, cfg)

    # Map features
    tl_statuses = scenario.get_traffic_light_status_at_iteration(iteration)
    tl_dict = _traffic_light_dict(tl_statuses)
    route_ids = scenario.get_route_roadblock_ids() if cfg.include_route else None
    map_feat = _extract_map_polylines(scenario.map_api, ego_xy, ego_yaw, route_ids, tl_dict, cfg)

    # Ego dynamics (vx,vy,ax,ay) rotated into the same ego-local frame as positions.
    # nuPlan EgoState exposes dynamics in the global frame; rotate by current ego_yaw.
    ego_vel_hist_local = global_to_local_xy(ego_past_vec[:, 4:6], np.zeros(2, dtype=np.float32), ego_yaw)
    ego_acc_hist_local = global_to_local_xy(ego_past_vec[:, 6:8], np.zeros(2, dtype=np.float32), ego_yaw)
    ego_dyn_hist_local = np.concatenate([ego_vel_hist_local, ego_acc_hist_local], axis=-1)

    ego_vel_fut_local = global_to_local_xy(ego_future_vec[:, 4:6], np.zeros(2, dtype=np.float32), ego_yaw)
    ego_acc_fut_local = global_to_local_xy(ego_future_vec[:, 6:8], np.zeros(2, dtype=np.float32), ego_yaw)
    ego_dyn_fut_local = np.concatenate([ego_vel_fut_local, ego_acc_fut_local], axis=-1)

    out: Dict[str, np.ndarray] = {
        # Ego
        "ego_hist": ego_past_local.astype(np.float32),   # [Tp, 3]
        "ego_future": ego_future_local.astype(np.float32),  # [Tf, 3]
        "ego_dyn_hist": ego_dyn_hist_local.astype(np.float32),  # [Tp, 4] vx,vy,ax,ay (ego-local)
        "ego_dyn_future": ego_dyn_fut_local.astype(np.float32),  # [Tf, 4] (ego-local)
        "ego_maneuver": ego_maneuver,  # [1]
        # Agents
        "agents_hist": agents_hist,  # [Nmax, Tp, D]
        "agents_hist_mask": agents_hist_mask,  # [Nmax, Tp]
        "agents_future": agents_future,  # [Nmax, Tf, 3]
        "agents_future_mask": agents_future_mask,  # [Nmax, Tf]
        "agents_maneuver": maneuver,  # [Nmax]
        # Structure labels
        "structure_t": structure_t,  # [1+Nmax,1+Nmax]
        "structure_t1": structure_t1,
        "structure_t_rule": structure_t_rule,
        "structure_t1_rule": structure_t1_rule,
        "structure_conf_t": np.array([float(structure_conf_t)], dtype=np.float32),
        "structure_conf_t1": np.array([float(structure_conf_t1)], dtype=np.float32),
        # Map
        "map_polylines": map_feat["map_polylines"].astype(np.float32),
        "map_poly_mask": map_feat["map_poly_mask"].astype(np.float32),
        "map_poly_type": np.array(map_feat["map_poly_type"], dtype=np.int64),
        "map_tl_status": np.array(map_feat["map_tl_status"], dtype=np.int64),
        "map_on_route": np.array(map_feat["map_on_route"], dtype=np.int64),
    }
    return out
