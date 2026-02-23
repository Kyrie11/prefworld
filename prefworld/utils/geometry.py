from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rotation_matrix(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def global_to_local_xy(
    xy: np.ndarray, origin_xy: np.ndarray, origin_yaw: float
) -> np.ndarray:
    """Transform global xy points into a local frame at origin pose."""
    assert xy.shape[-1] == 2
    R = rotation_matrix(-origin_yaw)
    return (xy - origin_xy) @ R.T


def local_to_global_xy(
    xy_local: np.ndarray, origin_xy: np.ndarray, origin_yaw: float
) -> np.ndarray:
    """Transform local xy points into global frame."""
    assert xy_local.shape[-1] == 2
    R = rotation_matrix(origin_yaw)
    return xy_local @ R.T + origin_xy


def global_to_local_pose(
    pose: np.ndarray, origin_xy: np.ndarray, origin_yaw: float
) -> np.ndarray:
    """Transform pose [x,y,yaw] in global to local."""
    assert pose.shape[-1] == 3
    xy_local = global_to_local_xy(pose[..., :2], origin_xy, origin_yaw)
    yaw_local = np.vectorize(wrap_angle)(pose[..., 2] - origin_yaw).astype(np.float32)
    return np.concatenate([xy_local, yaw_local[..., None]], axis=-1)


def pose_to_xy_yaw(pose_like) -> Tuple[float, float, float]:
    """Extract (x,y,yaw) from nuPlan StateSE2 / EgoState etc."""
    # nuPlan uses StateSE2 with attributes x,y,heading
    if hasattr(pose_like, "x") and hasattr(pose_like, "y") and hasattr(pose_like, "heading"):
        return float(pose_like.x), float(pose_like.y), float(pose_like.heading)
    # EgoState rear axle pose
    if hasattr(pose_like, "rear_axle"):
        ra = pose_like.rear_axle
        return float(ra.x), float(ra.y), float(ra.heading)
    raise TypeError(f"Unsupported pose-like type: {type(pose_like)}")
