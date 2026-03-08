from __future__ import annotations

import enum
from typing import Tuple


class Maneuver(enum.IntEnum):
    """Legacy coarse maneuver labels kept for backward compatibility.

    The paper's current discrete action is no longer a single maneuver id; it is a
    pair ``(reference-path branch, longitudinal constraint source)``.  We still keep
    these six coarse labels because the rest of the repo (metrics, planner fallbacks,
    cache labels) already uses them.
    """

    KEEP = 0
    LANE_CHANGE_LEFT = 1
    LANE_CHANGE_RIGHT = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    STOP = 5


class PathType(enum.IntEnum):
    """Discrete reference-path branch in the paper's action space."""

    KEEP = 0
    LANE_CHANGE_LEFT = 1
    LANE_CHANGE_RIGHT = 2
    BRANCH_STRAIGHT = 3
    BRANCH_LEFT = 4
    BRANCH_RIGHT = 5


class LonConstraint(enum.IntEnum):
    """Longitudinal constraint source in the paper's action space."""

    FREE_FLOW = 0
    FOLLOW = 1
    STOP_LINE = 2
    YIELD_TO = 3


MANEUVER_NAMES = [m.name for m in Maneuver]
PATH_TYPE_NAMES = [m.name for m in PathType]
LON_CONSTRAINT_NAMES = [m.name for m in LonConstraint]

NUM_MANEUVERS = len(Maneuver)
NUM_PATH_TYPES = len(PathType)
NUM_LON_CONSTRAINTS = len(LonConstraint)


def path_constraint_to_maneuver(path_type: int, lon_constraint: int) -> int:
    """Map the paper action tuple back to the legacy coarse maneuver family.

    This is intentionally many-to-one:
      - StopLine always maps to STOP.
      - Lane-change paths map to LCL/LCR.
      - Branch left/right map to turn left/right.
      - Keep/straight branches with free-flow/follow/yield map to KEEP.
    """

    p = int(path_type)
    c = int(lon_constraint)
    if c == int(LonConstraint.STOP_LINE):
        return int(Maneuver.STOP)
    if p == int(PathType.LANE_CHANGE_LEFT):
        return int(Maneuver.LANE_CHANGE_LEFT)
    if p == int(PathType.LANE_CHANGE_RIGHT):
        return int(Maneuver.LANE_CHANGE_RIGHT)
    if p == int(PathType.BRANCH_LEFT):
        return int(Maneuver.TURN_LEFT)
    if p == int(PathType.BRANCH_RIGHT):
        return int(Maneuver.TURN_RIGHT)
    return int(Maneuver.KEEP)


def maneuver_to_canonical_action(maneuver: int) -> Tuple[int, int]:
    """Canonical paper-style action used when only a legacy maneuver is available."""

    m = int(maneuver)
    if m == int(Maneuver.LANE_CHANGE_LEFT):
        return int(PathType.LANE_CHANGE_LEFT), int(LonConstraint.FREE_FLOW)
    if m == int(Maneuver.LANE_CHANGE_RIGHT):
        return int(PathType.LANE_CHANGE_RIGHT), int(LonConstraint.FREE_FLOW)
    if m == int(Maneuver.TURN_LEFT):
        return int(PathType.BRANCH_LEFT), int(LonConstraint.FREE_FLOW)
    if m == int(Maneuver.TURN_RIGHT):
        return int(PathType.BRANCH_RIGHT), int(LonConstraint.FREE_FLOW)
    if m == int(Maneuver.STOP):
        return int(PathType.KEEP), int(LonConstraint.STOP_LINE)
    return int(PathType.KEEP), int(LonConstraint.FREE_FLOW)
