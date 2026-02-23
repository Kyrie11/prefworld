from __future__ import annotations

import enum


class Maneuver(enum.IntEnum):
    """Discrete intention / maneuver labels."""

    KEEP = 0
    LANE_CHANGE_LEFT = 1
    LANE_CHANGE_RIGHT = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    STOP = 5


MANEUVER_NAMES = [m.name for m in Maneuver]
NUM_MANEUVERS = len(Maneuver)
