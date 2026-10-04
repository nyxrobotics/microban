# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Checks against the shared training HOME (constants.NEUTRAL_POSE + root pose).

Two uses: verifying a policy's full-precision HOME stamp (walking ``home_pose``,
get-up ``microban_getup_home_pose``; both are mjlab_microban's
``getup_home_pose()`` JSON), and measuring trunk tilt relative to the HOME
attitude (the trunk leans HOME_TRUNK_PITCH_RAD forward at HOME, so "standing as
trained" means projected gravity near HOME_PROJECTED_GRAVITY, not (0, 0, -1)).
"""

import json
import math
from collections.abc import Sequence

from constants import (
    HOME_PROJECTED_GRAVITY,
    HOME_ROOT_POS_Z_M,
    HOME_ROOT_QUAT_WXYZ,
    NEUTRAL_POSE,
)

HOME_POSE_TOLERANCE_RAD = 1.0e-6
HOME_ROOT_TOLERANCE = 1.0e-6


def home_pose_stamp_matches(value: str | None) -> bool:
    """True when a policy's HOME stamp is the robot's HOME.

    The stamp is ``{"joint_pos_rad": {21 joints}, "root_pos_m": [x, y, z],
    "root_quat_wxyz": [w, x, y, z]}``. The joints must be NEUTRAL_POSE and the
    root (0, 0, HOME_ROOT_POS_Z_M) with HOME_ROOT_QUAT_WXYZ, so an upright-HOME
    model is refused even if its joint stamp were somehow relabelled.
    """
    if not value:
        return False
    try:
        home = json.loads(value)
        joints = {str(name): float(angle) for name, angle in home["joint_pos_rad"].items()}
        root_pos = [float(item) for item in home["root_pos_m"]]
        root_quat = [float(item) for item in home["root_quat_wxyz"]]
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
        return False
    expected_root = (0.0, 0.0, HOME_ROOT_POS_Z_M)
    return (
        set(joints) == set(NEUTRAL_POSE)
        and all(
            math.isfinite(angle)
            and abs(angle - NEUTRAL_POSE[name]) <= HOME_POSE_TOLERANCE_RAD
            for name, angle in joints.items()
        )
        and len(root_pos) == 3
        and all(
            math.isfinite(actual) and abs(actual - expected) <= HOME_ROOT_TOLERANCE
            for actual, expected in zip(root_pos, expected_root)
        )
        and len(root_quat) == 4
        and all(
            math.isfinite(actual) and abs(actual - expected) <= HOME_ROOT_TOLERANCE
            for actual, expected in zip(root_quat, HOME_ROOT_QUAT_WXYZ)
        )
    )


def tilt_from_home_rad(projected_gravity: Sequence[float]) -> float | None:
    """Angle between the measured projected gravity and HOME's, in radians.

    0 at the HOME attitude (trunk 10 deg forward), 10 deg when the trunk is
    vertical. None for a non-finite, wrong-length or near-zero vector.
    """
    if len(projected_gravity) != 3:
        return None
    try:
        values = [float(item) for item in projected_gravity]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in values):
        return None
    norm = math.sqrt(sum(item * item for item in values))
    if norm < 1.0e-6:
        return None
    cosine = sum(a * b for a, b in zip(values, HOME_PROJECTED_GRAVITY)) / norm
    return math.acos(max(-1.0, min(1.0, cosine)))
