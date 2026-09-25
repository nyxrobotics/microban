# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Robot-local direct-arm contract for PICO controller teleoperation.

This is the expanded joint envelope used by the direct six-joint overlay for
the absolute HMD-origin controller mapping.  It is deliberately independent of
the narrower hand-target FK box embedded in the learned contract-v12 policy.
Keeping this contract on the robot means a wire sender cannot widen the arm
range by supplying different limits or a different home pose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

PICO_ARM_TARGET_CONTRACT_REVISION = (
    "microban_hmd_absolute_arm_fk_live_box_pitch100_roll120_elbow110_v1"
)
PICO_ARM_SIDES = ("left", "right")
PICO_ARM_JOINT_NAMES = {
    "left": ("left_shoulder_pitch", "left_shoulder_roll", "left_elbow"),
    "right": ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow"),
}
PICO_ARM_LOWER_RAD = {
    "left": tuple(math.radians(value) for value in (-100.0, 10.0, -110.0)),
    "right": tuple(math.radians(value) for value in (-100.0, -120.0, -110.0)),
}
PICO_ARM_UPPER_RAD = {
    "left": tuple(math.radians(value) for value in (100.0, 120.0, 0.0)),
    "right": tuple(math.radians(value) for value in (100.0, -10.0, 0.0)),
}
PICO_ARM_HOME_RAD = {
    "left": tuple(math.radians(value) for value in (0.0, 10.0, -20.0)),
    "right": tuple(math.radians(value) for value in (0.0, -10.0, -20.0)),
}
_BOUND_EPSILON_RAD = 1.0e-12


def parse_pico_arm_joint_target(
    value: Any,
) -> dict[str, tuple[float, float, float]]:
    """Parse one exact paired target inside the compiled-in reachable box.

    The function raises for every malformed/out-of-contract value.  Callers at
    the network boundary catch that failure and replace the complete arm
    snapshot rather than retaining any target from an earlier datagram.
    """

    if not isinstance(value, Mapping) or set(value) != set(PICO_ARM_SIDES):
        raise TypeError("arm_joint_target must contain exactly left and right")

    parsed: dict[str, tuple[float, float, float]] = {}
    for side in PICO_ARM_SIDES:
        vector = value[side]
        if (
            not isinstance(vector, Sequence)
            or isinstance(vector, (str, bytes, bytearray))
            or len(vector) != 3
        ):
            raise TypeError(f"arm_joint_target.{side} must contain three numbers")
        numbers: list[float] = []
        for index, item in enumerate(vector):
            if isinstance(item, bool):
                raise TypeError(
                    f"arm_joint_target.{side}[{index}] must be a number"
                )
            try:
                number = float(item)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(
                    f"arm_joint_target.{side}[{index}] must be a number"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(
                    f"arm_joint_target.{side}[{index}] must be finite"
                )
            lower = PICO_ARM_LOWER_RAD[side][index]
            upper = PICO_ARM_UPPER_RAD[side][index]
            if (
                number < lower - _BOUND_EPSILON_RAD
                or number > upper + _BOUND_EPSILON_RAD
            ):
                raise ValueError(
                    f"arm_joint_target.{side}[{index}] is outside the "
                    "authenticated direct-arm joint box"
                )
            # Only absorb representation roundoff at an inclusive endpoint;
            # values materially outside the box were rejected above.
            numbers.append(max(lower, min(upper, number)))
        parsed[side] = (numbers[0], numbers[1], numbers[2])
    return parsed
