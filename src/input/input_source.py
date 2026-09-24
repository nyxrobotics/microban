# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math

from constants import VX_MAX, VX_MAX_BACKWARD, VY_MAX, VTHETA_MAX_STATIONARY, VTHETA_MAX_MOVING


@dataclass
class UserInput:
    """Human or agent control state for one scheduler iteration."""

    active_moves: set[str] = field(default_factory=set)
    velocity: dict[str, float] = field(default_factory=lambda: {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})
    show_imu: bool = False

    # Low-level locomotion policy selected by the PICO controller.  The network
    # receiver accepts only these named modes and changes mode only while the
    # walking deadman is released.  Keyboard/gamepad operation stays on the
    # established walking policy.
    locomotion_policy: str = "walk"

    # Desired camera orientation in radians. Roll/pitch are gravity-aligned; yaw is
    # relative to the trunk (the IMU has no stable absolute-yaw reference). None means
    # "level and forward". VR teleop sends all three axes.
    head_orientation: dict[str, float] | None = None

    # Hold the physical head-yaw joint at the trunk's forward direction. Kept as an
    # explicit command instead of rewriting head_orientation["yaw"] at the PC so the
    # robot-side safety/slew limiter owns every discontinuity.
    head_yaw_front: bool = False

    # Whole-body leg tracking target, trunk-relative (meters): {"left": (dx,dy,dz),
    # "right": (dx,dy,dz)}. None means no target — the walking policy just walks/stands
    # (see FootTargetCommand in mjlab_microban: the tracking reward fades to zero as the
    # velocity command grows, so this and `velocity` are never in real conflict).
    foot_target: dict[str, tuple[float, float, float]] | None = None

    # Hand tracking target, trunk-relative (meters), per hand independently:
    # {"left": (dx,dy,dz) | None, "right": (dx,dy,dz) | None}. A hand entry of None (or
    # the whole dict being None) means that hand has no active target — the policy is
    # free to move that arm naturally (see HandTargetCommand.is_active).
    hand_target: dict[str, tuple[float, float, float] | None] | None = None


def scale_velocity(velocity: dict[str, float]) -> dict[str, float]:
    """Map a normalized velocity command in [-1, 1] per axis to physical limits.

    Applied centrally (in the scheduler) so the limits are identical for every input
    source — keyboard, gamepad, or agent. Forward and backward have different caps, and
    rotation gets a wider range when turning in place (vx = vy = 0) than while translating.
    """
    def finite_unit(name: str) -> float:
        value = float(velocity.get(name, 0.0))
        return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0

    vx = finite_unit("vx")
    vy = finite_unit("vy")
    vtheta = finite_unit("vtheta")

    moving = abs(vx) > 1e-6 or abs(vy) > 1e-6
    vtheta_max = VTHETA_MAX_MOVING if moving else VTHETA_MAX_STATIONARY
    vx_max = VX_MAX if vx >= 0.0 else VX_MAX_BACKWARD

    return {"vx": vx * vx_max, "vy": vy * VY_MAX, "vtheta": vtheta * vtheta_max}


class InputSource(ABC):
    """Abstract interface for human or agent input. Swap keyboard for gamepad without touching the rest."""

    def start(self) -> None:
        """Start the input source (e.g., launch a background thread)."""

    def stop(self) -> None:
        """Stop the input source and release resources."""

    def set_motion_inhibited(self, inhibited: bool) -> None:
        """Apply a robot-side safety interlock to motion-producing input.

        Sources without a latched deadman may leave this as a no-op. Network input
        overrides it so an IMU/fall safety event requires a fresh trigger release
        after the interlock is removed.
        """
        _ = inhibited

    @abstractmethod
    def read(self) -> UserInput:
        """Return the current input state. Must be non-blocking."""
