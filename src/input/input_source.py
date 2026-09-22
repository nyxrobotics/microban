# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from constants import VX_MAX, VX_MAX_BACKWARD, VY_MAX, VTHETA_MAX_STATIONARY, VTHETA_MAX_MOVING


@dataclass
class UserInput:
    """Human or agent control state for one scheduler iteration."""

    active_moves: set[str] = field(default_factory=set)
    velocity: dict[str, float] = field(default_factory=lambda: {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})
    show_imu: bool = False

    # Desired head orientation {"roll": , "pitch": } in the same gravity-aligned frame as
    # the trunk IMU (radians). None (the default, from every input source except VR teleop)
    # means "keep the head level" — see moves.walk.WalkMove's neck stabilization.
    head_orientation: dict[str, float] | None = None

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
    vx = max(-1.0, min(1.0, velocity.get("vx", 0.0)))
    vy = max(-1.0, min(1.0, velocity.get("vy", 0.0)))
    vtheta = max(-1.0, min(1.0, velocity.get("vtheta", 0.0)))

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

    @abstractmethod
    def read(self) -> UserInput:
        """Return the current input state. Must be non-blocking."""