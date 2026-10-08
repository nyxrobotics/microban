# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from constants import (
    VTHETA_MAX_MOVING,
    VTHETA_MAX_STATIONARY,
    VX_MAX,
    VX_MAX_BACKWARD,
    VY_MAX,
)


@dataclass
class UserInput:
    """Human or agent control state for one scheduler iteration."""

    active_moves: set[str] = field(default_factory=set)
    velocity: dict[str, float] = field(default_factory=lambda: {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})
    show_imu: bool = False

    # Real-hardware master gate.  For a power-controlling input source (the
    # PICO/network bridge, or the local gamepad), B makes torque_enabled false;
    # A makes torque_enabled true while leaving policy_enabled false so the
    # scheduler slowly returns every joint to NEUTRAL_POSE; R3 toggles
    # policy_enabled.  The scheduler, not an individual learned move, owns this
    # gate so head, arms and legs cannot fight the neutral/limp state.
    #
    # Defaults preserve keyboard/simulator behaviour. None means keep the
    # current hardware gate, used when network input is unavailable.
    torque_enabled: bool | None = True
    policy_enabled: bool | None = True

    # Manual get-up-policy testing remains separate from the PICO hardware
    # gate.  It is intentionally not accepted over the PICO/network protocol.
    getup_armed: bool = False

    # Desired camera orientation in radians. Roll/pitch are gravity-aligned; yaw is
    # relative to the trunk (the IMU has no stable absolute-yaw reference). None means
    # "level and forward". VR teleop sends all three axes.
    head_orientation: dict[str, float] | None = None


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

    # Sources which expose an explicit B/A/R3 motor-power state set this true.
    # main.py then starts the real robot limp and enables Scheduler's global
    # hardware gate.  Other sources retain the established startup behaviour.
    controls_motor_power: bool = False

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
