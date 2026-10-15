# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from observer import Observation
from constants import NEUTRAL_POSE


class MoveState(Enum):
    INACTIVE = auto()
    STARTING = auto()
    ACTIVE = auto()
    STOPPING = auto()


@dataclass
class MotorCommand:
    """
    Final motor command built by the scheduler pipeline.
    
    Initialized with the neutral pose.
    """

    target_angles: dict[str, float] = field(default_factory=lambda: dict(NEUTRAL_POSE))


@dataclass
class WholebodyClaim:
    """Set while whole-body tracking drives the robot (moves/wholebody.py).

    Its policy drives the arms and its contract drives the neck to the
    reference's neck angles, so the arm overlay and the HMD head move yield.
    """

    active: bool = False


class Move(ABC):
    """Base class for all motion behaviors."""

    def __init__(self) -> None:
        self.state: MoveState = MoveState.INACTIVE

    def preload(self) -> None:
        """Called before the control loop starts. Override to load heavy resources."""

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        """Called each tick while state is STARTING.
        Must set self.state = MoveState.ACTIVE when the transition is done."""
        self.state = MoveState.ACTIVE

    @abstractmethod
    def step(self, obs: Observation, command: MotorCommand) -> None:
        """Called each tick while state is ACTIVE."""

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        """Called each tick while state is STOPPING.
        Must set self.state = MoveState.INACTIVE when the transition is done."""
        self.state = MoveState.INACTIVE

    def can_balance(self, user_input) -> bool:
        """Whether this move, started now for ``user_input``, keeps a standing
        robot upright by active feedback (a learned standing/walking actor).

        The scheduler keeps GetupMove on as the standing balancer after a
        get-up until the requested "walk" move answers True here. Conservative
        default: a static hold or scripted motion cannot balance.
        """
        _ = user_input
        return False

    def follows_fallen_reference(self, now_s: float, projected_gravity) -> bool:
        """Whether a trunk tilted past the fall angle is following a reference
        whose own trunk is tilted that far (whole-body tracking).

        The scheduler does not count such a tilt as a fall.  Default: never.
        """
        _ = now_s, projected_gravity
        return False

    def on_safety_resume(self, obs: Observation) -> None:
        """Resynchronize private transition state after a scheduler safety hold.

        A safety hold deliberately skips normal move callbacks. Stateful moves may
        override this to discard stale interpolation/policy history before dispatch
        resumes; the lifecycle state itself is preserved.
        """
        _ = obs
