# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Independent six-joint PICO controller arm overlay."""

from __future__ import annotations

import math

from constants import KP_DEFAULT, KP_RL, MOTOR_TO_ID, NEUTRAL_POSE
from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from observer import Observation
from pico_arm_contract import (
    PICO_ARM_HOME_RAD,
    PICO_ARM_JOINT_NAMES,
    PICO_ARM_SIDES,
    parse_pico_arm_joint_target,
)

PICO_ARM_JOINT_ORDER = tuple(
    name for side in PICO_ARM_SIDES for name in PICO_ARM_JOINT_NAMES[side]
)


class PicoArmTrackingMove(Move):
    """Overlay bounded controller IK targets after the locomotion actor.

    The move deliberately owns only both shoulder pitch/roll and elbow joints.
    Registering it immediately after ``walk`` lets the learned policy continue
    to own all leg joints in both standing and walking states.  A live released
    right trigger keeps this move active and slews to the exact PICO FK home;
    losing the PICO session stops the move and returns to the robot's global
    neutral pose.
    """

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        *,
        slew_rate_rad_s: float = 4.0,
        neutral_return_duration_s: float = 0.8,
    ) -> None:
        super().__init__()
        if not math.isfinite(slew_rate_rad_s) or slew_rate_rad_s <= 0.0:
            raise ValueError("slew_rate_rad_s must be finite and positive")
        if (
            not math.isfinite(neutral_return_duration_s)
            or neutral_return_duration_s <= 0.0
        ):
            raise ValueError(
                "neutral_return_duration_s must be finite and positive"
            )
        self._controller = controller
        self._slew_rate_rad_s = float(slew_rate_rad_s)
        self._neutral_return_duration_s = float(neutral_return_duration_s)
        self._last_targets: dict[str, float] = {}
        self._last_time_s: float | None = None
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}
        self._suspended_for_getup = False

    @staticmethod
    def _pico_home() -> dict[str, float]:
        return {
            name: PICO_ARM_HOME_RAD[side][index]
            for side in PICO_ARM_SIDES
            for index, name in enumerate(PICO_ARM_JOINT_NAMES[side])
        }

    @staticmethod
    def _global_neutral() -> dict[str, float]:
        return {name: float(NEUTRAL_POSE[name]) for name in PICO_ARM_JOINT_ORDER}

    def _initialize_from_observation(self, obs: Observation) -> None:
        previous = self._last_targets
        targets: dict[str, float] = {}
        for name in PICO_ARM_JOINT_ORDER:
            try:
                measured = float(
                    obs.robot_state.motor_positions.get(
                        name, previous.get(name, NEUTRAL_POSE[name])
                    )
                )
            except (TypeError, ValueError, OverflowError):
                measured = math.nan
            targets[name] = (
                measured
                if math.isfinite(measured)
                else previous.get(name, float(NEUTRAL_POSE[name]))
            )
        self._last_targets = targets
        try:
            now = float(obs.robot_state.time_s)
        except (TypeError, ValueError, OverflowError):
            now = math.nan
        if math.isfinite(now):
            self._last_time_s = now

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        self._initialize_from_observation(obs)
        if self._controller is not None:
            ids = [MOTOR_TO_ID[name] for name in PICO_ARM_JOINT_ORDER]
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        command.target_angles.update(self._last_targets)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self._suspended_for_getup = False
        self.state = MoveState.ACTIVE

    def _desired_live_target(self, obs: Observation) -> dict[str, float]:
        if not obs.user_input.arm_tracking_enabled:
            return self._pico_home()
        try:
            parsed = parse_pico_arm_joint_target(obs.user_input.arm_joint_target)
        except (TypeError, ValueError, OverflowError):
            # A custom InputSource cannot bypass the receiver's all-or-nothing
            # validation and leave an earlier arm target latched in this move.
            return self._pico_home()
        return {
            name: parsed[side][index]
            for side in PICO_ARM_SIDES
            for index, name in enumerate(PICO_ARM_JOINT_NAMES[side])
        }

    def _write_slew_limited(
        self,
        obs: Observation,
        command: MotorCommand,
        desired: dict[str, float],
    ) -> None:
        if not self._last_targets:
            self._initialize_from_observation(obs)
        try:
            now = float(obs.robot_state.time_s)
        except (TypeError, ValueError, OverflowError):
            now = math.nan
        if not math.isfinite(now):
            now = self._last_time_s if self._last_time_s is not None else 0.0
        previous_time = self._last_time_s if self._last_time_s is not None else now
        dt = max(0.001, min(0.1, now - previous_time))
        max_step = self._slew_rate_rad_s * dt

        for name in PICO_ARM_JOINT_ORDER:
            target = float(desired[name])
            previous = self._last_targets[name]
            difference = target - previous
            current = (
                target
                if abs(difference) <= max_step
                else previous + math.copysign(max_step, difference)
            )
            self._last_targets[name] = current
            command.target_angles[name] = current
        self._last_time_s = now

    def _handle_getup_ownership(self, obs: Observation) -> bool:
        if "getup" in obs.user_input.active_moves:
            self._suspended_for_getup = True
            return True
        if self._suspended_for_getup:
            self._initialize_from_observation(obs)
            self._suspended_for_getup = False
        return False

    def step(self, obs: Observation, command: MotorCommand) -> None:
        if self._handle_getup_ownership(obs):
            return
        self._write_slew_limited(obs, command, self._desired_live_target(obs))

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if "getup" in obs.user_input.active_moves:
            # Get-up is registered after this overlay and owns all 18 body
            # joints.  Do not change its commands or gains during the hand-off.
            self._last_targets = {}
            self._last_time_s = None
            self._stop_start_time_s = None
            self._stop_start_angles = {}
            self._suspended_for_getup = False
            self.state = MoveState.INACTIVE
            return

        desired = self._global_neutral()
        try:
            now = float(obs.robot_state.time_s)
        except (TypeError, ValueError, OverflowError):
            now = math.nan
        if not math.isfinite(now):
            now = self._stop_start_time_s if self._stop_start_time_s is not None else 0.0
        if self._stop_start_time_s is None:
            self._initialize_from_observation(obs)
            self._stop_start_time_s = now
            self._stop_start_angles = dict(self._last_targets)
        elapsed = max(0.0, now - self._stop_start_time_s)
        fraction = min(1.0, elapsed / self._neutral_return_duration_s)
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        for name in PICO_ARM_JOINT_ORDER:
            start = self._stop_start_angles[name]
            target = (
                desired[name]
                if fraction >= 1.0
                else start + (desired[name] - start) * blend
            )
            self._last_targets[name] = target
            command.target_angles[name] = target
        self._last_time_s = now

        if fraction >= 1.0:
            if self._controller is not None:
                ids = [MOTOR_TO_ID[name] for name in PICO_ARM_JOINT_ORDER]
                gain = KP_RL if "walk" in obs.user_input.active_moves else KP_DEFAULT
                self._controller.sync_write_kp(ids, [gain] * len(ids))
            self._last_targets = {}
            self._last_time_s = None
            self._stop_start_time_s = None
            self._stop_start_angles = {}
            self._suspended_for_getup = False
            self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        # The scheduler held measured positions while callbacks were skipped.
        # Discard every pre-fault target/time pair before normal STOPPING runs.
        if self.state != MoveState.INACTIVE:
            self._initialize_from_observation(obs)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self._suspended_for_getup = False
