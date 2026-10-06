# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Runtime for the Microban PICO hybrid teleoperation policy.

The PICO policy observes all 21 encoders, the locomotion command and the PICO
foot/hand targets (83 values) and drives the 18 body joints with the raw-action
rule every Microban policy shares: target = clip(HOME + raw, -pi, +pi), the
raw output fed back as the previous action.  Its contract (src/policy_contract.py)
is checked before any torque is used; an unusable output raises
``PicoHybridPolicyRuntimeError`` before any target is written so the selector
can hold the body in the same control cycle.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from constants import (
    KP_HARDWARE_NEUTRAL,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    SERVO_TARGET_RANGE_RAD,
)
from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from observer import Observation
from policy_contract import (
    ACTION_WIDTH,
    PolicyContractError,
    load_installed_policy,
    parse_policy,
)

AGENT_NAME = "pico_teleop.onnx"
OBSERVATION_WIDTH = 83
# Live simultaneous-both-feet targets must stay inside this fraction of the
# (conservative, stationary) training support.
LIVE_BODY_TARGET_SAFETY_MARGIN = 0.8
SUPPORT_FOOT_FLOOR_BAND_M = 0.0025
ALL_MOTOR_IDS = list(MOTOR_TO_ID.values())

PicoHybridPolicyContractError = PolicyContractError


class PicoHybridPolicyRuntimeError(RuntimeError):
    """Inference produced a result that is unsafe to send to actuators."""


def _bounded_targets(
    values: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> list[float]:
    if len(values) != len(lower) or len(lower) != len(upper):
        raise PicoHybridPolicyRuntimeError("body target has the wrong width")
    result: list[float] = []
    for value, lo, hi in zip(values, lower, upper):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PicoHybridPolicyRuntimeError(
                "body target contains a non-finite value"
            )
        # Clipping at the policy's training support is safer than extrapolating a
        # human-size or corrupt target into an unseen command.
        result.append(max(lo, min(hi, numeric)))
    return result


def _raw_imu_gyro(gyro_sensor_xyz: Sequence[float]) -> Sequence[float]:
    """The policy observes the IMU-site axes, like the training observation."""

    return gyro_sensor_xyz


class PicoHybridMove(Move):
    """Run the 83-observation PICO policy.

    The ``hmd_head`` move continues to own the head/neck targets.  This move
    owns exactly ``OBSERVATION_DOF_ORDER`` and uses the same smooth return and
    get-up hand-off behavior as ``WalkMove``.
    """

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        policy_path: str | Path = Path("src/agents") / AGENT_NAME,
        neutral_return_duration_s: float = 0.8,
        *,
        session: Any | None = None,
        gyro_transform: Callable[[Sequence[float]], Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._neutral_return_duration_s = neutral_return_duration_s
        if session is None:
            loaded = load_installed_policy("pico", policy_path)
            session, contract = loaded.session, loaded.contract
            self._self_test_rows = loaded.self_test_rows
        else:
            # An injected session (tests, tools) is checked but not self-tested
            # here; load_installed_policy is the deployment path.
            contract = parse_policy("pico", session)
            self._self_test_rows = 0
        assert contract.pico is not None
        self._session = session
        self._contract = contract
        self._targets = contract.pico
        self._gyro_transform = gyro_transform or _raw_imu_gyro
        self._observation_defaults = tuple(
            float(NEUTRAL_POSE[name]) for name in contract.observation_joint_names
        )
        self._action_defaults = tuple(
            float(NEUTRAL_POSE[name]) for name in OBSERVATION_DOF_ORDER
        )
        self._last_action = np.zeros(ACTION_WIDTH, dtype=np.float32)
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}

    @property
    def contract(self):
        return self._contract

    @property
    def self_test_rows(self) -> int:
        return self._self_test_rows

    def can_balance(self, user_input) -> bool:
        # Only constructed once its contract validated and the self-test passed.
        _ = user_input
        return True

    @staticmethod
    def _physical_target(value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PicoHybridPolicyRuntimeError(
                "physical motor target became non-finite"
            )
        return numeric

    def _measured_action_positions(self, obs: Observation) -> dict[str, float]:
        """Return a complete finite measured pose before any command is written."""

        positions: dict[str, float] = {}
        for name in OBSERVATION_DOF_ORDER:
            try:
                value = float(
                    obs.robot_state.motor_positions.get(name, NEUTRAL_POSE[name])
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise PicoHybridPolicyRuntimeError(
                    f"motor position for {name} is not numeric"
                ) from exc
            if not math.isfinite(value):
                raise PicoHybridPolicyRuntimeError(
                    f"motor position for {name} is non-finite"
                )
            positions[name] = value
        return positions

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        measured_positions = self._measured_action_positions(obs)
        if self._controller is not None:
            # A learned policy runs every joint, head and neck included, at the
            # gain it was trained with.
            self._controller.sync_write_kp(ALL_MOTOR_IDS, [KP_RL] * len(ALL_MOTOR_IDS))
        for name in OBSERVATION_DOF_ORDER:
            command.target_angles[name] = self._physical_target(measured_positions[name])
        self._last_action.fill(0.0)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self.state = MoveState.ACTIVE

    def _body_targets(self, obs: Observation) -> tuple[list[float], list[float]]:
        foot_mapping: Mapping[str, Sequence[float]] = obs.user_input.foot_target or {}
        foot_values: list[float] = []
        for side in ("left", "right"):
            foot_values.extend(foot_mapping.get(side, (0.0, 0.0, 0.0)))
        for start in (0, 3):
            vector = tuple(float(value) for value in foot_values[start : start + 3])
            if vector[2] <= SUPPORT_FOOT_FLOOR_BAND_M and vector != (0.0, 0.0, 0.0):
                raise PicoHybridPolicyRuntimeError(
                    "support-foot floor-band target must be exact XYZ zero"
                )
        active_feet = [
            any(abs(value) > 1.0e-12 for value in foot_values[start : start + 3])
            for start in (0, 3)
        ]
        if all(active_feet):
            live_lower = tuple(
                value * LIVE_BODY_TARGET_SAFETY_MARGIN
                for value in self._targets.both_feet_lower
            )
            live_upper = tuple(
                value * LIVE_BODY_TARGET_SAFETY_MARGIN
                for value in self._targets.both_feet_upper
            )
            if any(
                value < live_lower[index] or value > live_upper[index]
                for index, value in enumerate(foot_values)
            ):
                raise PicoHybridPolicyRuntimeError(
                    "simultaneous foot targets exceed their conservative live bound"
                )
            if any(
                float(obs.user_input.velocity[axis]) != 0.0
                for axis in ("vx", "vy", "vtheta")
            ):
                raise PicoHybridPolicyRuntimeError(
                    "simultaneous foot targets require zero locomotion command"
                )
        feet = _bounded_targets(
            foot_values, self._targets.foot_lower, self._targets.foot_upper
        )

        hand_mapping: Mapping[str, Sequence[float] | None] = (
            obs.user_input.hand_target or {}
        )
        hand_values: list[float] = []
        active: list[float] = []
        for side in ("left", "right"):
            target = hand_mapping.get(side)
            active.append(0.0 if target is None else 1.0)
            hand_values.extend((0.0, 0.0, 0.0) if target is None else target)
        hands = _bounded_targets(
            hand_values, self._targets.hand_lower, self._targets.hand_upper
        )
        hands.extend(active)
        return feet, hands

    def build_observation(self, obs: Observation) -> list[float]:
        try:
            gyro_policy = [
                float(value) for value in self._gyro_transform(obs.robot_state.gyro)
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise PicoHybridPolicyRuntimeError(
                "failed to map gyro to the policy frame"
            ) from exc
        gravity = [float(value) for value in obs.robot_state.projected_gravity]
        if len(gyro_policy) != 3 or len(gravity) != 3:
            raise PicoHybridPolicyRuntimeError(
                "gyro and projected gravity must be 3-vectors"
            )

        values: list[float] = gyro_policy + gravity
        for name, default in zip(
            self._contract.observation_joint_names, self._observation_defaults
        ):
            values.append(float(obs.robot_state.motor_positions[name]) - default)
        for name in self._contract.observation_joint_names:
            values.append(float(obs.robot_state.motor_velocities[name]))
        values.extend(float(value) for value in self._last_action)
        values.extend(
            float(obs.user_input.velocity[axis]) for axis in ("vx", "vy", "vtheta")
        )
        feet, hands = self._body_targets(obs)
        values.extend(feet)
        values.extend(hands)
        if len(values) != OBSERVATION_WIDTH or not all(
            math.isfinite(value) for value in values
        ):
            raise PicoHybridPolicyRuntimeError(
                f"unsafe policy observation (width={len(values)})"
            )
        return values

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Match the existing fall gate.  Scheduler/get-up arbitration owns the
        # sustained-fall transition; this tick only holds measured action joints.
        gravity = obs.robot_state.projected_gravity
        if len(gravity) != 3 or not all(
            math.isfinite(float(value)) for value in gravity
        ):
            raise PicoHybridPolicyRuntimeError("projected gravity is invalid")
        if float(gravity[2]) > -0.5:
            measured_positions = self._measured_action_positions(obs)
            for name in OBSERVATION_DOF_ORDER:
                command.target_angles[name] = self._physical_target(
                    measured_positions[name]
                )
            return

        with np.errstate(over="ignore", invalid="ignore"):
            policy_input = np.asarray([self.build_observation(obs)], dtype=np.float32)
        if not np.isfinite(policy_input).all():
            raise PicoHybridPolicyRuntimeError(
                "policy observation is not finite in float32"
            )
        outputs = self._session.run(None, {self._contract.input_name: policy_input})
        if len(outputs) != 1:
            raise PicoHybridPolicyRuntimeError("policy returned more than one output")
        try:
            action = np.asarray(outputs[0], dtype=np.float64)
            finite = bool(np.isfinite(action).all())
        except (TypeError, ValueError, OverflowError) as exc:
            raise PicoHybridPolicyRuntimeError(
                "policy returned a non-numeric output"
            ) from exc
        if action.shape != (1, ACTION_WIDTH) or not finite:
            raise PicoHybridPolicyRuntimeError(
                f"unsafe policy output shape/value: {action.shape}"
            )
        # No actor transform: the same raw float32 action recurs in the next
        # observation, and the absolute target is clipped like every other
        # policy: target = clip(HOME + raw * 1.0, -pi, +pi).
        with np.errstate(over="ignore", invalid="ignore"):
            raw_action = action[0].astype(np.float32)
        if not np.isfinite(raw_action).all():
            raise PicoHybridPolicyRuntimeError("raw action is not finite in float32")
        for index, (name, raw_value, absolute_maximum) in enumerate(
            zip(
                OBSERVATION_DOF_ORDER,
                raw_action,
                self._targets.raw_action_guard,
                strict=True,
            )
        ):
            if abs(float(raw_value)) > absolute_maximum:
                raise PicoHybridPolicyRuntimeError(
                    "raw action escaped the recorded finite-amplitude guard "
                    f"for {name} at action index {index}"
                )
        targets = [
            self._action_defaults[index] + float(raw_action[index])
            for index in range(ACTION_WIDTH)
        ]
        if not all(math.isfinite(target) for target in targets):
            raise PicoHybridPolicyRuntimeError("raw action produced a non-finite target")
        for name, target in zip(OBSERVATION_DOF_ORDER, targets, strict=True):
            command.target_angles[name] = self._physical_target(
                max(-SERVO_TARGET_RANGE_RAD, min(SERVO_TARGET_RANGE_RAD, target))
            )
        self._last_action = raw_action.copy()

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if "getup" in obs.user_input.active_moves:
            self._finish_stop()
            return
        if self._stop_start_time_s is None:
            measured_positions = self._measured_action_positions(obs)
            self._stop_start_time_s = float(obs.robot_state.time_s)
            self._stop_start_angles = measured_positions
        elapsed = max(0.0, float(obs.robot_state.time_s) - self._stop_start_time_s)
        duration = max(1e-6, self._neutral_return_duration_s)
        fraction = min(1.0, elapsed / duration)
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        for name in OBSERVATION_DOF_ORDER:
            start = self._stop_start_angles[name]
            target = (
                NEUTRAL_POSE[name]
                if fraction >= 1.0
                else start + (NEUTRAL_POSE[name] - start) * blend
            )
            command.target_angles[name] = self._physical_target(target)
        if fraction >= 1.0:
            if self._controller is not None:
                # Back at HOME with no learned policy: the static holding gain.
                self._controller.sync_write_kp(
                    ALL_MOTOR_IDS, [KP_HARDWARE_NEUTRAL] * len(ALL_MOTOR_IDS)
                )
            self._finish_stop()

    def _finish_stop(self) -> None:
        self._last_action.fill(0.0)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        _ = obs
        self._last_action.fill(0.0)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
