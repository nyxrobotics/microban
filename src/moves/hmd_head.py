# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Safe three-axis camera-head tracking for VR teleoperation."""

import math

from constants import NEUTRAL_POSE
from moves.move import MotorCommand, Move, MoveState
from observer import Observation


HEAD_JOINTS = ("head", "neck_roll", "neck_pitch")
JOINT_LIMITS = {
    # Small margins keep commanded positions away from the mechanical stops.
    "head": (-math.radians(85.0), math.radians(85.0)),
    "neck_roll": (-math.radians(23.0), math.radians(23.0)),
    "neck_pitch": (-math.radians(85.0), math.radians(23.0)),
}


def _body_roll_pitch(body_quat: list[float]) -> tuple[float, float] | None:
    if len(body_quat) != 4 or not all(math.isfinite(value) for value in body_quat):
        return None
    norm = math.sqrt(sum(value * value for value in body_quat))
    if norm < 1e-6:
        return None
    w, x, y, z = (value / norm for value in body_quat)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[row][k] * b[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _rx(angle: float) -> list[list[float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]


def _ry(angle: float) -> list[list[float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]


def _rz(angle: float) -> list[list[float]]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]


def _extract_zxy(rotation: list[list[float]]) -> tuple[float, float, float]:
    """Return yaw, roll, pitch for Rz(yaw) @ Rx(roll) @ Ry(pitch)."""
    roll = math.asin(max(-1.0, min(1.0, rotation[2][1])))
    pitch = math.atan2(-rotation[2][0], rotation[2][2])
    yaw = math.atan2(-rotation[0][1], rotation[1][1])
    return yaw, roll, pitch


class HmdHeadTrackingMove(Move):
    """Track HMD roll/pitch/yaw while retaining robot-side limits.

    Roll and pitch are stabilized against trunk tilt. Yaw is deliberately trunk-
    relative because the robot IMU has no reliable absolute heading. Holding the
    right controller trigger sets ``head_yaw_front`` and slews yaw to zero while
    roll/pitch tracking remains active.
    """

    def __init__(self, slew_rate_rad_s: float = 2.5) -> None:
        super().__init__()
        self._slew_rate_rad_s = slew_rate_rad_s
        self._last_targets: dict[str, float] = {}
        self._last_time_s: float | None = None
        self._suspended_for_getup = False

    def _initialize_from_observation(self, obs: Observation) -> None:
        previous = self._last_targets
        targets = {}
        for name in HEAD_JOINTS:
            measured = float(
                obs.robot_state.motor_positions.get(
                    name, previous.get(name, NEUTRAL_POSE[name])
                )
            )
            targets[name] = (
                measured
                if math.isfinite(measured)
                else previous.get(name, NEUTRAL_POSE[name])
            )
        self._last_targets = targets
        now = float(obs.robot_state.time_s)
        self._last_time_s = now if math.isfinite(now) else self._last_time_s

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        self._initialize_from_observation(obs)
        for name, target in self._last_targets.items():
            command.target_angles[name] = target
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Get-up owns all 21 joints. Suppress this move even if registration order is
        # changed later, then resume from measured angles instead of stale targets.
        if self._handle_getup_ownership(obs):
            return
        self._drive(obs, command)

    def _handle_getup_ownership(self, obs: Observation) -> bool:
        if "getup" in obs.user_input.active_moves:
            self._suspended_for_getup = True
            return True
        if self._suspended_for_getup:
            self._initialize_from_observation(obs)
            self._suspended_for_getup = False
        return False

    def _drive(self, obs: Observation, command: MotorCommand) -> None:
        requested = obs.user_input.head_orientation or {}
        values = {
            axis: float(requested.get(axis, 0.0))
            for axis in ("roll", "pitch", "yaw")
        }
        for axis, value in values.items():
            if not math.isfinite(value):
                values[axis] = 0.0

        trunk_angles = _body_roll_pitch(obs.robot_state.body_quat)
        if trunk_angles is None:
            # Without a valid gravity frame, an angle-compensation command is unsafe.
            # Hold measured/last-known-finite neck targets until IMU data recovers.
            self._initialize_from_observation(obs)
            command.target_angles.update(self._last_targets)
            return
        trunk_roll, trunk_pitch = trunk_angles
        # The physical chain is trunk -> yaw(Z) -> roll(X) -> pitch(Y). Compose
        # rotations before solving that chain so mixed HMD poses and trunk tilt do
        # not suffer the large cross-axis error produced by subtracting Euler angles.
        camera_yaw = 0.0 if obs.user_input.head_yaw_front else values["yaw"]
        desired_camera = _matmul(
            _matmul(_rz(camera_yaw), _rx(values["roll"])),
            _ry(values["pitch"]),
        )
        # IMU yaw is not a stable global heading, so compensate only the measured
        # ZYX roll/pitch tilt and keep HMD yaw relative to the trunk heading.
        trunk_tilt = _matmul(_ry(trunk_pitch), _rx(trunk_roll))
        neck_rotation = _matmul(_transpose(trunk_tilt), desired_camera)
        solved_yaw, solved_roll, solved_pitch = _extract_zxy(neck_rotation)
        desired = {
            "head": 0.0 if obs.user_input.head_yaw_front else solved_yaw,
            "neck_roll": solved_roll,
            "neck_pitch": solved_pitch,
        }
        self._write_slew_limited(obs, command, desired)

    def _write_slew_limited(
        self,
        obs: Observation,
        command: MotorCommand,
        desired: dict[str, float],
    ) -> None:
        if not self._last_targets:
            self._initialize_from_observation(obs)
        now = float(obs.robot_state.time_s)
        if not math.isfinite(now):
            now = self._last_time_s if self._last_time_s is not None else 0.0
        previous_time = self._last_time_s if self._last_time_s is not None else now
        dt = max(0.001, min(0.1, now - previous_time))
        max_step = self._slew_rate_rad_s * dt

        for name in HEAD_JOINTS:
            lo, hi = JOINT_LIMITS[name]
            target = max(lo, min(hi, desired[name] + NEUTRAL_POSE[name]))
            previous = self._last_targets[name]
            delta = max(-max_step, min(max_step, target - previous))
            current = previous + delta
            self._last_targets[name] = current
            command.target_angles[name] = current
        self._last_time_s = now

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        # STOPPING can overlap an automatic get-up after a network timeout. Get-up
        # still owns the neck in that state; resume the neutral slew from measured
        # angles only after recovery releases it.
        if self._handle_getup_ownership(obs):
            return
        desired = {name: 0.0 for name in HEAD_JOINTS}
        self._write_slew_limited(obs, command, desired)
        if all(abs(self._last_targets[name] - NEUTRAL_POSE[name]) < 1e-3 for name in HEAD_JOINTS):
            self._last_targets = {}
            self._last_time_s = None
            self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        # The physical neck may have moved while callbacks were suppressed. Continue
        # from feedback, never from the pre-fault target/time pair.
        if self.state != MoveState.INACTIVE:
            self._initialize_from_observation(obs)
