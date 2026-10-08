"""PICO hybrid runtime: observation, raw-action target rule, guards and gains.

The contract itself (metadata, manifest, self-test) is test_policy_contract.py.
"""

import math
import unittest

import numpy as np

from constants import (
    KP_HARDWARE_NEUTRAL,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    SERVO_TARGET_RANGE_RAD,
)
from input.input_source import UserInput
from moves.move import MotorCommand, MoveState
from moves.pico_hybrid import (
    PicoHybridMove,
    PicoHybridPolicyRuntimeError,
)
from observer import Observation, RobotState
from policy_contract import PolicyContractError
from policy_fixtures import (
    PICO_RAW_ACTION_GUARD,
    fake_session,
    pico_contract_metadata,
)


class FakeController:
    def __init__(self):
        self.kp_writes = []

    def sync_write_kp(self, ids, values):
        self.kp_writes.append((list(ids), list(values)))


def pico_session(output=None, metadata=None):
    session = fake_session("pico", metadata)
    if output is not None:
        session.outputs = [np.asarray(output, dtype=np.float64).reshape(-1).tolist()]
    return session


def set_output(session, output):
    session.outputs = [np.asarray(output, dtype=np.float64).reshape(-1).tolist()]


def observation(time_s=0.0):
    positions = {name: NEUTRAL_POSE[name] for name in MOTOR_TO_ID}
    velocities = {name: 0.0 for name in MOTOR_TO_ID}
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            gyro=[0.1, 0.2, 0.3],
            projected_gravity=[0.0, 0.0, -1.0],
            motor_positions=positions,
            motor_velocities=velocities,
        ),
        user_input=UserInput(
            active_moves={"walk"},
            velocity={"vx": 0.2, "vy": -0.1, "vtheta": 0.3},
            foot_target={
                "left": (9.0, -9.0, 0.04),
                "right": (0.0, 0.0, 0.0),
            },
            hand_target={"left": (0.02, 0.03, -0.04), "right": None},
        ),
    )


def clipped_target(index, raw_value):
    name = OBSERVATION_DOF_ORDER[index]
    return max(
        -SERVO_TARGET_RANGE_RAD,
        min(SERVO_TARGET_RANGE_RAD, NEUTRAL_POSE[name] + float(np.float32(raw_value))),
    )


class PicoHybridContractTest(unittest.TestCase):
    def test_contract_session_loads_with_the_trained_command_support(self):
        move = PicoHybridMove(session=pico_session())
        self.assertEqual(move.contract.kind, "pico")
        self.assertEqual(move.contract.pico.raw_action_guard, (PICO_RAW_ACTION_GUARD,) * 18)
        self.assertEqual(move.contract.pico.foot_upper, (0.03, 0.03, 0.05) * 2)

    def test_other_kinds_and_empty_metadata_are_refused(self):
        for case, metadata in (
            ("walk_kind", {**pico_contract_metadata(), "microban_policy_kind": "walk"}),
            ("empty", {}),
        ):
            with self.subTest(case=case), self.assertRaises(PolicyContractError):
                PicoHybridMove(session=pico_session(metadata=metadata))


class PicoHybridStepTest(unittest.TestCase):
    def test_observation_is_exact_83_ordered_values_and_clips_targets(self):
        move = PicoHybridMove(session=pico_session(), gyro_transform=lambda value: value)
        values = move.build_observation(observation())
        self.assertEqual(len(values), 83)
        self.assertEqual(values[:6], [0.1, 0.2, 0.3, 0.0, 0.0, -1.0])
        self.assertEqual(values[6:27], [0.0] * 21)
        self.assertEqual(values[27:48], [0.0] * 21)
        self.assertEqual(values[48:66], [0.0] * 18)
        self.assertEqual(values[66:69], [0.2, -0.1, 0.3])
        self.assertEqual(values[69:75], [0.03, -0.03, 0.04, 0.0, 0.0, 0.0])
        self.assertEqual(values[75:83], [0.02, 0.03, -0.04, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_gyro_is_the_raw_imu_site_reading(self):
        move = PicoHybridMove(session=pico_session())
        self.assertEqual(move.build_observation(observation())[:3], [0.1, 0.2, 0.3])

    def test_step_clips_target_and_preserves_raw_recurrence(self):
        raw = np.linspace(-3.5, 3.5, 18, dtype=np.float32).reshape(1, 18)
        session = pico_session(raw)
        move = PicoHybridMove(session=session)
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)

        # Every policy: target = clip(HOME + raw * 1.0, -pi, +pi), and the
        # previous-action observation is the unclipped raw output.
        expected = {
            name: clipped_target(index, raw[0, index])
            for index, name in enumerate(OBSERVATION_DOF_ORDER)
        }
        self.assertTrue(any(abs(value) == SERVO_TARGET_RANGE_RAD for value in expected.values()))
        for name, value in expected.items():
            self.assertEqual(command.target_angles[name], value)
        next_observation = move.build_observation(obs)
        np.testing.assert_array_equal(np.asarray(next_observation[48:66], dtype=np.float32), raw[0])

    def test_extreme_targets_are_clipped_at_the_guard_boundary(self):
        raw = np.asarray(
            [[PICO_RAW_ACTION_GUARD, *(-24.0 if index % 2 else 24.0 for index in range(1, 18))]],
            dtype=np.float32,
        )
        move = PicoHybridMove(session=pico_session(raw))
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)
        self.assertEqual(move.state, MoveState.ACTIVE)
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            self.assertEqual(command.target_angles[name], clipped_target(index, raw[0, index]))
            self.assertEqual(abs(command.target_angles[name]), SERVO_TARGET_RANGE_RAD)
        np.testing.assert_array_equal(move._last_action, raw[0])

    def test_guard_escape_is_rejected_before_any_target_write(self):
        outside = float(np.nextafter(np.float32(PICO_RAW_ACTION_GUARD), np.float32(math.inf)))
        for sign in (-1.0, 1.0):
            session = pico_session()
            move = PicoHybridMove(session=session)
            obs = observation()
            command = MotorCommand()
            move.on_start(obs, command)
            held = dict(command.target_angles)
            raw = np.zeros(18)
            raw[7] = sign * outside
            set_output(session, raw)
            with (
                self.subTest(sign=sign),
                self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "finite-amplitude guard"),
            ):
                move.step(obs, command)
            self.assertEqual(command.target_angles, held)
            np.testing.assert_array_equal(move._last_action, np.zeros(18, dtype=np.float32))

    def test_late_float32_overflow_is_rejected_before_any_target_write(self):
        session = pico_session()
        move = PicoHybridMove(session=session)
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        held = dict(command.target_angles)
        set_output(session, np.full(18, 1.0e100))
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "float32"):
            move.step(obs, command)
        self.assertEqual(command.target_angles, held)

    def test_observation_float32_overflow_is_rejected_before_inference(self):
        session = pico_session()
        move = PicoHybridMove(session=session)
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        held = dict(command.target_angles)
        obs.robot_state.motor_positions[OBSERVATION_DOF_ORDER[0]] = 1.0e100
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "observation.*float32"):
            move.step(obs, command)
        self.assertEqual(session.calls, [])
        self.assertEqual(command.target_angles, held)

    def test_nonfinite_policy_output_fails_closed(self):
        session = pico_session()
        move = PicoHybridMove(session=session)
        output = np.zeros(18)
        output[0] = math.nan
        set_output(session, output)
        obs = observation()
        move.on_start(obs, MotorCommand())
        with self.assertRaises(PicoHybridPolicyRuntimeError):
            move.step(obs, MotorCommand())

    def test_fall_tick_holds_measured_action_joints_without_inference(self):
        session = pico_session()
        move = PicoHybridMove(session=session)
        obs = observation()
        move.on_start(obs, MotorCommand())
        obs.robot_state.projected_gravity = [0.0, 1.0, 0.0]
        command = MotorCommand()
        move.step(obs, command)
        self.assertEqual(session.calls, [])
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(command.target_angles[name], obs.robot_state.motor_positions[name])

    def test_simultaneous_both_feet_require_live_bounds_and_zero_twist(self):
        move = PicoHybridMove(session=pico_session(), gyro_transform=lambda value: value)
        obs = observation()
        obs.user_input.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
        obs.user_input.foot_target = {
            "left": (0.008, -0.008, 0.016),
            "right": (-0.008, 0.008, 0.016),
        }
        feet, _hands = move._body_targets(obs)
        np.testing.assert_allclose(
            feet, [*obs.user_input.foot_target["left"], *obs.user_input.foot_target["right"]]
        )
        obs.user_input.foot_target["left"] = (0.0080001, 0.0, 0.01)
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "live bound"):
            move._body_targets(obs)
        obs.user_input.foot_target["left"] = (0.005, 0.0, 0.01)
        obs.user_input.velocity["vx"] = 0.01
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "zero locomotion command"):
            move._body_targets(obs)

    def test_support_floor_band_must_arrive_as_exact_zero(self):
        move = PicoHybridMove(session=pico_session(), gyro_transform=lambda value: value)
        obs = observation()
        obs.user_input.foot_target = {"left": (0.001, 0.0, 0.0), "right": (0.0, 0.0, 0.0)}
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "floor-band"):
            move._body_targets(obs)
        obs.user_input.foot_target["left"] = (0.0, 0.0, 0.0025)
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "floor-band"):
            move._body_targets(obs)
        obs.user_input.foot_target["left"] = (0.0, 0.0, 0.0)
        feet, _hands = move._body_targets(obs)
        self.assertEqual(feet, [0.0] * 6)


class PicoHybridStartStopTest(unittest.TestCase):
    def test_policy_runs_every_joint_at_p125_and_home_holds_at_p900(self):
        controller = FakeController()
        move = PicoHybridMove(controller=controller, session=pico_session())
        obs = observation(10.0)
        move.on_start(obs, MotorCommand())
        self.assertEqual(controller.kp_writes, [(list(MOTOR_TO_ID.values()), [KP_RL] * 21)])
        move.state = MoveState.STOPPING
        move.on_stop(obs, MotorCommand())
        obs.robot_state.time_s = 10.8
        move.on_stop(obs, MotorCommand())
        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(
            controller.kp_writes[-1], (list(MOTOR_TO_ID.values()), [KP_HARDWARE_NEUTRAL] * 21)
        )

    def test_release_ends_at_home_without_discontinuity(self):
        move = PicoHybridMove(session=pico_session())
        obs = observation(10.0)
        move.on_start(obs, MotorCommand())
        move.state = MoveState.STOPPING
        obs.robot_state.motor_positions = {
            name: NEUTRAL_POSE[name] + 0.2 for name in MOTOR_TO_ID
        }
        first = MotorCommand()
        move.on_stop(obs, first)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(first.target_angles[name], NEUTRAL_POSE[name] + 0.2)
        obs.robot_state.time_s = 10.8
        final = MotorCommand()
        move.on_stop(obs, final)
        self.assertEqual(move.state, MoveState.INACTIVE)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(final.target_angles[name], NEUTRAL_POSE[name])
        self.assertEqual(final.target_angles, MotorCommand().target_angles)

    def test_getup_cancels_release_interpolation(self):
        move = PicoHybridMove(session=pico_session())
        obs = observation(10.0)
        move.on_start(obs, MotorCommand())
        move.state = MoveState.STOPPING
        obs.user_input.active_moves.add("getup")
        move.on_stop(obs, MotorCommand())
        self.assertEqual(move.state, MoveState.INACTIVE)

    def test_start_rejects_nonfinite_motor_position_before_gain_write(self):
        controller = FakeController()
        move = PicoHybridMove(controller=controller, session=pico_session())
        obs = observation()
        obs.robot_state.motor_positions[OBSERVATION_DOF_ORDER[0]] = math.nan
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "non-finite"):
            move.on_start(obs, MotorCommand())
        self.assertEqual(controller.kp_writes, [])
        self.assertEqual(move.state, MoveState.INACTIVE)

    def test_stop_rejects_nonfinite_motor_position_before_interpolation(self):
        move = PicoHybridMove(session=pico_session())
        obs = observation()
        move.on_start(obs, MotorCommand())
        move.state = MoveState.STOPPING
        obs.robot_state.motor_positions[OBSERVATION_DOF_ORDER[-1]] = math.inf
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "non-finite"):
            move.on_stop(obs, MotorCommand())
        self.assertIsNone(move._stop_start_time_s)
        self.assertEqual(move._stop_start_angles, {})


if __name__ == "__main__":
    unittest.main()
