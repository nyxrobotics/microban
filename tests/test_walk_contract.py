import math
import unittest

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
from moves.walk import WalkMove
from policy_contract import PolicyContractError
from observer import Observation, RobotState
from policy_fixtures import (
    ACTION_COUNT,
    OTHER_HOME,
    WALK_BIAS,
    WALK_OBS_WIDTH,
    WALK_POSITION_GAIN,
    FakeSession,
    LinearWalkSession,
    walk_contract_metadata,
)


def observation(positions=None, time_s=0.0):
    positions = dict(NEUTRAL_POSE) if positions is None else positions
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            gyro=[0.0, 0.0, 0.0],
            projected_gravity=[0.0, 0.0, -1.0],
            body_quat=[1.0, 0.0, 0.0, 0.0],
            motor_positions=positions,
            motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
        ),
        user_input=UserInput(active_moves={"walk"}),
    )


def fake_walk(metadata=None, **kwargs):
    session = FakeSession(
        walk_contract_metadata() if metadata is None else metadata,
        input_width=kwargs.pop("input_width", WALK_OBS_WIDTH),
        **kwargs,
    )
    return WalkMove(controller=None, session=session), session


class GainController:
    def __init__(self):
        self.kp_writes = []

    def sync_write_kp(self, ids, gains):
        self.kp_writes.append((list(ids), list(gains)))


class WalkContractTest(unittest.TestCase):
    def test_contract_at_this_home_loads_with_robot_constants(self):
        move = WalkMove(controller=None, session=LinearWalkSession())
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(move._default_pose[name], NEUTRAL_POSE[name])
            self.assertEqual(
                move._action_clip[name], (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)
            )
        self.assertEqual(move.action_scale, 1.0)
        self.assertEqual(move.contract.kind, "walk")

    def test_other_home_reference_phase_and_other_kinds_are_refused(self):
        with self.assertRaises(PolicyContractError):
            fake_walk(walk_contract_metadata(OTHER_HOME))
        # The 65-wide reference-phase actors are not part of the contract.
        with self.assertRaises(PolicyContractError):
            fake_walk(input_width=WALK_OBS_WIDTH + 2)
        with self.assertRaises(PolicyContractError):
            fake_walk({**walk_contract_metadata(), "microban_policy_kind": "getup"})


class WalkGainTest(unittest.TestCase):
    def test_policy_runs_every_joint_at_the_trained_gain_and_holds_home_at_p900(self):
        controller = GainController()
        move = WalkMove(controller=controller, session=LinearWalkSession())
        move.on_start(observation(), MotorCommand())
        self.assertEqual(controller.kp_writes, [(list(MOTOR_TO_ID.values()), [KP_RL] * 21)])

        move.state = MoveState.STOPPING
        move.on_stop(observation(time_s=1.0), MotorCommand())
        self.assertEqual(len(controller.kp_writes), 1)
        move.on_stop(observation(time_s=2.0), MotorCommand())
        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(
            controller.kp_writes[-1], (list(MOTOR_TO_ID.values()), [KP_HARDWARE_NEUTRAL] * 21)
        )


class WalkTargetRuleTest(unittest.TestCase):
    def test_real_onnx_target_is_home_plus_raw_action(self):
        move = WalkMove(controller=None, session=LinearWalkSession())
        offsets = {name: 0.01 * (index - 9) for index, name in enumerate(OBSERVATION_DOF_ORDER)}
        positions = dict(NEUTRAL_POSE)
        for name, offset in offsets.items():
            positions[name] += offset
        obs = observation(positions)
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            raw = WALK_POSITION_GAIN * offsets[name] + WALK_BIAS[index]
            self.assertAlmostEqual(move._last_action[index], raw, places=6)
            self.assertAlmostEqual(
                command.target_angles[name], NEUTRAL_POSE[name] + raw, places=6
            )

    def test_large_raw_output_is_clipped_but_recurs_raw(self):
        move = WalkMove(controller=None, session=LinearWalkSession())
        positions = dict(NEUTRAL_POSE)
        signs = {}
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            signs[name] = 1.0 if index % 2 else -1.0
            positions[name] += signs[name] * 20.0
        obs = observation(positions)
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)
        self.assertFalse(move.policy_faulted)
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            self.assertEqual(command.target_angles[name], signs[name] * SERVO_TARGET_RANGE_RAD)
            self.assertGreater(abs(move._last_action[index]), 9.0)
        next_obs = move.build_observation(obs)
        start = 6 + 2 * ACTION_COUNT
        self.assertEqual(next_obs[start : start + ACTION_COUNT], move._last_action)

    def test_hundreds_of_radians_is_not_a_fault(self):
        raw = [300.0 * (1 if index % 2 else -1) for index in range(ACTION_COUNT)]
        move, _ = fake_walk(outputs=[raw])
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)
        self.assertFalse(move.policy_faulted)
        self.assertEqual(move._last_action, raw)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(abs(command.target_angles[name]), SERVO_TARGET_RANGE_RAD)

    def test_non_finite_output_latches_a_fault_and_holds_last_targets(self):
        good = [0.1] * ACTION_COUNT
        bad = list(good)
        bad[4] = math.nan
        move, session = fake_walk(outputs=[good, bad, good])
        obs = observation()
        first = MotorCommand()
        move.on_start(obs, first)
        move.step(obs, first)
        held = {name: first.target_angles[name] for name in OBSERVATION_DOF_ORDER}
        self.assertEqual(move._last_action, good)

        measured = {name: value + 0.05 for name, value in NEUTRAL_POSE.items()}
        faulted_obs = observation(measured)
        for _ in range(3):
            command = MotorCommand()
            move.step(faulted_obs, command)
            self.assertTrue(move.policy_faulted)
            for name in OBSERVATION_DOF_ORDER:
                self.assertEqual(command.target_angles[name], held[name])
            for name in ("neck_roll", "neck_pitch"):
                self.assertEqual(command.target_angles[name], measured[name])
        # The fault latches: no further inference after the bad output.
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(move._last_action, good)

        restart = MotorCommand()
        move.on_start(faulted_obs, restart)
        self.assertFalse(move.policy_faulted)
        move.step(faulted_obs, restart)
        self.assertEqual(len(session.calls), 3)

    def test_inference_exception_and_wrong_width_are_faults(self):
        for case, outputs in (("width", [[0.0] * (ACTION_COUNT - 1)]), ("inf", [[math.inf] * ACTION_COUNT])):
            with self.subTest(case=case):
                move, _ = fake_walk(outputs=outputs)
                obs = observation()
                command = MotorCommand()
                move.on_start(obs, command)
                move.step(obs, command)
                self.assertTrue(move.policy_faulted)
                for name in OBSERVATION_DOF_ORDER:
                    self.assertEqual(command.target_angles[name], NEUTRAL_POSE[name])

        move, session = fake_walk()

        def explode(*_args):
            raise RuntimeError("inference failed")

        session.run = explode
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)
        self.assertTrue(move.policy_faulted)

    def test_stop_returns_to_neutral(self):
        move, _ = fake_walk()
        positions = {name: value + 0.3 for name, value in NEUTRAL_POSE.items()}
        move.state = MoveState.STOPPING
        move.on_stop(observation(positions, time_s=1.0), MotorCommand())
        final = MotorCommand()
        move.on_stop(observation(positions, time_s=2.0), final)
        self.assertEqual(move.state, MoveState.INACTIVE)
        for name in OBSERVATION_DOF_ORDER:
            self.assertAlmostEqual(final.target_angles[name], NEUTRAL_POSE[name], places=12)



if __name__ == "__main__":
    unittest.main()
