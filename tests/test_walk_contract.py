import math
import unittest

from constants import (
    HOME_PROJECTED_GRAVITY,
    HOME_ROOT_QUAT_WXYZ,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    SERVO_TARGET_RANGE_RAD,
)
from input.input_source import UserInput
from moves.move import MotorCommand, MoveState
from moves.walk import WalkMove, WalkPolicyContractError
from observer import Observation, RobotState
from policy_fixtures import (
    ACTION_COUNT,
    CENTERED_HOME,
    CENTERED_ROOT_POS_M,
    CENTERED_ROOT_QUAT_WXYZ,
    OLD_HOME,
    WALK_BIAS,
    WALK_OBS_WIDTH,
    WALK_POLICY_FIXTURE,
    WALK_POSITION_GAIN,
    FakeSession,
    csv,
    home_pose_json,
    walk_contract_metadata,
)


def pitched_quat(degrees):
    half = math.radians(degrees) / 2.0
    return [math.cos(half), 0.0, math.sin(half), 0.0]


def observation(positions=None, time_s=0.0, body_quat=None, head_orientation=None):
    """Standing still at HOME (trunk 10 deg forward) unless told otherwise."""
    positions = dict(NEUTRAL_POSE) if positions is None else positions
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            gyro=[0.0, 0.0, 0.0],
            projected_gravity=list(HOME_PROJECTED_GRAVITY),
            body_quat=list(HOME_ROOT_QUAT_WXYZ) if body_quat is None else body_quat,
            motor_positions=positions,
            motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
        ),
        user_input=UserInput(
            active_moves={"walk"}, head_orientation=head_orientation
        ),
    )


def fake_walk(metadata=None, **kwargs):
    session = FakeSession(
        walk_contract_metadata() if metadata is None else metadata,
        input_width=kwargs.pop("input_width", WALK_OBS_WIDTH),
        **kwargs,
    )
    return WalkMove(controller=None, session=session), session


class WalkContractTest(unittest.TestCase):
    def test_fixture_with_the_deployed_contract_loads(self):
        move = WalkMove(controller=None, policy_path=WALK_POLICY_FIXTURE)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(move._default_pose[name], NEUTRAL_POSE[name])
            self.assertEqual(
                move._action_clip[name], (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)
            )
        self.assertEqual(move.action_scale, 1.0)
        self.assertFalse(move._use_reference_phase)

    def test_missing_contract_fields_fail_closed(self):
        for key in (
            "walk_contract_version",
            "previous_action_semantics",
            "default_joint_pos",
            "joint_names",
            "action_scale",
            "action_clip_lower",
            "action_clip_upper",
            "home_pose",
        ):
            metadata = walk_contract_metadata()
            del metadata[key]
            with self.subTest(missing=key), self.assertRaises(WalkPolicyContractError):
                fake_walk(metadata)

    def test_inconsistent_contract_fields_fail_closed(self):
        three_decimal = walk_contract_metadata()
        three_decimal["default_joint_pos"] = ",".join(
            f"{NEUTRAL_POSE[name]:.3f}"
            for name in three_decimal["joint_names"].split(",")
        )
        cases = {
            "old_home": walk_contract_metadata(OLD_HOME),
            "three_decimal_default": three_decimal,
            "version": {**walk_contract_metadata(), "walk_contract_version": "v1"},
            "semantics": {
                **walk_contract_metadata(),
                "previous_action_semantics": "clipped_target",
            },
            "scale": {**walk_contract_metadata(), "action_scale": "0.5"},
            # The centered upright HOME v3 actors: their version string, their
            # HOME stamp (joints and identity root), or both.
            "centered_v3_contract": {
                **walk_contract_metadata(CENTERED_HOME),
                "walk_contract_version": "v3_centered_home_servo_range",
                "home_pose": home_pose_json(
                    CENTERED_HOME,
                    root_pos_m=CENTERED_ROOT_POS_M,
                    root_quat_wxyz=CENTERED_ROOT_QUAT_WXYZ,
                ),
            },
            "v3_version_at_this_home": {
                **walk_contract_metadata(),
                "walk_contract_version": "v3_centered_home_servo_range",
            },
            "centered_home_stamp": {
                **walk_contract_metadata(),
                "home_pose": home_pose_json(CENTERED_HOME),
            },
            "upright_root_stamp": {
                **walk_contract_metadata(),
                "home_pose": home_pose_json(root_quat_wxyz=CENTERED_ROOT_QUAT_WXYZ),
            },
            "centered_root_height_stamp": {
                **walk_contract_metadata(),
                "home_pose": home_pose_json(root_pos_m=CENTERED_ROOT_POS_M),
            },
            "home_stamp_not_json": {**walk_contract_metadata(), "home_pose": "{"},
            "old_v2_clip157_contract": {
                **walk_contract_metadata(),
                "walk_contract_version": "v2_centered_home_clip157",
                "action_clip_lower": csv([-1.57] * ACTION_COUNT),
                "action_clip_upper": csv([1.57] * ACTION_COUNT),
            },
            "v4_with_old_clip157": {
                **walk_contract_metadata(),
                "action_clip_lower": csv([-1.57] * ACTION_COUNT),
                "action_clip_upper": csv([1.57] * ACTION_COUNT),
            },
            "narrow_clip": {
                **walk_contract_metadata(),
                "action_clip_lower": csv([-1.0] * ACTION_COUNT),
            },
            "slightly_narrow_clip": {
                **walk_contract_metadata(),
                "action_clip_upper": csv([3.14] * ACTION_COUNT),
            },
            "three_decimal_clip": {
                **walk_contract_metadata(),
                "action_clip_upper": ",".join(["3.142"] * ACTION_COUNT),
            },
            "wide_clip": {
                **walk_contract_metadata(),
                "action_clip_upper": csv([math.pi + 1.0e-5] * ACTION_COUNT),
            },
            "one_joint_wide": {
                **walk_contract_metadata(),
                "action_clip_lower": csv([-math.pi] * (ACTION_COUNT - 1) + [-4.0]),
            },
            "short_clip": {
                **walk_contract_metadata(),
                "action_clip_upper": csv([SERVO_TARGET_RANGE_RAD] * (ACTION_COUNT - 1)),
            },
            "nan_clip": {
                **walk_contract_metadata(),
                "action_clip_upper": ",".join(["nan"] * ACTION_COUNT),
            },
            "action_order": {
                **walk_contract_metadata(),
                "action_joint_names": ",".join(reversed(OBSERVATION_DOF_ORDER)),
            },
        }
        for case, metadata in cases.items():
            with self.subTest(case=case), self.assertRaises(WalkPolicyContractError):
                fake_walk(metadata)
        for case, kwargs in (
            ("input_width", {"input_width": WALK_OBS_WIDTH + 1}),
            ("output_width", {"output_width": ACTION_COUNT - 1}),
        ):
            with self.subTest(case=case), self.assertRaises(WalkPolicyContractError):
                fake_walk(**kwargs)

    def test_installed_walk_onnx_is_accepted_only_on_contract(self):
        # Whatever is installed must carry the contract or be refused;
        # never silently accepted.
        try:
            move = WalkMove(controller=None)
        except WalkPolicyContractError:
            return
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(move._default_pose[name], NEUTRAL_POSE[name])


class WalkTargetRuleTest(unittest.TestCase):
    def test_real_onnx_target_is_home_plus_raw_action(self):
        move = WalkMove(controller=None, policy_path=WALK_POLICY_FIXTURE)
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
        move = WalkMove(controller=None, policy_path=WALK_POLICY_FIXTURE)
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

    def test_stop_returns_to_neutral_without_any_ankle_bias(self):
        move, _ = fake_walk()
        positions = {name: value + 0.3 for name, value in NEUTRAL_POSE.items()}
        move.state = MoveState.STOPPING
        move.on_stop(observation(positions, time_s=1.0), MotorCommand())
        final = MotorCommand()
        move.on_stop(observation(positions, time_s=2.0), final)
        self.assertEqual(move.state, MoveState.INACTIVE)
        for name in OBSERVATION_DOF_ORDER:
            self.assertAlmostEqual(final.target_angles[name], NEUTRAL_POSE[name], places=12)

    def test_neck_holds_home_and_cancels_sway_around_it(self):
        move, _ = fake_walk()
        cases = (
            # (trunk pitch from vertical, VR head pitch, expected neck pitch), deg
            (10.0, None, 0.0),  # standing at HOME: the trained neck angle
            (0.0, None, 10.0),  # trunk vertical: keep the HOME head attitude
            (15.0, None, -5.0),  # sway forward of HOME
            (10.0, 0.0, -10.0),  # VR: the commanded world attitude (level)
            (10.0, 5.0, -5.0),
        )
        for trunk_deg, vr_pitch_deg, neck_deg in cases:
            head = (
                None
                if vr_pitch_deg is None
                else {"roll": 0.0, "pitch": math.radians(vr_pitch_deg)}
            )
            obs = observation(body_quat=pitched_quat(trunk_deg), head_orientation=head)
            command = MotorCommand()
            move.on_start(obs, command)
            move.step(obs, command)
            with self.subTest(trunk=trunk_deg, vr=vr_pitch_deg):
                self.assertAlmostEqual(
                    command.target_angles["neck_pitch"], math.radians(neck_deg), places=9
                )
                self.assertAlmostEqual(command.target_angles["neck_roll"], 0.0, places=9)

    def test_ankle_bias_interface_is_gone(self):
        with self.assertRaises(TypeError):
            WalkMove(controller=None, ankle_pitch_bias_rad=0.0)  # type: ignore[call-arg]
        move, _ = fake_walk()
        self.assertFalse(hasattr(move, "seed_next_start_from_hardware_neutral"))


if __name__ == "__main__":
    unittest.main()
