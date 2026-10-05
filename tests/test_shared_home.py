"""One reference pose (the centered HOME) and one target rule for every policy."""

import inspect
import json
import math
import unittest

import gc300_main
from constants import (
    HOME_PITCH_RAD,
    HOME_ROOT_POS_Z_M,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    SERVO_TARGET_RANGE_RAD,
)
from home_pose import HOME_POSE
from input.input_source import UserInput
from moves import walk as walk_module
from moves.getup import GetupMove
from moves.move import MotorCommand
from moves.pico_hybrid import EXPECTED_ACTION_DEFAULT_JOINT_POS, PICO_TELEOP_HOME_POSE
from observer import Observation, RobotState
from pico_arm_contract import PICO_ARM_HOME_RAD, PICO_ARM_JOINT_NAMES
from policy_fixtures import (
    ACTION_COUNT,
    GETUP_OBS_WIDTH,
    OLD_HOME,
    FakeSession,
    getup_contract_metadata,
)
from scheduler import Scheduler

# The centered HOME of mjlab_microban (HOME_FRAME at commit cb55431, now its
# config/home_pose.yaml), in degrees: pinned here so a changed
# config/home_pose.yaml is a deliberate, reviewed edit of this test too.
TRAINING_HOME_DEG = {
    "head": 0.0,
    "neck_roll": 0.0,
    "neck_pitch": 0.0,
    "left_shoulder_roll": 10.0,
    "right_shoulder_roll": -10.0,
    "left_shoulder_pitch": 0.0,
    "right_shoulder_pitch": 0.0,
    "left_elbow": -20.0,
    "right_elbow": -20.0,
    "left_hip_roll": 5.0,
    "right_hip_roll": -5.0,
    "left_hip_pitch": 1.198384259489,
    "right_hip_pitch": 1.198384259489,
    "left_hip_yaw": 0.0,
    "right_hip_yaw": 0.0,
    "left_knee": 0.0,
    "right_knee": 0.0,
    "left_ankle_roll": -5.0,
    "right_ankle_roll": 5.0,
    "left_ankle_pitch": -1.198384259489,
    "right_ankle_pitch": -1.198384259489,
}


class SharedHomeTest(unittest.TestCase):
    def test_neutral_pose_is_the_centered_training_home(self):
        self.assertEqual(set(NEUTRAL_POSE), set(MOTOR_TO_ID))
        for name, degrees in TRAINING_HOME_DEG.items():
            self.assertEqual(NEUTRAL_POSE[name], math.radians(degrees), name)
        self.assertEqual(dict(HOME_POSE["joint_pos_deg"]), TRAINING_HOME_DEG)
        self.assertEqual(HOME_PITCH_RAD, math.radians(1.198384259489))
        self.assertEqual(HOME_ROOT_POS_Z_M, 0.170554885633559)
        self.assertEqual(SERVO_TARGET_RANGE_RAD, math.pi)

    def test_every_runtime_copy_derives_from_neutral_pose(self):
        self.assertEqual(PICO_TELEOP_HOME_POSE, NEUTRAL_POSE)
        self.assertEqual(
            EXPECTED_ACTION_DEFAULT_JOINT_POS,
            tuple(NEUTRAL_POSE[name] for name in OBSERVATION_DOF_ORDER),
        )
        for side, names in PICO_ARM_JOINT_NAMES.items():
            self.assertEqual(
                PICO_ARM_HOME_RAD[side], tuple(NEUTRAL_POSE[name] for name in names)
            )

    def test_no_gc300_specific_model_or_ankle_bias(self):
        self.assertEqual(gc300_main.GC300_AGENT_NAME, "walk.onnx")
        self.assertEqual(walk_module.AGENT_NAME, "walk.onnx")
        self.assertFalse(hasattr(gc300_main, "GC300_FORWARD_ANKLE_BIAS_RAD"))
        self.assertNotIn(
            "neutral_ankle_pitch_bias_rad", inspect.signature(Scheduler).parameters
        )
        self.assertNotIn(
            "ankle_pitch_bias_rad", inspect.signature(walk_module.WalkMove).parameters
        )


def getup_observation(positions=None):
    positions = dict(NEUTRAL_POSE) if positions is None else positions
    return Observation(
        robot_state=RobotState(
            time_s=0.0,
            gyro=[0.0, 0.0, 0.0],
            projected_gravity=[0.0, 0.0, -1.0],
            motor_positions=positions,
            motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
        ),
        user_input=UserInput(active_moves={"getup"}, torque_enabled=True, getup_armed=True),
    )


def fake_getup(metadata, outputs=None):
    session = FakeSession(metadata, input_width=GETUP_OBS_WIDTH, outputs=outputs)
    return GetupMove(controller=None, session=session), session


class GetupHomeTest(unittest.TestCase):
    def test_model_at_the_centered_home_is_ready_and_uses_full_precision_home(self):
        raw = [0.25 * (index - 9) for index in range(ACTION_COUNT)]
        raw[0] = 400.0
        move, session = fake_getup(getup_contract_metadata(), outputs=[raw])
        self.assertTrue(move.model_ready)
        obs = getup_observation()
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)
        # Observation joint_pos residual is exactly zero at HOME (no 3-decimal
        # rounding of default_joint_pos leaks into the policy input).
        self.assertEqual(session.calls[0][6 : 6 + ACTION_COUNT], [0.0] * ACTION_COUNT)
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            expected = max(
                -SERVO_TARGET_RANGE_RAD,
                min(SERVO_TARGET_RANGE_RAD, NEUTRAL_POSE[name] + raw[index]),
            )
            self.assertEqual(command.target_angles[name], expected)
        # A huge raw output saturates exactly at the servo range (+pi).
        self.assertEqual(command.target_angles[OBSERVATION_DOF_ORDER[0]], math.pi)
        self.assertEqual(move._last_action, raw)

    def test_only_the_v5_servo_range_clip_is_accepted(self):
        def with_clip(lower, upper, version="v5"):
            metadata = getup_contract_metadata()
            metadata["microban_getup_contract"] = version
            metadata["action_clip_lower"] = ",".join(lower)
            metadata["action_clip_upper"] = ",".join(upper)
            return metadata

        servo_lo = [repr(-math.pi)] * ACTION_COUNT
        servo_hi = [repr(math.pi)] * ACTION_COUNT
        cases = {
            # The deployed +-1.57 policies (contract v4) and any v4 stamp.
            "old_v4_clip157": with_clip(["-1.570"] * 18, ["1.570"] * 18, "v4"),
            "v4_with_servo_clip": with_clip(servo_lo, servo_hi, "v4"),
            "v5_with_clip157": with_clip(["-1.570"] * 18, ["1.570"] * 18),
            "narrow_upper": with_clip(servo_lo, ["3.0"] * ACTION_COUNT),
            "narrow_one_joint": with_clip(servo_lo, servo_hi[:17] + ["3.141"]),
            # mjlab's 3-decimal formatter would publish 3.142 > pi.
            "three_decimal": with_clip(["-3.142"] * ACTION_COUNT, ["3.142"] * ACTION_COUNT),
            "wide_upper": with_clip(servo_lo, [repr(math.pi + 1.0e-5)] * ACTION_COUNT),
            "wide_one_joint": with_clip(servo_lo[:17] + ["-3.2"], servo_hi),
            "short": with_clip(servo_lo[:-1], servo_hi[:-1]),
            "nan": with_clip(["nan"] * ACTION_COUNT, servo_hi),
            "missing": {
                key: value
                for key, value in getup_contract_metadata().items()
                if key != "action_clip_upper"
            },
        }
        for case, metadata in cases.items():
            with self.subTest(case=case):
                move, _ = fake_getup(metadata)
                self.assertFalse(move.model_ready)
        self.assertTrue(fake_getup(with_clip(servo_lo, servo_hi))[0].model_ready)

    def test_model_trained_at_another_home_is_rejected(self):
        move, _ = fake_getup(getup_contract_metadata(OLD_HOME))
        self.assertFalse(move.model_ready)

    def test_home_stamp_is_required_and_must_match(self):
        missing = getup_contract_metadata()
        del missing["microban_getup_home_pose"]
        stale_stamp = getup_contract_metadata()
        stale_stamp["microban_getup_home_pose"] = getup_contract_metadata(OLD_HOME)[
            "microban_getup_home_pose"
        ]
        stale_defaults = getup_contract_metadata()
        stale_defaults["default_joint_pos"] = getup_contract_metadata(OLD_HOME)[
            "default_joint_pos"
        ]
        root = json.loads(getup_contract_metadata()["microban_getup_home_pose"])
        root["root_pos_m"][2] = 0.175
        wrong_root = getup_contract_metadata()
        wrong_root["microban_getup_home_pose"] = json.dumps(root)
        for case, metadata in (
            ("missing", missing),
            ("stale_stamp", stale_stamp),
            ("stale_defaults", stale_defaults),
            ("wrong_root", wrong_root),
        ):
            with self.subTest(case=case):
                move, _ = fake_getup(metadata)
                self.assertFalse(move.model_ready)

    def test_rejected_model_recovers_toward_the_shared_neutral_pose(self):
        move, session = fake_getup(getup_contract_metadata(OLD_HOME))
        start = {name: value + 0.3 for name, value in NEUTRAL_POSE.items()}
        obs = getup_observation(start)
        move.on_start(obs, MotorCommand())
        for tick in range(200):
            obs.robot_state.time_s = 0.02 * (tick + 1)
            command = MotorCommand()
            move.step(obs, command)
        self.assertEqual(session.calls, [])
        for name in MOTOR_TO_ID:
            self.assertAlmostEqual(command.target_angles[name], NEUTRAL_POSE[name], places=12)


if __name__ == "__main__":
    unittest.main()
