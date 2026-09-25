import math
import unittest

from constants import KP_DEFAULT, KP_RL, MOTOR_TO_ID, NEUTRAL_POSE
from input.input_source import UserInput
from moves.move import MotorCommand, MoveState
from moves.pico_arms import PICO_ARM_JOINT_ORDER, PicoArmTrackingMove
from observer import Observation, RobotState
from pico_arm_contract import (
    PICO_ARM_HOME_RAD,
    PICO_ARM_JOINT_NAMES,
    PICO_ARM_LOWER_RAD,
    PICO_ARM_SIDES,
    PICO_ARM_TARGET_CONTRACT_REVISION,
    PICO_ARM_UPPER_RAD,
)

TARGET = {
    "left": (
        math.radians(12.0),
        math.radians(18.0),
        math.radians(-32.0),
    ),
    "right": (
        math.radians(-12.0),
        math.radians(-18.0),
        math.radians(-32.0),
    ),
}


def observation(
    time_s: float,
    *,
    enabled: bool,
    target=TARGET,
    active_moves=("pico_arms",),
    positions=None,
) -> Observation:
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            motor_positions=(
                dict(NEUTRAL_POSE) if positions is None else dict(positions)
            ),
        ),
        user_input=UserInput(
            active_moves=set(active_moves),
            locomotion_policy="pico_teleop",
            arm_tracking_enabled=enabled,
            arm_joint_target=target,
        ),
    )


class FakeController:
    def __init__(self):
        self.kp_writes = []

    def sync_write_kp(self, ids, gains):
        self.kp_writes.append((list(ids), list(gains)))


class PicoArmContractTest(unittest.TestCase):
    def test_direct_contract_is_expanded_and_distinct_from_v12_policy_fk_box(self):
        from moves.pico_hybrid import EXPECTED_V12_HAND_TARGET_FK

        self.assertEqual(
            PICO_ARM_TARGET_CONTRACT_REVISION,
            "microban_hmd_absolute_arm_fk_live_box_pitch100_roll120_elbow110_v1",
        )
        self.assertNotEqual(
            PICO_ARM_TARGET_CONTRACT_REVISION,
            EXPECTED_V12_HAND_TARGET_FK["revision"],
        )
        self.assertEqual(EXPECTED_V12_HAND_TARGET_FK["side_order"], ["left", "right"])
        self.assertEqual(
            EXPECTED_V12_HAND_TARGET_FK["joint_order"],
            ["shoulder_pitch", "shoulder_roll", "elbow"],
        )
        expected_lower_deg = {
            "left": (-100.0, 10.0, -110.0),
            "right": (-100.0, -120.0, -110.0),
        }
        expected_upper_deg = {
            "left": (100.0, 120.0, 0.0),
            "right": (100.0, -10.0, 0.0),
        }
        for side in PICO_ARM_SIDES:
            with self.subTest(side=side):
                for actual, expected in zip(
                    PICO_ARM_LOWER_RAD[side],
                    expected_lower_deg[side],
                    strict=True,
                ):
                    self.assertAlmostEqual(math.degrees(actual), expected)
                for actual, expected in zip(
                    PICO_ARM_UPPER_RAD[side],
                    expected_upper_deg[side],
                    strict=True,
                ):
                    self.assertAlmostEqual(math.degrees(actual), expected)

        # Release HOME stays unchanged even though the direct tracking envelope
        # no longer aliases the learned policy's narrow FK metadata.
        self.assertEqual(
            {
                side: tuple(round(math.degrees(value), 10) for value in values)
                for side, values in PICO_ARM_HOME_RAD.items()
            },
            {"left": (0.0, 10.0, -20.0), "right": (0.0, -10.0, -20.0)},
        )
        self.assertEqual(
            EXPECTED_V12_HAND_TARGET_FK["joint_lower_deg"],
            [[-25.0, 10.0, -50.0], [-25.0, -30.0, -50.0]],
        )
        self.assertEqual(
            EXPECTED_V12_HAND_TARGET_FK["joint_upper_deg"],
            [[25.0, 30.0, -10.0], [25.0, -10.0, -10.0]],
        )


class PicoArmTrackingMoveTest(unittest.TestCase):
    def test_held_right_trigger_overwrites_only_six_arms_standing_or_walking(self):
        for active_moves in (("pico_arms",), ("walk", "pico_arms")):
            with self.subTest(active_moves=active_moves):
                move = PicoArmTrackingMove(slew_rate_rad_s=1000.0)
                obs = observation(
                    0.0,
                    enabled=True,
                    active_moves=active_moves,
                )
                move.on_start(obs, MotorCommand())

                obs.robot_state.time_s = 0.1
                command = MotorCommand(
                    target_angles={name: 7.0 for name in MOTOR_TO_ID}
                )
                move.step(obs, command)

                expected = {
                    name: TARGET[side][index]
                    for side in PICO_ARM_SIDES
                    for index, name in enumerate(PICO_ARM_JOINT_NAMES[side])
                }
                for name in MOTOR_TO_ID:
                    self.assertEqual(
                        command.target_angles[name], expected.get(name, 7.0)
                    )

    def test_valid_release_commands_exact_pico_home_not_global_pitch(self):
        move = PicoArmTrackingMove(slew_rate_rad_s=1000.0)
        obs = observation(0.0, enabled=True)
        move.on_start(obs, MotorCommand())
        obs.robot_state.time_s = 0.1
        move.step(obs, MotorCommand())

        obs.user_input.arm_tracking_enabled = False
        obs.user_input.arm_joint_target = PICO_ARM_HOME_RAD
        obs.robot_state.time_s = 0.2
        command = MotorCommand()
        move.step(obs, command)
        for side in PICO_ARM_SIDES:
            for index, name in enumerate(PICO_ARM_JOINT_NAMES[side]):
                self.assertEqual(
                    command.target_angles[name], PICO_ARM_HOME_RAD[side][index]
                )
        self.assertEqual(command.target_angles["left_shoulder_pitch"], 0.0)
        self.assertNotEqual(
            command.target_angles["left_shoulder_pitch"],
            NEUTRAL_POSE["left_shoulder_pitch"],
        )

    def test_invalid_custom_input_cannot_retain_previous_target(self):
        move = PicoArmTrackingMove(slew_rate_rad_s=1000.0)
        obs = observation(0.0, enabled=True)
        move.on_start(obs, MotorCommand())
        obs.robot_state.time_s = 0.1
        move.step(obs, MotorCommand())

        obs.user_input.arm_joint_target = {
            "left": TARGET["left"],
            "right": (0.0, 0.0),
        }
        obs.robot_state.time_s = 0.2
        command = MotorCommand()
        move.step(obs, command)
        for side in PICO_ARM_SIDES:
            for index, name in enumerate(PICO_ARM_JOINT_NAMES[side]):
                self.assertEqual(
                    command.target_angles[name], PICO_ARM_HOME_RAD[side][index]
                )

    def test_session_loss_slews_to_global_neutral_and_clears_state(self):
        controller = FakeController()
        move = PicoArmTrackingMove(
            controller=controller,
            slew_rate_rad_s=1000.0,
            neutral_return_duration_s=0.1,
        )
        obs = observation(0.0, enabled=True)
        move.on_start(obs, MotorCommand())
        self.assertEqual(controller.kp_writes[-1][1], [KP_RL] * 6)
        obs.robot_state.time_s = 0.1
        move.step(obs, MotorCommand())

        obs.user_input = UserInput()
        for side in PICO_ARM_SIDES:
            for index, name in enumerate(PICO_ARM_JOINT_NAMES[side]):
                obs.robot_state.motor_positions[name] = TARGET[side][index]
        obs.robot_state.time_s = 0.2
        move.state = MoveState.STOPPING
        command = MotorCommand()
        move.on_stop(obs, command)
        self.assertEqual(move.state, MoveState.STOPPING)
        for side in PICO_ARM_SIDES:
            for index, name in enumerate(PICO_ARM_JOINT_NAMES[side]):
                self.assertEqual(command.target_angles[name], TARGET[side][index])
        obs.robot_state.time_s = 0.31
        command = MotorCommand()
        move.on_stop(obs, command)
        self.assertEqual(move.state, MoveState.INACTIVE)
        for name in PICO_ARM_JOINT_ORDER:
            self.assertEqual(command.target_angles[name], NEUTRAL_POSE[name])
        self.assertEqual(controller.kp_writes[-1][1], [KP_DEFAULT] * 6)
        self.assertEqual(move._last_targets, {})

    def test_getup_owns_arms_and_clears_stopping_overlay(self):
        move = PicoArmTrackingMove(slew_rate_rad_s=1000.0)
        obs = observation(0.0, enabled=True, active_moves=("pico_arms", "getup"))
        move.on_start(obs, MotorCommand())

        command = MotorCommand(target_angles={name: 3.0 for name in MOTOR_TO_ID})
        move.step(obs, command)
        self.assertTrue(all(value == 3.0 for value in command.target_angles.values()))

        move.state = MoveState.STOPPING
        move.on_stop(obs, command)
        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(move._last_targets, {})

    def test_safety_resume_discards_pre_fault_target_origin(self):
        move = PicoArmTrackingMove(slew_rate_rad_s=1.0)
        obs = observation(0.0, enabled=True)
        move.on_start(obs, MotorCommand())
        obs.robot_state.time_s = 0.1
        move.step(obs, MotorCommand())

        measured = dict(NEUTRAL_POSE)
        measured["left_shoulder_pitch"] = -0.2
        obs.robot_state.motor_positions = measured
        obs.robot_state.time_s = 10.0
        move.on_safety_resume(obs)
        self.assertEqual(move._last_targets["left_shoulder_pitch"], -0.2)
        self.assertEqual(move._last_time_s, 10.0)


if __name__ == "__main__":
    unittest.main()
