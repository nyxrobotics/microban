import unittest
import math

import numpy as np

from input.input_source import UserInput
from moves.hmd_head import HmdHeadTrackingMove, JOINT_LIMITS
from moves.move import MotorCommand, MoveState
from observer import Observation, RobotState


def observation(time_s, orientation=None, yaw_front=False, active_moves=None, body_quat=None):
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            body_quat=[1.0, 0.0, 0.0, 0.0] if body_quat is None else body_quat,
            motor_positions={"head": 0.0, "neck_roll": 0.0, "neck_pitch": 0.0},
        ),
        user_input=UserInput(
            active_moves={"hmd_head"} if active_moves is None else set(active_moves),
            head_orientation=orientation,
            head_yaw_front=yaw_front,
        ),
    )


class HmdHeadMoveTest(unittest.TestCase):
    def test_yaw_tracks_hmd_and_trigger_slews_to_front(self):
        move = HmdHeadTrackingMove(slew_rate_rad_s=1.0)
        command = MotorCommand()
        move.on_start(observation(0.0, {"roll": 0.0, "pitch": 0.0, "yaw": 1.0}), command)
        command = MotorCommand()
        move.step(observation(0.5, {"roll": 0.0, "pitch": 0.0, "yaw": 1.0}), command)
        self.assertAlmostEqual(command.target_angles["head"], 0.1)  # dt is safety-capped at 0.1 s
        command = MotorCommand()
        move.step(
            observation(0.6, {"roll": 0.0, "pitch": 0.0, "yaw": 1.0}, yaw_front=True),
            command,
        )
        self.assertAlmostEqual(command.target_angles["head"], 0.0)

    def test_joint_limits_are_enforced(self):
        move = HmdHeadTrackingMove(slew_rate_rad_s=1000.0)
        move.on_start(observation(0.0), MotorCommand())
        command = MotorCommand()
        move.step(observation(0.1, {"roll": 9.0, "pitch": -9.0, "yaw": 9.0}), command)
        self.assertLessEqual(command.target_angles["head"], JOINT_LIMITS["head"][1])
        self.assertLessEqual(command.target_angles["neck_roll"], JOINT_LIMITS["neck_roll"][1])
        self.assertGreaterEqual(command.target_angles["neck_pitch"], JOINT_LIMITS["neck_pitch"][0])

    def test_stop_returns_to_neutral_with_slew_limit(self):
        move = HmdHeadTrackingMove(slew_rate_rad_s=1.0)
        move.on_start(observation(0.0), MotorCommand())
        move._last_targets = {"head": 0.2, "neck_roll": 0.0, "neck_pitch": 0.0}
        move._last_time_s = 0.0
        move.state = MoveState.STOPPING
        command = MotorCommand()
        move.on_stop(observation(0.1), command)
        self.assertAlmostEqual(command.target_angles["head"], 0.1)
        self.assertEqual(move.state, MoveState.STOPPING)

    def test_getup_owns_head_during_stop_and_resume_uses_measured_pose(self):
        move = HmdHeadTrackingMove(slew_rate_rad_s=1.0)
        move.on_start(observation(0.0), MotorCommand())
        move._last_targets = {"head": 0.8, "neck_roll": 0.0, "neck_pitch": 0.0}
        move.state = MoveState.STOPPING

        during_getup = MotorCommand()
        move.on_stop(observation(0.1, active_moves={"getup"}), during_getup)
        self.assertNotIn("head", {
            name: value
            for name, value in during_getup.target_angles.items()
            if value != 0.0
        })

        resumed_obs = observation(0.2, active_moves=set())
        resumed_obs.robot_state.motor_positions["head"] = 0.3
        resumed = MotorCommand()
        move.on_stop(resumed_obs, resumed)
        self.assertAlmostEqual(resumed.target_angles["head"], 0.299, places=6)

    def test_mixed_trunk_tilt_is_compensated_by_matrix_composition(self):
        def rx(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

        def ry(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

        def rz(angle):
            c, s = math.cos(angle), math.sin(angle)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

        body_roll, body_pitch = map(math.radians, (4.0, 3.0))
        half_roll, half_pitch = body_roll / 2.0, body_pitch / 2.0
        body_quat = [
            math.cos(half_pitch) * math.cos(half_roll),
            math.cos(half_pitch) * math.sin(half_roll),
            math.sin(half_pitch) * math.cos(half_roll),
            -math.sin(half_pitch) * math.sin(half_roll),
        ]
        desired_angles = dict(
            zip(("yaw", "roll", "pitch"), map(math.radians, (10.0, 5.0, -10.0)))
        )
        move = HmdHeadTrackingMove(slew_rate_rad_s=1000.0)
        move.on_start(observation(0.0, body_quat=body_quat), MotorCommand())
        command = MotorCommand()
        move.step(observation(0.1, desired_angles, body_quat=body_quat), command)

        trunk = ry(body_pitch) @ rx(body_roll)
        neck = (
            rz(command.target_angles["head"])
            @ rx(command.target_angles["neck_roll"])
            @ ry(command.target_angles["neck_pitch"])
        )
        desired = (
            rz(desired_angles["yaw"])
            @ rx(desired_angles["roll"])
            @ ry(desired_angles["pitch"])
        )
        np.testing.assert_allclose(trunk @ neck, desired, atol=1e-9)

    def test_invalid_imu_quaternion_holds_measured_head(self):
        move = HmdHeadTrackingMove(slew_rate_rad_s=1000.0)
        move.on_start(observation(0.0), MotorCommand())
        obs = observation(
            0.1,
            {"roll": 0.2, "pitch": -0.5, "yaw": 0.4},
            body_quat=[float("nan"), 0.0, 0.0, 0.0],
        )
        obs.robot_state.motor_positions = {
            "head": 0.12,
            "neck_roll": -0.03,
            "neck_pitch": 0.08,
        }
        command = MotorCommand()
        move.step(obs, command)
        for name, measured in obs.robot_state.motor_positions.items():
            self.assertEqual(command.target_angles[name], measured)

    def test_safety_resume_resynchronizes_slew_from_measured_head(self):
        move = HmdHeadTrackingMove(slew_rate_rad_s=1.0)
        move.on_start(observation(0.0), MotorCommand())
        move._last_targets = {"head": 0.8, "neck_roll": 0.0, "neck_pitch": 0.0}
        move.state = MoveState.STOPPING
        obs = observation(10.0, active_moves=set())
        obs.robot_state.motor_positions["head"] = 0.3

        move.on_safety_resume(obs)
        command = MotorCommand()
        move.on_stop(obs, command)
        self.assertAlmostEqual(command.target_angles["head"], 0.299, places=6)


if __name__ == "__main__":
    unittest.main()
