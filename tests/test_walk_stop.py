import unittest

from constants import NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from input.input_source import UserInput
from moves.move import MotorCommand, MoveState
from moves.walk import WalkMove
from observer import Observation, RobotState


class WalkStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.move = WalkMove(controller=None, neutral_return_duration_s=0.8)

    def setUp(self):
        self.move._stop_start_time_s = None
        self.move._stop_start_angles = {}
        self.move.state = MoveState.STOPPING
        self.positions = {
            name: NEUTRAL_POSE[name] + 0.2 for name in OBSERVATION_DOF_ORDER
        }

    def obs(self, time_s):
        return Observation(robot_state=RobotState(time_s=time_s, motor_positions=self.positions))

    def test_first_stop_tick_holds_measured_pose_then_smoothly_returns(self):
        first = MotorCommand()
        self.move.on_stop(self.obs(10.0), first)
        for name in OBSERVATION_DOF_ORDER:
            self.assertAlmostEqual(first.target_angles[name], self.positions[name])

        halfway = MotorCommand()
        self.move.on_stop(self.obs(10.4), halfway)
        for name in OBSERVATION_DOF_ORDER:
            expected = NEUTRAL_POSE[name] + 0.1
            self.assertAlmostEqual(halfway.target_angles[name], expected)

        final = MotorCommand()
        self.move.on_stop(self.obs(10.8), final)
        for name in OBSERVATION_DOF_ORDER:
            self.assertAlmostEqual(final.target_angles[name], NEUTRAL_POSE[name])
        self.assertEqual(self.move.state, MoveState.INACTIVE)

    def test_getup_cancels_return_without_gain_side_effect(self):
        obs = self.obs(10.0)
        obs.user_input = UserInput(active_moves={"getup"})
        command = MotorCommand()
        self.move.on_stop(obs, command)
        self.assertEqual(self.move.state, MoveState.INACTIVE)

    def test_fall_debounce_window_holds_measured_pose(self):
        obs = self.obs(10.0)
        obs.robot_state.projected_gravity = [0.0, 0.0, 0.0]
        command = MotorCommand()
        self.move.step(obs, command)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(command.target_angles[name], self.positions[name])

    def test_safety_resume_restarts_stop_from_current_measurement(self):
        self.move._stop_start_time_s = 1.0
        self.move._stop_start_angles = {
            name: NEUTRAL_POSE[name] - 0.4 for name in OBSERVATION_DOF_ORDER
        }
        obs = self.obs(10.0)
        self.move.on_safety_resume(obs)
        command = MotorCommand()
        self.move.on_stop(obs, command)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(command.target_angles[name], self.positions[name])


if __name__ == "__main__":
    unittest.main()
