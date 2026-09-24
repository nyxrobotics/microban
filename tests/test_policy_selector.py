import tempfile
import unittest
from pathlib import Path

from input.input_source import UserInput
from moves.move import MotorCommand, Move, MoveState
from moves.policy_selector import PolicySelectableWalkMove
from observer import Observation, RobotState


class FakeMove(Move):
    def __init__(self, stop_ticks=1):
        super().__init__()
        self.start_count = 0
        self.step_count = 0
        self.stop_count = 0
        self.stop_ticks = stop_ticks

    def on_start(self, obs, command):
        self.start_count += 1
        self.state = MoveState.ACTIVE

    def step(self, obs, command):
        self.step_count += 1

    def on_stop(self, obs, command):
        self.stop_count += 1
        if self.stop_count >= self.stop_ticks:
            self.state = MoveState.INACTIVE


def observation(policy="walk", active=True):
    return Observation(
        robot_state=RobotState(time_s=0.0),
        user_input=UserInput(
            active_moves={"walk"} if active else set(),
            locomotion_policy=policy,
        ),
    )


class PolicySelectorTest(unittest.TestCase):
    def test_selects_exactly_one_policy_per_activation(self):
        walk = FakeMove()
        pico = FakeMove()
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.on_start(observation("pico_teleop"), MotorCommand())
        selector.step(observation("pico_teleop"), MotorCommand())
        self.assertEqual(pico.start_count, 1)
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(walk.start_count, 0)

    def test_unexpected_live_change_stops_before_other_policy_can_start(self):
        walk = FakeMove(stop_ticks=2)
        pico = FakeMove()
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.on_start(observation("walk"), MotorCommand())
        selector.step(observation("pico_teleop"), MotorCommand())
        self.assertEqual(selector.state, MoveState.STOPPING)
        self.assertEqual(walk.stop_count, 1)
        self.assertEqual(pico.start_count, 0)
        selector.on_stop(observation("pico_teleop", active=False), MotorCommand())
        self.assertEqual(selector.state, MoveState.INACTIVE)
        selector.on_start(observation("pico_teleop"), MotorCommand())
        self.assertEqual(pico.start_count, 1)

    def test_missing_pico_policy_fails_only_when_requested(self):
        walk = FakeMove()
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = PolicySelectableWalkMove(
                pico_policy_path=Path(temp_dir) / "missing.onnx",
                legacy_move=walk,
            )
            selector.preload()
            selector.on_start(observation("walk"), MotorCommand())
            selector.on_stop(observation("walk", active=False), MotorCommand())
            with self.assertRaises(RuntimeError):
                selector.on_start(observation("pico_teleop"), MotorCommand())


if __name__ == "__main__":
    unittest.main()
