import math
import tempfile
import unittest
from pathlib import Path

from constants import NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from input.input_source import UserInput
from input.network_input import NetworkInputSource
from moves.move import MotorCommand, Move, MoveState
from moves.policy_selector import PolicySelectableWalkMove
from moves.walk import WalkMove
from observer import Observation, RobotState


class FakeMove(Move):
    def __init__(
        self,
        *,
        stop_ticks=1,
        preload_error: Exception | None = None,
        start_error: Exception | None = None,
        step_error: Exception | None = None,
        marker=0.0,
    ):
        super().__init__()
        self.start_count = 0
        self.step_count = 0
        self.stop_count = 0
        self.stop_ticks = stop_ticks
        self.preload_error = preload_error
        self.start_error = start_error
        self.step_error = step_error
        self.marker = marker
        self.last_velocity = None

    def preload(self):
        if self.preload_error is not None:
            raise self.preload_error

    def on_start(self, obs, command):
        self.start_count += 1
        command.target_angles["probe"] = self.marker
        if self.start_error is not None:
            raise self.start_error
        self.state = MoveState.ACTIVE

    def step(self, obs, command):
        self.step_count += 1
        self.last_velocity = dict(obs.user_input.velocity)
        command.target_angles["probe"] = self.marker + obs.user_input.velocity["vx"]
        if self.step_error is not None:
            raise self.step_error

    def on_stop(self, obs, command):
        self.stop_count += 1
        if self.stop_count >= self.stop_ticks:
            self.state = MoveState.INACTIVE


def observation(policy="walk", active=True, vx=0.0, degraded=False):
    return Observation(
        robot_state=RobotState(time_s=0.0),
        user_input=UserInput(
            active_moves={"walk"} if active else set(),
            locomotion_policy=policy,
            learned_policy_degraded=degraded,
            velocity={"vx": vx, "vy": 0.0, "vtheta": 0.0},
        ),
    )


def bridge_pico_packet(seq: int, *, trigger_held: bool) -> dict:
    """Current bridge wire shape: policy is fixed; only deadman state changes."""

    return {
        "version": 1,
        "session_id": "trigger-only-bridge",
        "seq": seq,
        "active_moves": ["walk", "hmd_head"] if trigger_held else [],
        "locomotion_policy": "pico_teleop",
        "velocity": {
            "vx": 0.4 if trigger_held else 0.0,
            "vy": 0.0,
            "vtheta": 0.0,
        },
        "head_orientation": (
            {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            if trigger_held
            else None
        ),
        "head_yaw_front": False,
        "body_target_contract": "microban_pico_offsets_v2_both_feet_stationary",
        "body_target_safety_margin": 0.8,
        "foot_target": (
            {"left": [0.01, 0.0, 0.02], "right": [0.0, 0.0, 0.0]}
            if trigger_held
            else None
        ),
        "hand_target": (
            {"left": [0.01, 0.0, 0.0], "right": [-0.01, 0.0, 0.0]}
            if trigger_held
            else None
        ),
    }


class PolicySelectorTest(unittest.TestCase):
    def test_bridge_packet_to_selector_is_trigger_only_without_x(self):
        source = NetworkInputSource(stale_after_s=0.5)
        released_packet = bridge_pico_packet(0, trigger_held=False)
        self.assertNotIn("primary_button", released_packet)
        source._apply(released_packet)
        released = source.read()
        self.assertEqual(released.locomotion_policy, "pico_teleop")
        self.assertNotIn("walk", released.active_moves)

        source._apply(bridge_pico_packet(1, trigger_held=True))
        held = source.read()
        self.assertEqual(held.locomotion_policy, "pico_teleop")
        self.assertIn("walk", held.active_moves)

        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.on_start(
            Observation(robot_state=RobotState(time_s=0.0), user_input=held),
            MotorCommand(),
        )
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.start_count, 1)
        self.assertEqual(walk.start_count, 0)

    def test_selects_exactly_one_policy_per_activation(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.on_start(observation("pico_teleop"), MotorCommand())
        selector.step(observation("pico_teleop"), MotorCommand())
        self.assertEqual(pico.start_count, 1)
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(walk.start_count, 0)
        self.assertEqual(selector.effective_policy, "pico_teleop")

    def test_missing_policy_keeps_requested_joystick_on_legacy(self):
        walk = FakeMove(marker=10.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = PolicySelectableWalkMove(
                pico_policy_path=Path(temp_dir) / "missing.onnx",
                legacy_move=walk,
            )
            selector.preload()
            command = MotorCommand()
            obs = observation("pico_teleop", vx=0.65)
            selector.on_start(obs, command)
            selector.step(obs, command)

        self.assertEqual(selector.effective_policy, "walk")
        self.assertTrue(selector.fallback_latched)
        self.assertEqual(walk.start_count, 1)
        self.assertEqual(walk.step_count, 1)
        self.assertEqual(walk.last_velocity["vx"], 0.65)
        self.assertEqual(command.target_angles["probe"], 10.65)
        self.assertIn("not installed", selector.fallback_reason)

    def test_preload_failure_does_not_disable_legacy(self):
        walk = FakeMove()
        pico = FakeMove(preload_error=ValueError("bad metadata"))
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.preload()
        selector.on_start(observation("pico_teleop"), MotorCommand())
        self.assertEqual(selector.effective_policy, "walk")
        self.assertIn("bad metadata", selector.fallback_reason)

    def test_start_failure_falls_back_without_losing_activation(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(start_error=RuntimeError("cannot initialize"), marker=20.0)
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        command = MotorCommand()
        obs = observation("pico_teleop", vx=-0.4)

        selector.on_start(obs, command)

        self.assertEqual(selector.state, MoveState.ACTIVE)
        self.assertEqual(selector.effective_policy, "walk")
        self.assertEqual(walk.start_count, 1)
        self.assertEqual(command.target_angles["probe"], 10.0)
        self.assertIn("cannot initialize", selector.fallback_reason)

    def test_inference_failure_starts_and_steps_legacy_in_same_cycle(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(step_error=RuntimeError("non-finite output"), marker=20.0)
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        obs = observation("pico_teleop", vx=0.55)
        selector.on_start(obs, MotorCommand())
        command = MotorCommand()

        selector.step(obs, command)

        self.assertEqual(selector.state, MoveState.ACTIVE)
        self.assertEqual(selector.effective_policy, "walk")
        self.assertTrue(selector.fallback_latched)
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(walk.start_count, 1)
        self.assertEqual(walk.step_count, 1)
        self.assertEqual(walk.last_velocity["vx"], 0.55)
        self.assertEqual(command.target_angles["probe"], 10.55)
        self.assertIn("non-finite output", selector.fallback_reason)

    def test_inference_failure_runs_real_walk_actor_in_same_cycle(self):
        walk = WalkMove(controller=None)
        pico = FakeMove(step_error=RuntimeError("intentional learned failure"))
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = PolicySelectableWalkMove(
                pico_policy_path=Path(temp_dir) / "absent.onnx",
                legacy_move=walk,
                pico_move=pico,
            )
            robot_state = RobotState(
                time_s=0.0,
                gyro=[0.0, 0.0, 0.0],
                projected_gravity=[0.0, 0.0, -1.0],
                motor_positions={
                    name: float(value) for name, value in NEUTRAL_POSE.items()
                },
                motor_velocities={name: 0.0 for name in NEUTRAL_POSE},
            )
            obs = Observation(
                robot_state=robot_state,
                user_input=UserInput(
                    active_moves={"walk"},
                    locomotion_policy="pico_teleop",
                    velocity={"vx": 0.1, "vy": 0.0, "vtheta": 0.0},
                ),
            )
            selector.on_start(obs, MotorCommand())
            command = MotorCommand()

            selector.step(obs, command)

        targets = [
            float(command.target_angles[name]) for name in OBSERVATION_DOF_ORDER
        ]
        self.assertEqual(selector.state, MoveState.ACTIVE)
        self.assertEqual(selector.effective_policy, "walk")
        self.assertTrue(selector.fallback_latched)
        self.assertEqual(pico.step_count, 1)
        self.assertTrue(all(math.isfinite(value) for value in targets))
        self.assertTrue(any(abs(float(value)) > 0.0 for value in walk._last_action))
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            expected = walk._default_pose[name] + float(walk._last_action[index])
            self.assertAlmostEqual(command.target_angles[name], expected, places=7)
        self.assertIn("intentional learned failure", selector.fallback_reason)

    def test_tracker_degradation_falls_back_then_latches_until_release(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.on_start(observation("pico_teleop", vx=0.2), MotorCommand())

        degraded_command = MotorCommand()
        selector.step(observation("walk", vx=0.3, degraded=True), degraded_command)
        selector.step(observation("pico_teleop", vx=0.4), MotorCommand())

        self.assertEqual(selector.effective_policy, "walk")
        self.assertTrue(selector.fallback_latched)
        self.assertEqual(walk.step_count, 2)
        self.assertEqual(pico.step_count, 0)
        self.assertEqual(degraded_command.target_angles["probe"], 10.3)

        selector.on_stop(observation("pico_teleop", active=False), MotorCommand())
        self.assertEqual(selector.state, MoveState.INACTIVE)
        selector.on_start(observation("pico_teleop", vx=0.5), MotorCommand())
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.start_count, 2)

    def test_degraded_learned_request_latches_existing_walk_until_release(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.on_start(observation("walk"), MotorCommand())

        selector.step(observation("walk", vx=0.3, degraded=True), MotorCommand())
        selector.step(observation("pico_teleop", vx=0.4), MotorCommand())

        self.assertTrue(selector.fallback_latched)
        self.assertEqual(selector.effective_policy, "walk")
        self.assertEqual(walk.step_count, 2)
        self.assertEqual(pico.start_count, 0)

        selector.on_stop(observation("pico_teleop", active=False), MotorCommand())
        selector.on_start(observation("pico_teleop", vx=0.5), MotorCommand())
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.start_count, 1)

    def test_wire_policy_change_switches_without_stopping_joystick(self):
        walk = FakeMove()
        pico = FakeMove()
        selector = PolicySelectableWalkMove(legacy_move=walk, pico_move=pico)
        selector.on_start(observation("walk"), MotorCommand())
        selector.step(observation("pico_teleop", vx=0.7), MotorCommand())
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.start_count, 1)
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(pico.last_velocity["vx"], 0.7)

        selector.step(observation("walk", vx=-0.6), MotorCommand())
        self.assertEqual(selector.effective_policy, "walk")
        self.assertFalse(selector.fallback_latched)
        self.assertEqual(walk.start_count, 2)
        self.assertEqual(walk.step_count, 1)
        self.assertEqual(walk.last_velocity["vx"], -0.6)

        selector.step(observation("pico_teleop", vx=0.25), MotorCommand())
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.start_count, 2)
        self.assertEqual(pico.step_count, 2)
        self.assertEqual(pico.last_velocity["vx"], 0.25)

    def test_atomic_replacement_can_reload_without_process_restart(self):
        walk = FakeMove()
        replacement = FakeMove(marker=30.0)
        attempts = []

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pico.onnx"
            path.write_bytes(b"bad")

            def factory(_controller, received_path):
                attempts.append(received_path.read_bytes())
                if received_path.read_bytes() == b"bad":
                    raise ValueError("rejected contract")
                return replacement

            selector = PolicySelectableWalkMove(
                pico_policy_path=path,
                legacy_move=walk,
                learned_move_factory=factory,
            )
            selector.preload()
            selector.on_start(observation("pico_teleop"), MotorCommand())
            self.assertEqual(selector.effective_policy, "walk")

            path.write_bytes(b"valid replacement with another fingerprint")
            self.assertTrue(selector.request_learned_reload())
            assert selector._reload_thread is not None
            selector._reload_thread.join(timeout=1.0)
            selector.on_stop(observation("pico_teleop", active=False), MotorCommand())
            selector.on_start(observation("pico_teleop"), MotorCommand())

        self.assertEqual(attempts[0], b"bad")
        self.assertEqual(attempts[-1], b"valid replacement with another fingerprint")
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(replacement.start_count, 1)
        self.assertIn("rejected contract", selector.fallback_reason)


if __name__ == "__main__":
    unittest.main()
