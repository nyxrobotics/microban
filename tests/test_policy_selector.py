import math
import tempfile
import unittest
from pathlib import Path

from constants import KP_HARDWARE_NEUTRAL, KP_RL, MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from input.input_source import UserInput
from input.network_input import NetworkInputSource
from moves.move import MotorCommand, Move, MoveState
from moves.policy_selector import PolicySelectableWalkMove
from moves.walk import WalkMove
from observer import Observation, RobotState
from policy_contract import PolicyContractError
from policy_fixtures import LinearWalkSession
from test_getup_transitions import observation as getup_observation
from test_getup_transitions import transition_only_move


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


def observation(policy="walk", active=True, vx=0.0):
    return Observation(
        robot_state=RobotState(time_s=0.0),
        user_input=UserInput(
            active_moves={"walk"} if active else set(),
            locomotion_policy=policy,
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
        "torque_enabled": True,
        "policy_enabled": True,
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
        source = NetworkInputSource()
        released_packet = bridge_pico_packet(0, trigger_held=False)
        self.assertNotIn("primary_button", released_packet)
        source._apply(released_packet)
        released = source.read()
        # Released (no left-trigger walk request) with R3 still enabled keeps
        # the selected PICO standing actor active at zero velocity (see
        # NetworkInputSource's balance_only).
        self.assertEqual(released.locomotion_policy, "pico_teleop")
        self.assertTrue(released.balance_only)
        self.assertIn("walk", released.active_moves)
        self.assertEqual(released.velocity, {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})

        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
        selector.on_start(
            Observation(robot_state=RobotState(time_s=0.0), user_input=released),
            MotorCommand(),
        )
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.start_count, 1)
        self.assertEqual(walk.start_count, 0)

        source._apply(bridge_pico_packet(1, trigger_held=True))
        held = source.read()
        self.assertEqual(held.locomotion_policy, "pico_teleop")
        self.assertIn("walk", held.active_moves)
        selector.step(
            Observation(robot_state=RobotState(time_s=0.0), user_input=held),
            MotorCommand(),
        )
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(walk.start_count, 0)

    def test_selects_exactly_one_policy_per_activation(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
        selector.on_start(observation("pico_teleop"), MotorCommand())
        selector.step(observation("pico_teleop"), MotorCommand())
        self.assertEqual(pico.start_count, 1)
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(walk.start_count, 0)
        self.assertEqual(selector.effective_policy, "pico_teleop")

    def test_missing_policy_keeps_requested_joystick_on_fallback(self):
        walk = FakeMove(marker=10.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = PolicySelectableWalkMove(
                pico_policy_path=Path(temp_dir) / "missing.onnx",
                fallback_move=walk,
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

    def test_preload_failure_does_not_disable_fallback(self):
        walk = FakeMove()
        pico = FakeMove(preload_error=ValueError("bad metadata"))
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
        selector.preload()
        selector.on_start(observation("pico_teleop"), MotorCommand())
        self.assertEqual(selector.effective_policy, "walk")
        self.assertIn("bad metadata", selector.fallback_reason)

    def test_contract_rejection_uses_existing_walk_fallback(self):
        walk = FakeMove(marker=10.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pico_teleop.onnx"
            path.write_bytes(b"present but not a contract policy")

            def reject_contract(_controller, _path):
                raise PolicyContractError(
                    "policy contract validation failed"
                )

            selector = PolicySelectableWalkMove(
                pico_policy_path=path,
                fallback_move=walk,
                learned_move_factory=reject_contract,
            )
            selector.preload()
            selector.on_start(
                observation("pico_teleop", vx=0.4),
                MotorCommand(),
            )

        self.assertEqual(selector.effective_policy, "walk")
        self.assertTrue(selector.fallback_latched)
        self.assertEqual(walk.start_count, 1)
        self.assertIn("policy contract validation", selector.fallback_reason)

    def test_start_failure_falls_back_without_losing_activation(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(start_error=RuntimeError("cannot initialize"), marker=20.0)
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
        command = MotorCommand()
        obs = observation("pico_teleop", vx=-0.4)

        selector.on_start(obs, command)

        self.assertEqual(selector.state, MoveState.ACTIVE)
        self.assertEqual(selector.effective_policy, "walk")
        self.assertEqual(walk.start_count, 1)
        self.assertEqual(command.target_angles["probe"], 10.0)
        self.assertIn("cannot initialize", selector.fallback_reason)

    def test_inference_failure_starts_and_steps_fallback_in_same_cycle(self):
        walk = FakeMove(marker=10.0)
        pico = FakeMove(step_error=RuntimeError("non-finite output"), marker=20.0)
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
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
        walk = WalkMove(controller=None, session=LinearWalkSession())
        pico = FakeMove(step_error=RuntimeError("intentional learned failure"))
        with tempfile.TemporaryDirectory() as temp_dir:
            selector = PolicySelectableWalkMove(
                pico_policy_path=Path(temp_dir) / "absent.onnx",
                fallback_move=walk,
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

    def test_missing_optional_body_targets_keep_pico_actor_and_joystick(self):
        source = NetworkInputSource()
        source._apply(bridge_pico_packet(0, trigger_held=False))
        source._apply(bridge_pico_packet(1, trigger_held=True))
        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
        selector.on_start(
            Observation(robot_state=RobotState(time_s=0.0), user_input=source.read()),
            MotorCommand(),
        )

        missing_targets = bridge_pico_packet(2, trigger_held=True)
        missing_targets["foot_target"] = None
        source._apply(missing_targets)
        continued = source.read()
        self.assertEqual(continued.locomotion_policy, "pico_teleop")
        self.assertFalse(continued.learned_policy_degraded)
        self.assertIsNone(continued.foot_target)
        # hand_target is untouched here (only foot_target was cleared in this
        # packet) and is independent of the foot channel.
        self.assertEqual(
            continued.hand_target,
            {"left": (0.01, 0.0, 0.0), "right": (-0.01, 0.0, 0.0)},
        )
        command = MotorCommand()
        selector.step(
            Observation(robot_state=RobotState(time_s=0.0), user_input=continued),
            command,
        )

        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertFalse(selector.fallback_latched)
        self.assertEqual(walk.start_count, 0)
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(pico.last_velocity["vx"], 0.4)
        self.assertEqual(command.target_angles["probe"], 20.4)

        source._apply(bridge_pico_packet(3, trigger_held=True))
        selector.step(
            Observation(robot_state=RobotState(time_s=0.0), user_input=source.read()),
            MotorCommand(),
        )
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(pico.step_count, 2)
        self.assertEqual(walk.start_count, 0)

    def test_bad_body_target_contract_does_not_latch_fallback_during_handoff(self):
        source = NetworkInputSource()
        source._apply(bridge_pico_packet(0, trigger_held=False))
        invalid = bridge_pico_packet(1, trigger_held=True)
        invalid["body_target_contract"] = "wrong"
        source._apply(invalid)
        continued = source.read()
        self.assertEqual(continued.locomotion_policy, "pico_teleop")
        self.assertFalse(continued.learned_policy_degraded)
        self.assertIsNone(continued.foot_target)
        self.assertIsNone(continued.hand_target)

        walk = FakeMove(marker=10.0)
        pico = FakeMove(marker=20.0)
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
        selector.on_start(observation("walk"), MotorCommand())
        selector.step(
            Observation(robot_state=RobotState(time_s=0.0), user_input=continued),
            MotorCommand(),
        )

        self.assertFalse(selector.fallback_latched)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertEqual(walk.start_count, 1)
        self.assertEqual(walk.step_count, 0)
        self.assertEqual(pico.start_count, 1)
        self.assertEqual(pico.step_count, 1)
        self.assertEqual(pico.last_velocity["vx"], 0.4)

    def test_wire_policy_change_switches_without_stopping_joystick(self):
        walk = FakeMove()
        pico = FakeMove()
        selector = PolicySelectableWalkMove(fallback_move=walk, pico_move=pico)
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
                fallback_move=walk,
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


class _KpRecorder:
    def __init__(self):
        self.kp_writes = []
        self.goal_writes = []

    def sync_write_kp(self, ids, gains):
        self.kp_writes.append((list(ids), list(gains)))

    def sync_write_goal_position(self, ids, positions):
        self.goal_writes.append((list(ids), list(positions)))


class HoldGainTest(unittest.TestCase):
    """A static hold is P900 on all 21 joints (docs/policies.md), whatever it follows."""

    ALL_IDS = list(MOTOR_TO_ID.values())

    def _selector(self, controller, pico_move=None):
        return PolicySelectableWalkMove(
            controller=controller,
            pico_move=pico_move,
            pico_policy_path=Path(tempfile.gettempdir()) / "missing_pico_policy.onnx",
        )

    def _obs(self, policy="pico_teleop", time_s=0.0, active=True):
        return Observation(
            robot_state=RobotState(
                time_s=time_s,
                motor_positions={name: NEUTRAL_POSE[name] + 0.01 for name in MOTOR_TO_ID},
            ),
            user_input=UserInput(
                active_moves={"walk"} if active else set(), locomotion_policy=policy
            ),
        )

    @staticmethod
    def _gains(controller):
        gains = {}
        for ids, values in controller.kp_writes:
            gains.update(zip(ids, values))
        return gains

    def test_hold_after_getup_raises_every_joint_after_reseeding_goals(self):
        controller = _KpRecorder()
        getup = transition_only_move(controller)
        getup.on_stop(getup_observation(active_moves=("walk",)), MotorCommand())
        self.assertEqual(set(self._gains(controller).values()), {KP_RL})
        move = self._selector(controller)
        written = len(controller.kp_writes)
        move.on_start(self._obs(), MotorCommand())
        self.assertEqual(move.effective_policy, "hold")
        self.assertEqual(len(controller.kp_writes), written)  # goals are sent first
        move.step(self._obs(), MotorCommand())
        ids = [MOTOR_TO_ID[name] for name in OBSERVATION_DOF_ORDER]
        self.assertEqual(controller.goal_writes[-1][0], ids)
        self.assertEqual(controller.kp_writes[-1], (self.ALL_IDS, [KP_HARDWARE_NEUTRAL] * 21))
        # head, neck_roll and neck_pitch included (P125 lets the neck sag)
        self.assertEqual(set(self._gains(controller).values()), {KP_HARDWARE_NEUTRAL})
        move.step(self._obs(), MotorCommand())
        self.assertEqual(len(controller.kp_writes), written + 1)

    def test_hold_after_a_pico_failure_raises_every_joint(self):
        controller = _KpRecorder()
        pico = FakeMove(step_error=RuntimeError("inference failed"))
        move = self._selector(controller, pico_move=pico)
        move.on_start(self._obs(), MotorCommand())
        self.assertEqual(move.effective_policy, "pico_teleop")
        controller.kp_writes.append((self.ALL_IDS, [KP_RL] * 21))  # PICO's start gain
        move.step(self._obs(), MotorCommand())
        self.assertEqual(move.effective_policy, "hold")
        self.assertEqual(set(self._gains(controller).values()), {KP_HARDWARE_NEUTRAL})

    def test_hold_returning_to_neutral_ends_at_p900_on_every_joint(self):
        controller = _KpRecorder()
        move = self._selector(controller)
        move.on_start(self._obs(), MotorCommand())
        move.step(self._obs(), MotorCommand())
        controller.kp_writes.clear()
        move.on_stop(self._obs(active=False), MotorCommand())
        move.on_stop(self._obs(time_s=5.0, active=False), MotorCommand())
        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(controller.kp_writes, [(self.ALL_IDS, [KP_HARDWARE_NEUTRAL] * 21)])


if __name__ == "__main__":
    unittest.main()
