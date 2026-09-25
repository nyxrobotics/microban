"""Serial-bus hold behavior at the scheduler boundary (no hardware needed)."""

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from constants import MOTOR_TO_ID, NEUTRAL_POSE
from input.input_source import UserInput
from moves.move import Move, MoveState
from observer import Observer, RobotState
from scheduler import Scheduler


def valid_state():
    return RobotState(
        gyro=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        body_quat=[1.0, 0.0, 0.0, 0.0],
        projected_gravity=[0.0, 0.0, -1.0],
        motor_positions={name: angle for name, angle in NEUTRAL_POSE.items()},
        motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
    )


class ScriptedObserver:
    def __init__(self, results, stop_path, events):
        self.results = list(results)
        self.stop_path = Path(stop_path)
        self.events = events
        self.reads = 0

    def read_state(self, _dt):
        if self.reads >= len(self.results):
            raise AssertionError("scheduler read beyond the scripted ticks")
        result = self.results[self.reads]
        self.reads += 1
        self.events.append(
            ("read_error" if isinstance(result, RuntimeError) else "read_ok", self.reads)
        )
        if self.reads == len(self.results):
            self.stop_path.touch()
        if isinstance(result, RuntimeError):
            raise result
        return result


class ScriptedInput:
    def __init__(self, snapshots, events):
        self.snapshots = list(snapshots)
        self.events = events
        self.reads = 0
        self.stopped = False

    def start(self):
        self.events.append(("input_start", None))

    def stop(self):
        self.stopped = True
        self.events.append(("input_stop", None))

    def read(self):
        if self.reads >= len(self.snapshots):
            raise AssertionError("scheduler read input beyond the scripted ticks")
        snapshot = self.snapshots[self.reads]
        self.reads += 1
        self.events.append(("input", self.reads))
        return snapshot

    def set_motion_inhibited(self, _inhibited):
        pass


class FakeController:
    def __init__(self, events, fail_goal_attempts=(), fail_torque_off_attempts=()):
        self.events = events
        self.fail_goal_attempts = set(fail_goal_attempts)
        self.fail_torque_off_attempts = set(fail_torque_off_attempts)
        self.goal_attempts = []
        self.goal_writes = []
        self.torque_writes = []
        self.torque_off_attempts = 0
        self.kp_writes = []

    def sync_write_goal_position(self, ids, positions):
        write = (list(ids), list(positions))
        self.goal_attempts.append(write)
        attempt = len(self.goal_attempts)
        self.events.append(("goal_attempt", attempt))
        if attempt in self.fail_goal_attempts:
            raise RuntimeError("simulated serial write timeout")
        self.goal_writes.append(write)
        self.events.append(("goal_success", attempt))

    def sync_write_torque_enable(self, ids, enabled):
        write = (list(ids), list(enabled))
        if not any(enabled):
            self.torque_off_attempts += 1
            self.events.append(("torque_off_attempt", self.torque_off_attempts))
            if self.torque_off_attempts in self.fail_torque_off_attempts:
                raise RuntimeError("simulated torque-off timeout")
        self.torque_writes.append(write)
        self.events.append(("torque", all(enabled)))

    def sync_write_kp(self, ids, values):
        self.kp_writes.append((list(ids), list(values)))

    def shutdown(self):
        self.events.append(("shutdown", None))


class AdvancingMove(Move):
    def __init__(self):
        super().__init__()
        self.state = MoveState.ACTIVE
        self.steps = 0

    def step(self, obs, command):
        self.steps += 1
        command.target_angles = dict(obs.robot_state.motor_positions)
        command.target_angles["head"] += 0.005 * self.steps


class MalformedReadController:
    def __init__(self, positions, velocities):
        self.positions = positions
        self.velocities = velocities

    def sync_read_present_position(self, _ids):
        return self.positions

    def sync_read_present_velocity(self, _ids):
        return self.velocities


def input_snapshot(*, torque=True, policy=True):
    return UserInput(
        active_moves={"advance"},
        torque_enabled=torque,
        policy_enabled=policy,
    )


class SchedulerSerialHoldTest(unittest.TestCase):
    def test_incomplete_or_nonfinite_bus_feedback_is_a_read_error(self):
        count = len(MOTOR_TO_ID)
        cases = {
            "missing position": ([0.0] * (count - 1), [0.0] * count),
            "missing velocity": ([0.0] * count, [0.0] * (count - 1)),
            "nonfinite position": ([math.nan] + [0.0] * (count - 1), [0.0] * count),
            "nonfinite velocity": ([0.0] * count, [math.inf] + [0.0] * (count - 1)),
        }
        for label, (positions, velocities) in cases.items():
            with self.subTest(label=label):
                observer = Observer(MalformedReadController(positions, velocities))
                with self.assertRaises(RuntimeError):
                    observer.read_state(0.02)

    def test_repeated_read_failures_hold_last_goal_and_resume_with_paced_ticks(self):
        events = []
        controller = FakeController(events)
        move = AdvancingMove()
        results = (
            [valid_state()]
            + [RuntimeError("serial timeout") for _ in range(4)]
            + [valid_state()]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "stop"
            source = ScriptedInput([input_snapshot() for _ in results], events)
            scheduler = Scheduler(
                frequency_hz=10.0,
                controller=controller,
                stop_flag_path=str(stop_path),
                input_source=source,
                moves={"advance": move},
                serial_hold_on_error=True,
            )
            observer = ScriptedObserver(results, stop_path, events)
            scheduler.observer = observer
            with redirect_stdout(io.StringIO()), patch(
                "scheduler.time.sleep",
                side_effect=lambda seconds: events.append(("sleep", seconds)),
            ):
                scheduler.run()

        self.assertEqual(observer.reads, 6)
        self.assertEqual(source.reads, 6, "input must still be polled on failed read ticks")
        self.assertTrue(source.stopped)
        self.assertEqual(move.steps, 2, "failed observations must not advance a move")
        self.assertEqual(len(controller.goal_writes), 2, "failed observations must not send goals")
        first_head = NEUTRAL_POSE["head"] + 0.005
        resumed_head = NEUTRAL_POSE["head"] + 0.010
        self.assertEqual(
            [entry["head"] for entry in scheduler._cmd_history],
            [first_head] * (scheduler._cmd_history.maxlen - 1) + [resumed_head],
            "each missed sample keeps the last sent target in delayed-current history",
        )
        self.assertEqual(
            controller.torque_writes,
            [(list(MOTOR_TO_ID.values()), [False] * len(MOTOR_TO_ID))],
        )

        # Each failed observation must be rate-limited before another bus read.
        error_indices = [i for i, event in enumerate(events) if event[0] == "read_error"]
        for error_index in error_indices:
            next_read_index = next(
                i for i in range(error_index + 1, len(events)) if events[i][0].startswith("read_")
            )
            self.assertIn(
                "sleep", [event[0] for event in events[error_index + 1 : next_read_index]]
            )

    def test_b_disables_torque_during_failed_read_without_new_goal(self):
        events = []
        controller = FakeController(events)
        move = AdvancingMove()
        results = [valid_state(), valid_state(), valid_state(), RuntimeError("serial timeout")]
        snapshots = [
            input_snapshot(torque=True, policy=False),  # A: enable torque and neutral return
            input_snapshot(torque=True, policy=True),   # R3: start policy move
            input_snapshot(torque=True, policy=True),   # policy move sends a goal
            input_snapshot(torque=False, policy=False), # B or link timeout during bad read
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "stop"
            source = ScriptedInput(snapshots, events)
            scheduler = Scheduler(
                frequency_hz=10.0,
                controller=controller,
                stop_flag_path=str(stop_path),
                input_source=source,
                moves={"advance": move},
                hardware_power_control=True,
                serial_hold_on_error=True,
            )
            scheduler.observer = ScriptedObserver(results, stop_path, events)
            with redirect_stdout(io.StringIO()), patch("scheduler.time.sleep"):
                scheduler.run()

        self.assertEqual(source.reads, 4)
        self.assertEqual(move.steps, 1)
        error_index = events.index(("read_error", 4))
        shutdown_index = events.index(("shutdown", None))
        self.assertNotIn(
            "goal_attempt",
            [event[0] for event in events[error_index:shutdown_index]],
            "a bad observation must not produce a new goal",
        )
        self.assertEqual(controller.torque_writes[0][1], [True] * len(MOTOR_TO_ID))
        self.assertEqual(controller.torque_writes[1][1], [False] * len(MOTOR_TO_ID))
        b_input_index = events.index(("input", 4))
        torque_off_index = events.index(("torque", False))
        self.assertLess(b_input_index, torque_off_index)
        self.assertLess(torque_off_index, shutdown_index, "B must work before cleanup")

    def test_failed_goal_write_keeps_last_successful_history_and_recovers(self):
        events = []
        controller = FakeController(events, fail_goal_attempts={2})
        move = AdvancingMove()
        results = [valid_state(), valid_state(), valid_state(), valid_state()]

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "stop"
            source = ScriptedInput([input_snapshot() for _ in results], events)
            scheduler = Scheduler(
                frequency_hz=10.0,
                controller=controller,
                stop_flag_path=str(stop_path),
                input_source=source,
                moves={"advance": move},
                serial_hold_on_error=True,
            )
            scheduler.observer = ScriptedObserver(results, stop_path, events)
            with redirect_stdout(io.StringIO()), patch("scheduler.time.sleep"):
                scheduler.run()

        self.assertEqual(source.reads, 4)
        self.assertEqual(move.steps, 3)
        self.assertEqual(len(controller.goal_attempts), 4)
        self.assertEqual(len(controller.goal_writes), 3)
        self.assertEqual(
            [write[1][list(MOTOR_TO_ID).index("head")] for write in controller.goal_writes],
            [
                NEUTRAL_POSE["head"] + 0.005,
                NEUTRAL_POSE["head"] + 0.005,
                NEUTRAL_POSE["head"] + 0.015,
            ],
        )
        self.assertEqual(
            [entry["head"] for entry in scheduler._cmd_history],
            [
                NEUTRAL_POSE["head"] + 0.005,
                NEUTRAL_POSE["head"] + 0.005,
                NEUTRAL_POSE["head"] + 0.005,
                NEUTRAL_POSE["head"] + 0.015,
            ],
            "history must retain the last sent target, not the failed new command",
        )

    def test_failed_initial_seed_goal_does_not_enable_torque_and_retries(self):
        events = []
        controller = FakeController(events, fail_goal_attempts={1})
        results = [valid_state(), valid_state(), valid_state()]
        snapshots = [
            input_snapshot(torque=True, policy=False),
            input_snapshot(torque=True, policy=False),
            input_snapshot(torque=True, policy=True),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "stop"
            source = ScriptedInput(snapshots, events)
            scheduler = Scheduler(
                frequency_hz=10.0,
                controller=controller,
                stop_flag_path=str(stop_path),
                input_source=source,
                moves={"advance": AdvancingMove()},
                hardware_power_control=True,
                serial_hold_on_error=True,
            )
            observer = ScriptedObserver(results, stop_path, events)
            scheduler.observer = observer
            with redirect_stdout(io.StringIO()), patch("scheduler.time.sleep"):
                scheduler.run()

        self.assertEqual(observer.reads, 3)
        self.assertEqual(source.reads, 3)
        self.assertEqual(len(controller.goal_attempts), 4)
        self.assertEqual(len(controller.goal_writes), 3)
        first_goal_success = events.index(("goal_success", 2))
        first_torque_on = events.index(("torque", True))
        self.assertLess(first_goal_success, first_torque_on)
        self.assertEqual(
            controller.goal_writes[0][1],
            [NEUTRAL_POSE[name] for name in MOTOR_TO_ID],
        )

    def test_failed_neutral_slew_retries_seed_before_advancing(self):
        events = []
        controller = FakeController(events, fail_goal_attempts={2})
        state = valid_state()
        state.motor_positions = {
            name: angle + 0.4 for name, angle in NEUTRAL_POSE.items()
        }
        results = [state, state, state]

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "stop"
            source = ScriptedInput(
                [input_snapshot(torque=True, policy=False) for _ in results], events
            )
            scheduler = Scheduler(
                frequency_hz=10.0,
                controller=controller,
                stop_flag_path=str(stop_path),
                input_source=source,
                moves={},
                hardware_power_control=True,
                serial_hold_on_error=True,
            )
            scheduler.observer = ScriptedObserver(results, stop_path, events)
            with redirect_stdout(io.StringIO()), patch("scheduler.time.sleep"):
                scheduler.run()

        self.assertEqual(len(controller.goal_attempts), 4)
        self.assertEqual(controller.goal_attempts[2], controller.goal_attempts[0])
        self.assertEqual(controller.goal_writes[1], controller.goal_writes[0])
        self.assertLess(controller.goal_writes[2][1][0], controller.goal_writes[1][1][0])

    def test_failed_b_torque_off_is_retried_on_next_bad_read(self):
        events = []
        controller = FakeController(events, fail_torque_off_attempts={1})
        results = [
            valid_state(),
            valid_state(),
            RuntimeError("serial timeout"),
            RuntimeError("serial timeout"),
        ]
        snapshots = [
            input_snapshot(torque=True, policy=False),
            input_snapshot(torque=True, policy=True),
            input_snapshot(torque=False, policy=False),
            input_snapshot(torque=False, policy=False),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "stop"
            source = ScriptedInput(snapshots, events)
            scheduler = Scheduler(
                frequency_hz=10.0,
                controller=controller,
                stop_flag_path=str(stop_path),
                input_source=source,
                moves={"advance": AdvancingMove()},
                hardware_power_control=True,
                serial_hold_on_error=True,
            )
            observer = ScriptedObserver(results, stop_path, events)
            scheduler.observer = observer
            with redirect_stdout(io.StringIO()), patch("scheduler.time.sleep"):
                scheduler.run()

        self.assertEqual(observer.reads, 4)
        self.assertEqual(source.reads, 4)
        self.assertGreaterEqual(controller.torque_off_attempts, 2)
        first_off = events.index(("torque_off_attempt", 1))
        second_off = events.index(("torque_off_attempt", 2))
        shutdown = events.index(("shutdown", None))
        self.assertLess(first_off, second_off)
        self.assertLess(second_off, shutdown, "B must be retried before cleanup")
        self.assertLess(second_off, events.index(("torque", False)))


if __name__ == "__main__":
    unittest.main()
