import math
import tempfile
import unittest

from constants import MOTOR_TO_ID, NEUTRAL_POSE
from input.input_source import UserInput
from moves.move import Move, MoveState
from observer import RobotState
from scheduler import Scheduler


def valid_state():
    return RobotState(
        gyro=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        body_quat=[1.0, 0.0, 0.0, 0.0],
        projected_gravity=[0.0, 0.0, -1.0],
        motor_positions={name: angle + 0.03 for name, angle in NEUTRAL_POSE.items()},
        motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
    )


class FakeController:
    def __init__(self, imu_status=None):
        self.imu_status = imu_status
        self.goal_writes = []
        self.torque_writes = []
        self.shutdown_called = False

    def get_imu_status(self):
        return dict(self.imu_status)

    def sync_write_goal_position(self, ids, positions):
        self.goal_writes.append((list(ids), list(positions)))

    def sync_write_torque_enable(self, ids, enabled):
        self.torque_writes.append((list(ids), list(enabled)))

    def shutdown(self):
        self.shutdown_called = True


class FakeObserver:
    def __init__(self, state):
        self.state = state

    def read_state(self, _dt):
        return self.state


class FakeInput:
    def __init__(self):
        self.inhibits = []
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def read(self):
        return UserInput(
            active_moves={"walk"},
            velocity={"vx": 1.0, "vy": 0.0, "vtheta": 0.0},
        )

    def set_motion_inhibited(self, inhibited):
        self.inhibits.append(inhibited)


class CountingMove(Move):
    def __init__(self):
        super().__init__()
        self.state = MoveState.ACTIVE
        self.steps = 0

    def step(self, _obs, _command):
        self.steps += 1


class SchedulerSafetyTest(unittest.TestCase):
    def scheduler(self, **kwargs):
        return Scheduler(controller=FakeController(), moves={}, **kwargs)

    def test_imu_validation_checks_status_and_all_policy_vectors(self):
        scheduler = self.scheduler()
        state = valid_state()
        self.assertTrue(
            scheduler._imu_is_safe(state, {"valid": True, "age_s": 0.02})[0]
        )
        # Simulation controllers intentionally have no freshness-status method.
        self.assertTrue(scheduler._imu_is_safe(state, None)[0])

        bad_cases = []
        bad_cases.append((state, {"valid": False, "age_s": 0.0}))
        bad_cases.append((state, {"valid": "false", "age_s": 0.0}))
        bad_cases.append((state, {"valid": True, "age_s": 0.2}))
        nan_gyro = valid_state()
        nan_gyro.gyro[0] = math.nan
        bad_cases.append((nan_gyro, {"valid": True, "age_s": 0.0}))
        zero_quat = valid_state()
        zero_quat.quat = [0.0, 0.0, 0.0, 0.0]
        bad_cases.append((zero_quat, {"valid": True, "age_s": 0.0}))
        nan_gravity = valid_state()
        nan_gravity.projected_gravity[2] = math.nan
        bad_cases.append((nan_gravity, {"valid": True, "age_s": 0.0}))

        for bad_state, status in bad_cases:
            with self.subTest(status=status, state=bad_state):
                self.assertFalse(scheduler._imu_is_safe(bad_state, status)[0])

    def test_measured_hold_requires_all_finite_joint_feedback(self):
        state = valid_state()
        command = Scheduler._measured_hold_command(state)
        self.assertIsNotNone(command)
        self.assertEqual(command.target_angles, state.motor_positions)
        state.motor_positions["head"] = math.nan
        self.assertIsNone(Scheduler._measured_hold_command(state))

    def test_persistent_imu_fault_holds_then_disables_torque_without_dispatch(self):
        controller = FakeController({"valid": False, "age_s": 1.0})
        input_source = FakeInput()
        move = CountingMove()
        with tempfile.TemporaryDirectory() as temp_dir:
            scheduler = Scheduler(
                frequency_hz=200.0,
                controller=controller,
                stop_flag_path=f"{temp_dir}/stop",
                input_source=input_source,
                moves={"walk": move},
                imu_shutdown_after_s=0.02,
            )
            state = valid_state()
            scheduler.observer = FakeObserver(state)
            scheduler.run()

        self.assertGreater(len(controller.goal_writes), 0)
        for ids, positions in controller.goal_writes:
            self.assertEqual(ids, list(MOTOR_TO_ID.values()))
            self.assertEqual(positions, [state.motor_positions[name] for name in MOTOR_TO_ID])
        self.assertEqual(move.steps, 0)
        self.assertTrue(input_source.inhibits)
        self.assertTrue(all(input_source.inhibits))
        self.assertTrue(input_source.stopped)
        self.assertTrue(controller.shutdown_called)
        self.assertTrue(controller.torque_writes)
        self.assertEqual(controller.torque_writes[-1][1], [False] * len(MOTOR_TO_ID))


if __name__ == "__main__":
    unittest.main()
