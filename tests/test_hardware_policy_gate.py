import unittest

from constants import KP_DEFAULT, MOTOR_TO_ID, NEUTRAL_POSE
from input.gamepad_input import GamepadInputSource, XBOX_BUTTONS
from input.input_source import UserInput
from moves.move import Move, MoveState
from observer import Observation, RobotState
from scheduler import Scheduler


class FakeController:
    def __init__(self):
        self.torque_writes = []
        self.kp_writes = []
        self.goal_writes = []

    def sync_write_torque_enable(self, ids, values):
        self.torque_writes.append((list(ids), list(values)))

    def sync_write_kp(self, ids, values):
        self.kp_writes.append((list(ids), list(values)))

    def sync_write_goal_position(self, ids, values):
        self.goal_writes.append((list(ids), list(values)))


class CountingMove(Move):
    def __init__(self):
        super().__init__()
        self.state = MoveState.ACTIVE
        self.steps = 0

    def step(self, _obs, _command):
        self.steps += 1


def observation(*, torque, policy, time_s=0.0, offset=0.4):
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            motor_positions={
                name: neutral + offset for name, neutral in NEUTRAL_POSE.items()
            },
            motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
            gyro=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            body_quat=[1.0, 0.0, 0.0, 0.0],
            projected_gravity=[0.0, 0.0, -1.0],
        ),
        user_input=UserInput(
            active_moves={"walk", "hmd_head", "pico_arms"},
            torque_enabled=torque,
            policy_enabled=policy,
        ),
    )


class HardwarePolicyGateTest(unittest.TestCase):
    def scheduler(self):
        controller = FakeController()
        move = CountingMove()
        scheduler = Scheduler(
            controller=controller,
            moves={"walk": move},
            hardware_power_control=True,
        )
        return scheduler, controller, move

    def test_b_disables_all_joint_torque_once_and_resets_moves(self):
        scheduler, controller, move = self.scheduler()
        obs = observation(torque=False, policy=False)

        mode, command = scheduler._apply_hardware_gate(
            obs, allow_initial_enable=True
        )
        scheduler._apply_hardware_gate(obs, allow_initial_enable=True)

        self.assertEqual(mode, "limp")
        self.assertEqual(command.target_angles, {})
        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(len(controller.torque_writes), 1)
        ids, values = controller.torque_writes[0]
        self.assertEqual(ids, list(MOTOR_TO_ID.values()))
        self.assertEqual(values, [False] * len(MOTOR_TO_ID))

    def test_a_enables_torque_and_slews_all_joints_without_policy(self):
        scheduler, controller, move = self.scheduler()
        scheduler._apply_hardware_gate(
            observation(torque=False, policy=False),
            allow_initial_enable=True,
        )
        obs = observation(torque=True, policy=False, time_s=1.0)

        mode, command = scheduler._apply_hardware_gate(
            obs, allow_initial_enable=True
        )

        self.assertEqual(mode, "neutral")
        self.assertEqual(set(command.target_angles), set(MOTOR_TO_ID))
        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(controller.torque_writes[-1][1], [True] * len(MOTOR_TO_ID))
        self.assertEqual(controller.kp_writes[-1][1], [KP_DEFAULT] * len(MOTOR_TO_ID))
        self.assertEqual(
            controller.goal_writes[0][1],
            [obs.robot_state.motor_positions[name] for name in MOTOR_TO_ID],
        )
        for name, measured in obs.robot_state.motor_positions.items():
            target = command.target_angles[name]
            self.assertLess(target, measured)
            self.assertGreater(target, NEUTRAL_POSE[name])

    def test_r3_requires_a_then_toggles_policy_and_neutral(self):
        scheduler, controller, move = self.scheduler()

        # A direct true/true startup packet cannot skip the A state.
        mode, _ = scheduler._apply_hardware_gate(
            observation(torque=True, policy=True),
            allow_initial_enable=True,
        )
        self.assertEqual(mode, "neutral")

        mode, _ = scheduler._apply_hardware_gate(
            observation(torque=True, policy=False, time_s=0.02),
            allow_initial_enable=True,
        )
        self.assertEqual(mode, "neutral")

        mode, command = scheduler._apply_hardware_gate(
            observation(torque=True, policy=True, time_s=0.04),
            allow_initial_enable=True,
        )
        self.assertEqual(mode, "policy")
        self.assertIsNone(command)

        move.state = MoveState.ACTIVE
        mode, command = scheduler._apply_hardware_gate(
            observation(torque=True, policy=False, time_s=0.06),
            allow_initial_enable=True,
        )
        self.assertEqual(mode, "neutral")
        self.assertEqual(set(command.target_angles), set(MOTOR_TO_ID))
        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(
            controller.goal_writes[-1][1],
            [
                NEUTRAL_POSE[name] + 0.4
                for name in MOTOR_TO_ID
            ],
        )

    def test_unsafe_imu_cannot_enable_torque_from_limp(self):
        scheduler, controller, _move = self.scheduler()
        mode, command = scheduler._apply_hardware_gate(
            observation(torque=True, policy=False),
            allow_initial_enable=False,
        )

        self.assertEqual(mode, "limp")
        self.assertEqual(command.target_angles, {})
        self.assertEqual(controller.torque_writes[-1][1], [False] * len(MOTOR_TO_ID))


class GamepadHardwareButtonsTest(unittest.TestCase):
    def test_b_a_r3_state_machine(self):
        source = GamepadInputSource(device_path="/dev/null")
        self.assertTrue(source.controls_motor_power)
        self.assertFalse(source.read().torque_enabled)
        self.assertFalse(source.read().policy_enabled)

        source._handle_button(XBOX_BUTTONS["R3"])
        self.assertFalse(source.read().policy_enabled)

        source._handle_button(XBOX_BUTTONS["A"])
        state = source.read()
        self.assertTrue(state.torque_enabled)
        self.assertFalse(state.policy_enabled)

        source._handle_button(XBOX_BUTTONS["R3"])
        self.assertTrue(source.read().policy_enabled)
        source._handle_button(XBOX_BUTTONS["R3"])
        self.assertFalse(source.read().policy_enabled)

        source._handle_button(XBOX_BUTTONS["B"])
        state = source.read()
        self.assertFalse(state.torque_enabled)
        self.assertFalse(state.policy_enabled)


if __name__ == "__main__":
    unittest.main()
