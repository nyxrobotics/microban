import unittest

from constants import KP_DEFAULT, KP_RL, MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from input.input_source import UserInput
from moves.getup import GetupMove
from moves.move import MotorCommand, Move, MoveState
from observer import Observation, RobotState


class FakeController:
    def __init__(self):
        self.kp_writes = []

    def sync_write_kp(self, ids, gains):
        self.kp_writes.append((list(ids), list(gains)))


class FakeTorqueController(FakeController):
    def __init__(self):
        super().__init__()
        self.torque_writes = []

    def sync_write_torque_enable(self, ids, values):
        self.torque_writes.append((list(ids), list(values)))


def transition_only_move(controller):
    move = GetupMove.__new__(GetupMove)
    Move.__init__(move)
    move._controller = controller
    move._joint_names = list(MOTOR_TO_ID)
    move._neck_joint_names = [
        name for name in move._joint_names if name not in OBSERVATION_DOF_ORDER
    ]
    move._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
    move._last_torque_enabled = None
    move._recovery_targets = None
    move._recovery_last_time_s = None
    return move


def observation(active_moves=(), *, torque_enabled=True, getup_armed=True, time_s=0.0):
    measured = {name: angle + 0.01 for name, angle in NEUTRAL_POSE.items()}
    velocities = {name: 0.0 for name in NEUTRAL_POSE}
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            motor_positions=measured,
            motor_velocities=velocities,
            gyro=[0.0, 0.0, 0.0],
            projected_gravity=[0.0, 0.0, -1.0],
        ),
        user_input=UserInput(
            active_moves=set(active_moves),
            torque_enabled=torque_enabled,
            getup_armed=getup_armed,
        ),
    )


class GetupTransitionTest(unittest.TestCase):
    def test_start_holds_measured_pose_and_owns_rl_gain(self):
        controller = FakeController()
        move = transition_only_move(controller)
        obs = observation({"getup"})
        command = MotorCommand()
        move.on_start(obs, command)

        self.assertEqual(move.state, MoveState.ACTIVE)
        self.assertEqual(command.target_angles, obs.robot_state.motor_positions)
        gains_by_name = dict(zip(move._joint_names, controller.kp_writes[-1][1]))
        for name in move._joint_names:
            expected = KP_RL if name in OBSERVATION_DOF_ORDER else KP_DEFAULT
            self.assertEqual(gains_by_name[name], expected)

    def test_stop_holds_pose_and_hands_policy_gains_to_walk(self):
        controller = FakeController()
        move = transition_only_move(controller)
        move.state = MoveState.STOPPING
        obs = observation({"walk", "hmd_head"})
        command = MotorCommand()
        move.on_stop(obs, command)

        self.assertEqual(move.state, MoveState.INACTIVE)
        self.assertEqual(command.target_angles, obs.robot_state.motor_positions)
        gains_by_name = dict(zip(move._joint_names, controller.kp_writes[-1][1]))
        for name in move._joint_names:
            expected = KP_RL if name in OBSERVATION_DOF_ORDER else KP_DEFAULT
            self.assertEqual(gains_by_name[name], expected)

    def test_safety_resume_discards_action_history_and_restarts_with_hold(self):
        controller = FakeController()
        move = transition_only_move(controller)
        move.state = MoveState.ACTIVE
        move._last_action = [0.5] * len(OBSERVATION_DOF_ORDER)
        move._last_torque_enabled = True
        move._recovery_targets = {"head": 0.3}
        obs = observation({"getup"})

        move.on_safety_resume(obs)
        self.assertEqual(move.state, MoveState.STARTING)
        self.assertEqual(move._last_action, [0.0] * len(OBSERVATION_DOF_ORDER))
        self.assertIsNone(move._last_torque_enabled)
        self.assertIsNone(move._recovery_targets)
        command = MotorCommand()
        move.on_start(obs, command)
        self.assertEqual(command.target_angles, obs.robot_state.motor_positions)

    def test_step_runs_the_real_policy_and_holds_the_neck(self):
        # Loads the actual onnx (not the transition-only bypass above): a
        # mismatch between build_observation()'s vector and what the policy
        # was actually trained/exported for must fail here, not at runtime.
        move = GetupMove(controller=None)
        obs = observation({"getup"})
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)

        for name in OBSERVATION_DOF_ORDER:
            self.assertIn(name, command.target_angles)
        for name in move._neck_joint_names:
            self.assertEqual(
                command.target_angles[name],
                obs.robot_state.motor_positions[name],
            )
        self.assertEqual(len(move._last_action), len(OBSERVATION_DOF_ORDER))

    def test_limp_cuts_torque_once_and_commands_nothing(self):
        controller = FakeTorqueController()
        move = transition_only_move(controller)
        obs = observation({"getup"}, torque_enabled=False, getup_armed=False)
        command = MotorCommand()

        move.step(obs, command)
        move.step(obs, command)

        # LIMP touches nothing: MotorCommand starts at NEUTRAL_POSE and step()
        # must leave it there rather than writing anything while torque is off.
        self.assertEqual(command.target_angles, dict(NEUTRAL_POSE))
        # Only the transition writes torque_enable, not every tick.
        self.assertEqual(len(controller.torque_writes), 1)
        ids, values = controller.torque_writes[0]
        self.assertEqual(set(values), {False})
        self.assertEqual(len(ids), len(move._joint_names))

    def test_recover_slews_toward_neutral_without_running_the_policy(self):
        controller = FakeTorqueController()
        move = transition_only_move(controller)
        # Start well off neutral so the slew has somewhere to go, and torque
        # was off a moment ago so the recovery-from-limp path engages.
        move._last_torque_enabled = False
        far = {name: angle + 0.5 for name, angle in NEUTRAL_POSE.items()}
        obs = Observation(
            robot_state=RobotState(
                time_s=0.0,
                motor_positions=far,
                motor_velocities={name: 0.0 for name in NEUTRAL_POSE},
                gyro=[0.0, 0.0, 0.0],
                projected_gravity=[0.0, 0.0, -1.0],
            ),
            user_input=UserInput(
                active_moves={"getup"}, torque_enabled=True, getup_armed=False
            ),
        )
        command = MotorCommand()
        move.step(obs, command)

        # Torque re-enabled exactly once on this transition.
        self.assertEqual(len(controller.torque_writes), 1)
        self.assertEqual(set(controller.torque_writes[0][1]), {True})
        # Moving toward NEUTRAL_POSE, not snapped there and not left at the
        # far starting point -- and never touching the policy (last_action
        # stays all zero).
        for name in OBSERVATION_DOF_ORDER:
            target = command.target_angles[name]
            self.assertNotEqual(target, far[name])
            self.assertLess(abs(target - NEUTRAL_POSE[name]), abs(0.5))
        self.assertEqual(move._last_action, [0.0] * len(OBSERVATION_DOF_ORDER))

    def test_arming_then_disarming_returns_to_recovery_without_double_torque_write(self):
        # Real GetupMove (loads the actual onnx): the armed tick must go
        # through the real policy, not the transition-only bypass.
        controller = FakeTorqueController()
        move = GetupMove(controller=controller)
        obs_armed = observation({"getup"}, torque_enabled=True, getup_armed=True)
        obs_disarmed = observation({"getup"}, torque_enabled=True, getup_armed=False)
        command = MotorCommand()

        move.step(obs_armed, command)
        move.step(obs_disarmed, command)

        # torque_enabled itself never changed across these two ticks, so no
        # second torque_enable write -- only arming state changed.
        self.assertEqual(len(controller.torque_writes), 1)


if __name__ == "__main__":
    unittest.main()
