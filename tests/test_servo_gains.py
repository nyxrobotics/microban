"""Servo P gains: one value per regime, sent only when a servo's value changes."""

import unittest

import robot_controller as rc_module
from constants import KP_HARDWARE_NEUTRAL, KP_RL, MOTOR_TO_ID
from test_hardware_joint_offset import make_controller


class RecordingBus:
    """Wrap the FakeBus factory to record every P-gain write."""

    def __init__(self, bus):
        self.writes = []
        original = bus.sync_write_position_p_gain

        def record(ids, gains):
            self.writes.append((list(ids), list(gains)))
            original(ids, gains)

        bus.sync_write_position_p_gain = record


def controller_with_recording():
    recorders = []
    controller, bus = make_controller(configure=lambda bus: recorders.append(RecordingBus(bus)))
    return controller, bus, recorders[0]


class GainValuesTest(unittest.TestCase):
    def test_two_gains_only(self):
        import constants

        self.assertEqual(KP_RL, 125)
        self.assertEqual(KP_HARDWARE_NEUTRAL, 900)
        self.assertFalse(hasattr(constants, "KP_DEFAULT"))
        self.assertFalse(hasattr(rc_module, "KP_DEFAULT"))


class SendOnlyChangedGainsTest(unittest.TestCase):
    def setUp(self):
        self.controller, self.bus, self.recorder = controller_with_recording()
        self.ids = list(MOTOR_TO_ID.values())

    def test_construction_sends_no_gain_and_the_first_write_reaches_every_servo(self):
        # The servos' gains are unknown at startup: nothing is assumed, and
        # main.py's first write goes to every servo.
        self.assertEqual(self.recorder.writes, [])
        self.controller.sync_write_kp(self.ids, [KP_HARDWARE_NEUTRAL] * len(self.ids))
        self.assertEqual(self.recorder.writes, [(self.ids, [KP_HARDWARE_NEUTRAL] * len(self.ids))])
        self.assertEqual(self.bus.kp, {motor_id: KP_HARDWARE_NEUTRAL for motor_id in self.ids})

    def test_same_values_are_not_sent_again(self):
        self.controller.sync_write_kp(self.ids, [KP_RL] * len(self.ids))
        self.controller.sync_write_kp(self.ids, [KP_RL] * len(self.ids))
        self.controller.sync_write_kp(self.ids[:6], [KP_RL] * 6)
        self.assertEqual(len(self.recorder.writes), 1)

    def test_only_the_servos_whose_value_changes_are_sent(self):
        self.controller.sync_write_kp(self.ids, [KP_RL] * len(self.ids))
        arms = self.ids[:6]
        self.controller.sync_write_kp(
            self.ids, [KP_HARDWARE_NEUTRAL if motor_id in arms else KP_RL for motor_id in self.ids]
        )
        self.assertEqual(self.recorder.writes[-1], (arms, [KP_HARDWARE_NEUTRAL] * 6))
        self.assertEqual(self.bus.kp[arms[0]], KP_HARDWARE_NEUTRAL)
        self.assertEqual(self.bus.kp[self.ids[-1]], KP_RL)

    def test_a_failed_write_is_retried_next_time(self):
        def fail(ids, gains):
            raise RuntimeError("bus error")

        original = self.bus.sync_write_position_p_gain
        self.bus.sync_write_position_p_gain = fail
        with self.assertRaises(RuntimeError):
            self.controller.sync_write_kp(self.ids, [KP_RL] * len(self.ids))
        self.bus.sync_write_position_p_gain = original
        self.controller.sync_write_kp(self.ids, [KP_RL] * len(self.ids))
        self.assertEqual(self.recorder.writes[-1], (self.ids, [KP_RL] * len(self.ids)))

    def test_a_rejoining_servo_gets_its_last_gain_again(self):
        motor_id = MOTOR_TO_ID["left_knee"]
        self.controller.sync_write_kp(self.ids, [KP_RL] * len(self.ids))
        self.controller._seed_rejoin_goal(motor_id)
        self.assertEqual(self.recorder.writes[-1], ([motor_id], [KP_RL]))

    def test_a_rejoining_servo_with_no_requested_gain_is_left_alone(self):
        before = len(self.recorder.writes)
        self.controller._seed_rejoin_goal(MOTOR_TO_ID["left_knee"])
        self.assertEqual(len(self.recorder.writes), before)


if __name__ == "__main__":
    unittest.main()
