"""Real-robot joint offsets act only at the RobotController servo boundary."""

import math
import pathlib
import unittest
from unittest import mock

import numpy as np

import robot_controller as rc_module
from constants import (
    HARDWARE_JOINT_OFFSET_DEG,
    HARDWARE_JOINT_OFFSET_MAX_RAD,
    HARDWARE_JOINT_OFFSET_RAD,
    MOTOR_SIGN,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def raw_to_rad(raw: int) -> float:
    """rustypot's position mapping: raw 0..4095 is [-pi, pi)."""
    return 2.0 * math.pi * raw / 4096.0 - math.pi


class FakeBus:
    """Records servo-coordinate writes and serves servo-coordinate readings."""

    def __init__(self, **_kwargs) -> None:
        # Distinct raw positions per servo, away from the 2048 centre.
        self.raw_position = {
            motor_id: 2048 + 37 * index - 300 for index, motor_id in enumerate(MOTOR_TO_ID.values())
        }
        self.raw_velocity = {motor_id: 5 * index - 40 for index, motor_id in enumerate(MOTOR_TO_ID.values())}
        self.torque = {motor_id: 0 for motor_id in MOTOR_TO_ID.values()}
        self.goal_writes: list[tuple[int, float]] = []
        self.kp = {}

    def servo_rad(self, motor_id: int) -> float:
        return raw_to_rad(self.raw_position[motor_id])

    def goals_for(self, motor_id: int) -> list[float]:
        return [value for written_id, value in self.goal_writes if written_id == motor_id]

    # Writes -------------------------------------------------------------
    def sync_write_goal_position(self, ids, positions):
        self.goal_writes.extend(zip(ids, positions))

    def sync_write_position_p_gain(self, ids, gains):
        self.kp.update(zip(ids, gains))

    def sync_write_torque_enable(self, ids, values):
        for motor_id, value in zip(ids, values):
            self.torque[motor_id] = 1 if value else 0

    def sync_write_status_return_level(self, ids, levels):
        pass

    # Reads --------------------------------------------------------------
    def sync_read_raw_data(self, ids, address, length):
        assert (address, length) == (128, 8)
        return [
            self.raw_velocity[motor_id].to_bytes(4, "little", signed=True)
            + self.raw_position[motor_id].to_bytes(4, "little", signed=True)
            for motor_id in ids
        ]

    def read_present_position(self, motor_id):
        return [self.servo_rad(motor_id)]

    def read_present_velocity(self, motor_id):
        return [float(self.raw_velocity[motor_id])]

    def read_torque_enable(self, motor_id):
        return [self.torque[motor_id]]

    def sync_read_torque_enable(self, ids):
        return [self.torque[motor_id] for motor_id in ids]

    def read_hardware_error_status(self, motor_id):
        return [0]

    def sync_read_present_current(self, ids):
        return [[100] for _ in ids]

    def sync_read_present_input_voltage(self, ids):
        return [[111] for _ in ids]

    def sync_read_position_p_gain(self, ids):
        return [[self.kp.get(motor_id, 400)] for motor_id in ids]


class FakeIMU:
    def __init__(self, **_kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def make_controller(offsets_deg: dict[str, float] | None = None):
    offsets = None
    if offsets_deg is not None:
        offsets = {name: float(np.deg2rad(offsets_deg.get(name, 0.0))) for name in MOTOR_TO_ID}
    with mock.patch.object(rc_module, "Xl330PyController", FakeBus), \
            mock.patch.object(rc_module, "ThreadedIMUReader", FakeIMU), \
            mock.patch("builtins.print"):
        controller = rc_module.RobotController(joint_offsets_rad=offsets)
    return controller, controller._controller


# left_ankle_pitch (sign +1) -1 deg, right_hip_roll (sign +1) +2 deg,
# right_ankle_pitch (sign -1) +1.5 deg, left_hip_yaw (sign -1) -3 deg.
TEST_OFFSETS_DEG = {
    "left_ankle_pitch": -1.0,
    "right_hip_roll": 2.0,
    "right_ankle_pitch": 1.5,
    "left_hip_yaw": -3.0,
    "neck_pitch": 0.5,
}


def expected_servo_goal(name: str, logical: float) -> float:
    return (logical + math.radians(TEST_OFFSETS_DEG.get(name, 0.0))) * MOTOR_SIGN[name]


def expected_logical(name: str, servo: float) -> float:
    return servo * MOTOR_SIGN[name] - math.radians(TEST_OFFSETS_DEG.get(name, 0.0))


class OffsetTableTest(unittest.TestCase):
    def test_table_lists_every_joint_and_defaults_to_zero(self) -> None:
        self.assertEqual(list(HARDWARE_JOINT_OFFSET_DEG), list(MOTOR_TO_ID))
        self.assertEqual(list(HARDWARE_JOINT_OFFSET_RAD), list(MOTOR_TO_ID))
        for name in MOTOR_TO_ID:
            with self.subTest(name=name):
                self.assertEqual(HARDWARE_JOINT_OFFSET_DEG[name], 0.0)
                self.assertEqual(HARDWARE_JOINT_OFFSET_RAD[name], 0.0)

    def test_invalid_tables_are_rejected(self) -> None:
        zero = {name: 0.0 for name in MOTOR_TO_ID}
        bad_tables = {
            "missing": {k: v for k, v in zero.items() if k != "neck_roll"},
            "unknown": {**zero, "tail": 0.0},
            "non_finite": {**zero, "left_knee": math.nan},
            "too_large": {**zero, "left_knee": HARDWARE_JOINT_OFFSET_MAX_RAD + 1e-6},
        }
        for label, table in bad_tables.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                rc_module.validated_hardware_offsets(table)

    def test_negative_zero_offset_is_normalised(self) -> None:
        checked = rc_module.validated_hardware_offsets({name: -0.0 for name in MOTOR_TO_ID})
        for name, value in checked.items():
            with self.subTest(name=name):
                self.assertEqual(value.hex(), (0.0).hex())

    def test_only_robot_controller_applies_offsets(self) -> None:
        users = sorted(
            path.relative_to(SRC).as_posix()
            for path in SRC.rglob("*.py")
            if "HARDWARE_JOINT_OFFSET" in path.read_text(encoding="utf-8")
        )
        self.assertEqual(users, ["constants.py", "robot_controller.py"])
        for path in (SRC / "sim").rglob("*.py"):
            with self.subTest(path=path.name):
                self.assertNotIn("robot_controller", path.read_text(encoding="utf-8"))

    def test_nonzero_offsets_are_logged_at_startup(self) -> None:
        with mock.patch.object(rc_module, "Xl330PyController", FakeBus), \
                mock.patch.object(rc_module, "ThreadedIMUReader", FakeIMU), \
                mock.patch("builtins.print") as printed:
            rc_module.RobotController(
                joint_offsets_rad={
                    name: math.radians(TEST_OFFSETS_DEG.get(name, 0.0)) for name in MOTOR_TO_ID
                }
            )
        lines = [call.args[0] for call in printed.call_args_list]
        self.assertEqual(len(lines), 1)
        self.assertIn("left_ankle_pitch=-1.000deg", lines[0])
        self.assertIn("right_hip_roll=+2.000deg", lines[0])
        self.assertNotIn("left_knee", lines[0])
        with mock.patch.object(rc_module, "Xl330PyController", FakeBus), \
                mock.patch.object(rc_module, "ThreadedIMUReader", FakeIMU), \
                mock.patch("builtins.print") as printed:
            rc_module.RobotController()
        printed.assert_not_called()


class ZeroOffsetIdentityTest(unittest.TestCase):
    """With every offset 0 the bus traffic is bit-for-bit the old sign-only mapping."""

    def test_goal_writes_match_sign_only_mapping(self) -> None:
        controller, bus = make_controller()
        ids = list(MOTOR_TO_ID.values())
        logical = [NEUTRAL_POSE[name] + 0.01 * index - 0.1 for index, name in enumerate(MOTOR_TO_ID)]
        logical[0] = -0.0
        controller.sync_write_neutral_goal_position(ids, logical)
        for (motor_id, written), value, name in zip(bus.goal_writes, logical, MOTOR_TO_ID):
            with self.subTest(name=name):
                old = value * MOTOR_SIGN[name]
                self.assertEqual(written.hex(), old.hex())
                self.assertEqual(math.copysign(1.0, written), math.copysign(1.0, old))

    def test_reads_match_sign_only_mapping(self) -> None:
        for label, offsets_deg in (
            ("default", None),
            ("typed_negative_zero", {name: -0.0 for name in MOTOR_TO_ID}),
        ):
            with self.subTest(table=label):
                self._check_reads_match_sign_only_mapping(offsets_deg)

    def _check_reads_match_sign_only_mapping(self, offsets_deg) -> None:
        controller, bus = make_controller(offsets_deg)
        ids = list(MOTOR_TO_ID.values())
        # An exact-zero reading on a negative-sign joint must stay -0.0, as
        # the old sign-only mapping produced.
        bus.raw_position[MOTOR_TO_ID["right_knee"]] = 2048
        positions = controller.sync_read_present_position(ids)
        for motor_id, name in zip(ids, MOTOR_TO_ID):
            old = raw_to_rad(bus.raw_position[motor_id]) * MOTOR_SIGN[name]
            with self.subTest(name=name):
                if motor_id // 10 == 5:
                    continue  # head IDs are polled one per tick
                self.assertEqual(positions[ids.index(motor_id)].hex(), old.hex())
                self.assertEqual(controller.read_present_position(motor_id).hex(), old.hex())
        controller._poll_one_head_position()
        controller._poll_one_head_position()
        for motor_id in controller._head_ids:
            name = controller._id_to_name[motor_id]
            with self.subTest(head=name):
                old = raw_to_rad(bus.raw_position[motor_id]) * MOTOR_SIGN[name]
                self.assertEqual(controller._last_positions[motor_id].hex(), old.hex())


class NonzeroOffsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller, self.bus = make_controller(TEST_OFFSETS_DEG)
        self.ids = list(MOTOR_TO_ID.values())

    def test_goal_writes_add_offset_and_cache_logical(self) -> None:
        logical = [NEUTRAL_POSE[name] + 0.02 for name in MOTOR_TO_ID]
        # Neutral fallback path (writes even while stale).
        self.controller.sync_write_neutral_goal_position(self.ids, logical)
        # Normal policy path once readings are live.
        self.controller.sync_read_present_position(self.ids)
        self.controller.sync_write_torque_enable(self.ids[:2], [True, True])
        self.bus.goal_writes.clear()
        body_ids = [motor_id for motor_id in self.ids if motor_id // 10 != 5]
        body_targets = [logical[self.ids.index(motor_id)] - 0.05 for motor_id in body_ids]
        self.controller.sync_write_goal_position(body_ids, body_targets)
        written = dict(self.bus.goal_writes)
        for motor_id, target in zip(body_ids, body_targets):
            name = self.controller._id_to_name[motor_id]
            with self.subTest(name=name):
                self.assertAlmostEqual(written[motor_id], expected_servo_goal(name, target), places=12)
                self.assertEqual(self.controller.last_goal_targets[name], target)
        self.assertAlmostEqual(
            written[MOTOR_TO_ID["left_ankle_pitch"]],
            body_targets[body_ids.index(MOTOR_TO_ID["left_ankle_pitch"])] - math.radians(1.0),
            places=12,
        )
        # Negative MOTOR_SIGN: servo = -(logical + offset).
        right_ankle = MOTOR_TO_ID["right_ankle_pitch"]
        self.assertAlmostEqual(
            written[right_ankle],
            -(body_targets[body_ids.index(right_ankle)] + math.radians(1.5)),
            places=12,
        )

    def test_every_read_path_subtracts_offset(self) -> None:
        positions = dict(zip(self.ids, self.controller.sync_read_present_position(self.ids)))
        for motor_id in self.ids:
            name = self.controller._id_to_name[motor_id]
            expected = expected_logical(name, self.bus.servo_rad(motor_id))
            with self.subTest(path="sync_raw", name=name):
                if motor_id // 10 != 5:
                    self.assertAlmostEqual(positions[motor_id], expected, places=12)
            with self.subTest(path="single", name=name):
                self.assertAlmostEqual(self.controller.read_present_position(motor_id), expected, places=12)
        for _ in self.controller._head_ids:
            self.controller._poll_one_head_position()
        for motor_id in self.controller._head_ids:
            name = self.controller._id_to_name[motor_id]
            with self.subTest(path="head_poll", name=name):
                self.assertAlmostEqual(
                    self.controller._last_positions[motor_id],
                    expected_logical(name, self.bus.servo_rad(motor_id)),
                    places=12,
                )

    def test_velocity_is_unaffected(self) -> None:
        controller_zero, bus_zero = make_controller()
        self.controller.sync_read_present_position(self.ids)
        controller_zero.sync_read_present_position(self.ids)
        self.assertEqual(
            self.controller.sync_read_present_velocity(self.ids),
            controller_zero.sync_read_present_velocity(self.ids),
        )
        for motor_id in self.ids:
            self.assertEqual(
                self.controller.read_present_velocity(motor_id),
                controller_zero.read_present_velocity(motor_id),
            )

    def test_pending_enable_seed_commands_measured_physical_pose(self) -> None:
        # Both MOTOR_SIGN values: the offset must be added before the sign.
        for name in ("left_ankle_pitch", "right_ankle_pitch", "left_hip_yaw"):
            with self.subTest(name=name):
                controller, bus = make_controller(TEST_OFFSETS_DEG)
                motor_id = MOTOR_TO_ID[name]
                # Stale at start: ON is deferred until a position reply arrives.
                controller.sync_write_torque_enable([motor_id], [True])
                self.assertIn(motor_id, controller._pending_enable)
                controller.sync_read_present_position(self.ids)
                self.assertNotIn(motor_id, controller._pending_enable)
                goals = bus.goals_for(motor_id)
                self.assertEqual(len(goals), 1)
                self.assertAlmostEqual(goals[0], bus.servo_rad(motor_id), places=12)
                self.assertAlmostEqual(
                    controller.last_goal_targets[name],
                    expected_logical(name, bus.servo_rad(motor_id)),
                    places=12,
                )
                self.assertEqual(bus.torque[motor_id], 1)

    def test_head_poll_applies_offset_before_sign(self) -> None:
        # The neck signs are CAD-inferred; pretend neck_roll turns out to be -1
        # so the order of offset and sign is observable on the head poll path.
        motor_id = MOTOR_TO_ID["neck_roll"]
        offset = math.radians(2.0)
        self.controller._id_to_sign[motor_id] = -1.0
        self.controller._id_to_offset[motor_id] = offset
        for _ in self.controller._head_ids:
            self.controller._poll_one_head_position()
        self.assertAlmostEqual(
            self.controller._last_positions[motor_id],
            -self.bus.servo_rad(motor_id) - offset,
            places=12,
        )
        self.assertAlmostEqual(
            self.controller._to_servo(motor_id, self.controller._last_positions[motor_id]),
            self.bus.servo_rad(motor_id),
            places=12,
        )

    def test_read_core_refuses_positions(self) -> None:
        # Positions must go through _from_servo; the generic register reader
        # would skip the offset.
        with self.assertRaises(ValueError):
            self.controller._read_core(
                "position", "sync_read_present_position", {}, lambda _m, v: float(v)
            )

    def test_torque_rejoin_seed_writes_reading_back_unchanged(self) -> None:
        self.controller.sync_read_present_position(self.ids)
        for name in ("right_hip_roll", "right_ankle_pitch", "left_hip_yaw"):
            motor_id = MOTOR_TO_ID[name]
            self.controller.sync_write_torque_enable([motor_id], [True])
            # The servo browns out and comes back with torque OFF at a new pose.
            self.bus.torque[motor_id] = 0
            self.bus.raw_position[motor_id] += 111
            self.bus.goal_writes.clear()
            self.controller._check_torque_state(motor_id)
            with self.subTest(name=name):
                self.assertEqual(self.bus.torque[motor_id], 1)
                self.assertEqual(self.bus.goals_for(motor_id), [self.bus.servo_rad(motor_id)])
                logical = expected_logical(name, self.bus.servo_rad(motor_id))
                self.assertAlmostEqual(self.controller.last_goal_targets[name], logical, places=12)
                self.assertAlmostEqual(self.controller._last_positions[motor_id], logical, places=12)
                # The next logical goal equal to the cached one re-commands the
                # same physical pose (rejoin slew starts from the measured goal).
                self.bus.goal_writes.clear()
                self.controller.sync_write_goal_position([motor_id], [logical])
                self.assertAlmostEqual(
                    self.bus.goals_for(motor_id)[0], self.bus.servo_rad(motor_id), places=12
                )

    def test_all_joint_enable_transition_seeds_measured_physical_pose(self) -> None:
        with mock.patch("builtins.print"):
            self.controller.sync_write_torque_enable(self.ids, [True] * len(self.ids))
        for motor_id in self.ids:
            name = self.controller._id_to_name[motor_id]
            with self.subTest(name=name):
                self.assertEqual(self.bus.torque[motor_id], 1)
                goals = self.bus.goals_for(motor_id)
                self.assertTrue(goals)
                self.assertEqual(goals[-1], self.bus.servo_rad(motor_id))
                self.assertAlmostEqual(
                    self.controller.last_goal_targets[name],
                    expected_logical(name, self.bus.servo_rad(motor_id)),
                    places=12,
                )

    def test_logical_round_trip_keeps_servo_position(self) -> None:
        for motor_id in self.ids:
            servo = self.bus.servo_rad(motor_id)
            logical = self.controller._from_servo(motor_id, servo)
            with self.subTest(motor_id=motor_id):
                self.assertAlmostEqual(self.controller._to_servo(motor_id, logical), servo, places=12)


if __name__ == "__main__":
    unittest.main()
