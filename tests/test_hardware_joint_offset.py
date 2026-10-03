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
    SERVO_GOAL_MAX_RAD,
    SERVO_GOAL_MIN_RAD,
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
        # EEPROM configuration: factory Position Control mode and limits.
        self.operating_mode = {motor_id: 3 for motor_id in MOTOR_TO_ID.values()}
        self.raw_min_limit = {motor_id: 0 for motor_id in MOTOR_TO_ID.values()}
        self.raw_max_limit = {motor_id: 4095 for motor_id in MOTOR_TO_ID.values()}
        # Remaining failed replies per ID for the configuration reads.
        self.config_failures: dict[int, int] = {}
        self.config_reads = 0

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

    def _config_read(self, motor_id, table):
        self.config_reads += 1
        if self.config_failures.get(motor_id, 0) > 0:
            self.config_failures[motor_id] -= 1
            raise RuntimeError("Timeout")
        return [table[motor_id]]

    def read_operating_mode(self, motor_id):
        return self._config_read(motor_id, self.operating_mode)

    def read_raw_min_position_limit(self, motor_id):
        return self._config_read(motor_id, self.raw_min_limit)

    def read_raw_max_position_limit(self, motor_id):
        return self._config_read(motor_id, self.raw_max_limit)

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


def make_controller(offsets_deg: dict[str, float] | None = None, configure=None, printed=None):
    """Build a RobotController on a FakeBus; ``configure(bus)`` edits it first."""
    offsets = None
    if offsets_deg is not None:
        offsets = {name: float(np.deg2rad(offsets_deg.get(name, 0.0))) for name in MOTOR_TO_ID}

    def bus_factory(**kwargs):
        bus = FakeBus(**kwargs)
        if configure is not None:
            configure(bus)
        return bus

    with mock.patch.object(rc_module, "Xl330PyController", bus_factory), \
            mock.patch.object(rc_module, "ThreadedIMUReader", FakeIMU), \
            mock.patch("builtins.print") as print_mock:
        try:
            controller = rc_module.RobotController(joint_offsets_rad=offsets)
        finally:
            if printed is not None:
                printed.extend(call.args[0] for call in print_mock.call_args_list)
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


class ServoGoalRangeTest(unittest.TestCase):
    """Every servo goal is saturated into the servo's raw range [-pi, pi - 2pi/4096]."""

    POSITIVE = "left_knee"  # MOTOR_SIGN +1
    NEGATIVE = "left_hip_yaw"  # MOTOR_SIGN -1

    def test_range_is_rustypots_raw_0_to_4095(self) -> None:
        self.assertEqual(SERVO_GOAL_MIN_RAD, raw_to_rad(0))
        self.assertEqual(SERVO_GOAL_MAX_RAD, raw_to_rad(4095))
        self.assertEqual(SERVO_GOAL_MIN_RAD, -math.pi)
        self.assertEqual(SERVO_GOAL_MAX_RAD, math.pi - 2.0 * math.pi / 4096.0)
        self.assertEqual(MOTOR_SIGN[self.POSITIVE], 1.0)
        self.assertEqual(MOTOR_SIGN[self.NEGATIVE], -1.0)

    def test_saturate_is_identity_inside_and_clamps_outside(self) -> None:
        for value in (
            -0.0, 0.0, 1.0e-300, -1.0, 2.5, SERVO_GOAL_MIN_RAD, SERVO_GOAL_MAX_RAD,
            np.nextafter(SERVO_GOAL_MIN_RAD, 0.0), np.nextafter(SERVO_GOAL_MAX_RAD, 0.0),
        ):
            with self.subTest(value=value):
                out = rc_module.saturate_servo_goal(float(value))
                self.assertEqual(out.hex(), float(value).hex())
        for value, edge in (
            (math.pi, SERVO_GOAL_MAX_RAD),
            (np.nextafter(SERVO_GOAL_MAX_RAD, 4.0), SERVO_GOAL_MAX_RAD),
            (100.0, SERVO_GOAL_MAX_RAD),
            (np.nextafter(SERVO_GOAL_MIN_RAD, -4.0), SERVO_GOAL_MIN_RAD),
            (-100.0, SERVO_GOAL_MIN_RAD),
        ):
            with self.subTest(value=value):
                self.assertEqual(rc_module.saturate_servo_goal(float(value)), edge)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rc_module.saturate_servo_goal(value)

    def _check_saturation(self, offsets_deg) -> None:
        offsets = offsets_deg or {}
        for path in ("neutral", "policy"):
            controller, bus = make_controller(offsets_deg)
            ids = list(MOTOR_TO_ID.values())
            if path == "policy":
                controller.sync_read_present_position(ids)
                bus.goal_writes.clear()
                write = controller.sync_write_goal_position
            else:
                write = controller.sync_write_neutral_goal_position
            for name in (self.POSITIVE, self.NEGATIVE, "left_ankle_pitch", "right_ankle_pitch"):
                motor_id = MOTOR_TO_ID[name]
                sign = MOTOR_SIGN[name]
                offset = math.radians(offsets.get(name, 0.0))
                for logical in (math.pi, -math.pi, 3.2, -3.2, 50.0, -50.0):
                    unbounded = (logical + offset) * sign
                    if SERVO_GOAL_MIN_RAD <= unbounded <= SERVO_GOAL_MAX_RAD:
                        continue
                    edge = SERVO_GOAL_MAX_RAD if unbounded > 0 else SERVO_GOAL_MIN_RAD
                    bus.goal_writes.clear()
                    write([motor_id], [logical])
                    with self.subTest(path=path, name=name, logical=logical):
                        self.assertEqual(bus.goals_for(motor_id), [edge])
                        # The cached goal is the logical pose actually commanded.
                        self.assertEqual(
                            controller.last_goal_targets[name],
                            edge * sign - offset,
                        )

    def test_out_of_range_goals_saturate_at_the_edges_without_offsets(self) -> None:
        self._check_saturation(None)

    def test_out_of_range_goals_saturate_at_the_edges_with_offsets(self) -> None:
        self._check_saturation(TEST_OFFSETS_DEG)

    def test_policy_targets_at_plus_minus_pi_map_to_both_edges(self) -> None:
        controller, bus = make_controller()
        ids = [MOTOR_TO_ID[self.POSITIVE], MOTOR_TO_ID[self.NEGATIVE]]
        controller.sync_write_neutral_goal_position(ids, [math.pi, math.pi])
        self.assertEqual(
            [value for _, value in bus.goal_writes], [SERVO_GOAL_MAX_RAD, SERVO_GOAL_MIN_RAD]
        )
        bus.goal_writes.clear()
        controller.sync_write_neutral_goal_position(ids, [-math.pi, -math.pi])
        # -pi is raw 0 (in range) for +1; +pi is past raw 4095 for -1.
        self.assertEqual(
            [value for _, value in bus.goal_writes], [-math.pi, SERVO_GOAL_MAX_RAD]
        )
        self.assertEqual(controller.last_goal_targets[self.POSITIVE], -math.pi)
        self.assertEqual(controller.last_goal_targets[self.NEGATIVE], -SERVO_GOAL_MAX_RAD)

    def test_in_range_edges_are_written_unchanged(self) -> None:
        controller, bus = make_controller()
        ids = [MOTOR_TO_ID[self.POSITIVE], MOTOR_TO_ID[self.NEGATIVE]]
        controller.sync_write_neutral_goal_position(
            ids, [SERVO_GOAL_MAX_RAD, -SERVO_GOAL_MAX_RAD]
        )
        controller.sync_write_neutral_goal_position(ids, [SERVO_GOAL_MIN_RAD, math.pi])
        self.assertEqual(
            [value.hex() for _, value in bus.goal_writes],
            [
                SERVO_GOAL_MAX_RAD.hex(), SERVO_GOAL_MAX_RAD.hex(),
                SERVO_GOAL_MIN_RAD.hex(), SERVO_GOAL_MIN_RAD.hex(),
            ],
        )
        self.assertEqual(controller.last_goal_targets[self.POSITIVE], SERVO_GOAL_MIN_RAD)
        self.assertEqual(controller.last_goal_targets[self.NEGATIVE], math.pi)

    def test_rejoin_slew_toward_a_saturated_target_finishes_at_the_exact_edge(self) -> None:
        for name, offsets_deg in (
            (self.POSITIVE, None),
            (self.NEGATIVE, None),
            ("right_ankle_pitch", TEST_OFFSETS_DEG),
            ("left_ankle_pitch", TEST_OFFSETS_DEG),
        ):
            with self.subTest(name=name, offsets=offsets_deg is not None):
                controller, bus = make_controller(offsets_deg)
                motor_id = MOTOR_TO_ID[name]
                controller.sync_read_present_position(list(MOTOR_TO_ID.values()))
                start = controller.last_goal_targets[name]
                target = 10.0 if MOTOR_SIGN[name] > 0 else -10.0  # past +pi servo
                controller._torque_rejoin_last_s[motor_id] = 0.0
                with mock.patch.object(rc_module.time, "monotonic", return_value=0.1):
                    controller.sync_write_goal_position([motor_id], [target])
                # One 0.1 s step of the 0.5 rad/s rejoin slew.
                self.assertAlmostEqual(
                    controller.last_goal_targets[name] - start,
                    math.copysign(0.05, target),
                    places=12,
                )
                for tick in range(2, 200):
                    if motor_id not in controller._torque_rejoin_last_s:
                        break
                    with mock.patch.object(
                        rc_module.time, "monotonic", return_value=0.1 * tick
                    ):
                        controller.sync_write_goal_position([motor_id], [target])
                self.assertNotIn(motor_id, controller._torque_rejoin_last_s)
                self.assertEqual(bus.goals_for(motor_id)[-1], SERVO_GOAL_MAX_RAD)
                self.assertTrue(
                    all(
                        SERVO_GOAL_MIN_RAD <= value <= SERVO_GOAL_MAX_RAD
                        for value in bus.goals_for(motor_id)
                    )
                )

    def test_non_finite_goal_holds_that_joints_last_goal(self) -> None:
        for path in ("neutral", "policy"):
            for bad in (math.nan, math.inf, -math.inf):
                with self.subTest(path=path, bad=bad):
                    controller, bus = make_controller(TEST_OFFSETS_DEG)
                    ids = [MOTOR_TO_ID[self.POSITIVE], MOTOR_TO_ID[self.NEGATIVE]]
                    if path == "policy":
                        controller.sync_read_present_position(list(MOTOR_TO_ID.values()))
                        write = controller.sync_write_goal_position
                    else:
                        write = controller.sync_write_neutral_goal_position
                    write(ids, [0.1, 0.2])
                    held = bus.goals_for(ids[1])[-1]
                    before = controller.last_goal_targets[self.NEGATIVE]
                    bus.goal_writes.clear()
                    with mock.patch("builtins.print") as printed:
                        write(ids, [0.3, bad])
                    # The finite joint moves; the bad one re-sends its last goal.
                    self.assertEqual(bus.goals_for(ids[0]), [controller._to_servo(ids[0], 0.3)])
                    self.assertEqual(bus.goals_for(ids[1]), [held])
                    self.assertEqual(controller.last_goal_targets[self.NEGATIVE], before)
                    self.assertTrue(all(math.isfinite(v) for _, v in bus.goal_writes))
                    self.assertIn("non-finite goal", printed.call_args.args[0])

    def test_rejoin_seed_at_the_raw_edges_writes_the_reading_back_exactly(self) -> None:
        for raw in (0, 4095):
            for name in ("right_hip_roll", "right_ankle_pitch", "left_hip_yaw"):
                controller, bus = make_controller(TEST_OFFSETS_DEG)
                ids = list(MOTOR_TO_ID.values())
                controller.sync_read_present_position(ids)
                motor_id = MOTOR_TO_ID[name]
                controller.sync_write_torque_enable([motor_id], [True])
                bus.torque[motor_id] = 0
                bus.raw_position[motor_id] = raw
                bus.goal_writes.clear()
                controller._check_torque_state(motor_id)
                with self.subTest(raw=raw, name=name):
                    self.assertEqual(bus.torque[motor_id], 1)
                    self.assertEqual(
                        [value.hex() for value in bus.goals_for(motor_id)],
                        [raw_to_rad(raw).hex()],
                    )
                    self.assertEqual(
                        controller.last_goal_targets[name],
                        controller._from_servo(motor_id, raw_to_rad(raw)),
                    )


class ServoLimitsTest(unittest.TestCase):
    """Operating Mode and Min/Max Position Limit are read at startup."""

    # Sign -1 with offset -3 deg, sign +1 with offset -1 deg, sign -1 with +1.5 deg.
    NARROW = {
        "left_hip_yaw": (1024, 3072),
        "left_ankle_pitch": (1500, 2600),
        "right_ankle_pitch": (1800, 4095),
    }

    def _narrow(self, bus) -> None:
        for name, (low, high) in self.NARROW.items():
            bus.raw_min_limit[MOTOR_TO_ID[name]] = low
            bus.raw_max_limit[MOTOR_TO_ID[name]] = high

    def test_default_limits_keep_the_full_range_exactly(self) -> None:
        lines: list[str] = []
        controller, _bus = make_controller(printed=lines)
        self.assertEqual(lines, [])
        for motor_id in MOTOR_TO_ID.values():
            lower, upper = controller._id_to_goal_bounds[motor_id]
            self.assertEqual(lower.hex(), SERVO_GOAL_MIN_RAD.hex())
            self.assertEqual(upper.hex(), SERVO_GOAL_MAX_RAD.hex())
        # Limits wider than the raw goal range are intersected with it.
        self.assertEqual(
            rc_module.servo_goal_bounds(-100, 5000), (SERVO_GOAL_MIN_RAD, SERVO_GOAL_MAX_RAD)
        )

    def test_non_position_mode_refuses_to_start(self) -> None:
        def configure(bus) -> None:
            bus.operating_mode[MOTOR_TO_ID["left_knee"]] = 4  # extended position
            bus.operating_mode[MOTOR_TO_ID["neck_roll"]] = 1  # velocity

        with self.assertRaises(RuntimeError) as raised:
            make_controller(configure=configure)
        message = str(raised.exception)
        self.assertIn("left_knee", message)
        self.assertIn("operating mode 4", message)
        self.assertIn("neck_roll", message)
        self.assertNotIn("right_knee", message)

    def test_empty_limit_range_refuses_to_start(self) -> None:
        def configure(bus) -> None:
            bus.raw_min_limit[MOTOR_TO_ID["right_elbow"]] = 3000
            bus.raw_max_limit[MOTOR_TO_ID["right_elbow"]] = 1000

        with self.assertRaisesRegex(RuntimeError, "right_elbow"):
            make_controller(configure=configure)

    def test_unreadable_servo_keeps_full_range_with_a_warning(self) -> None:
        silent = MOTOR_TO_ID["left_hip_yaw"]
        flaky = MOTOR_TO_ID["left_ankle_pitch"]

        def configure(bus) -> None:
            self._narrow(bus)
            bus.config_failures[silent] = 1000
            # Fewer failures than attempts per register: the retry reads it.
            bus.config_failures[flaky] = rc_module.SERVO_CONFIG_READ_ATTEMPTS - 1

        lines: list[str] = []
        controller, bus = make_controller(configure=configure, printed=lines)
        self.assertEqual(
            controller._id_to_goal_bounds[silent], (SERVO_GOAL_MIN_RAD, SERVO_GOAL_MAX_RAD)
        )
        self.assertEqual(
            controller._id_to_goal_bounds[flaky], rc_module.servo_goal_bounds(1500, 2600)
        )
        warnings = [line for line in lines if "unreadable" in line]
        self.assertTrue(warnings)
        self.assertTrue(all("left_hip_yaw" in line for line in warnings))
        self.assertTrue(any("full goal range" in line for line in warnings))
        # The silent servo still takes full-range goals.
        controller.sync_write_neutral_goal_position([silent], [-3.0])
        self.assertEqual(bus.goals_for(silent), [3.0])

    def test_bounds_map_back_inside_the_raw_limits(self) -> None:
        for low, high in ((1, 4094), (1024, 3072), (1500, 2600), (7, 7), (0, 2048), (100, 4095)):
            lower, upper = rc_module.servo_goal_bounds(low, high)
            with self.subTest(low=low, high=high):
                self.assertLessEqual(lower, upper)
                for bound in (lower, upper):
                    exact = (bound + math.pi) * 4096.0 / (2.0 * math.pi)
                    for raw in (int(exact), round(exact)):
                        self.assertGreaterEqual(raw, low)
                        self.assertLessEqual(raw, high)
                self.assertAlmostEqual(lower, raw_to_rad(low), places=12)
                self.assertAlmostEqual(upper, raw_to_rad(high), places=12)

    def test_narrow_limits_saturate_per_joint_with_sign_and_offset(self) -> None:
        lines: list[str] = []
        for path in ("neutral", "policy"):
            lines.clear()
            controller, bus = make_controller(
                TEST_OFFSETS_DEG, configure=self._narrow, printed=lines
            )
            limit_lines = [line for line in lines if line.startswith("Servo position limits")]
            self.assertEqual(len(limit_lines), len(self.NARROW))
            ids = list(MOTOR_TO_ID.values())
            if path == "policy":
                controller.sync_read_present_position(ids)
                write = controller.sync_write_goal_position
            else:
                write = controller.sync_write_neutral_goal_position
            for name, (low, high) in self.NARROW.items():
                motor_id = MOTOR_TO_ID[name]
                lower, upper = rc_module.servo_goal_bounds(low, high)
                sign = MOTOR_SIGN[name]
                offset = math.radians(TEST_OFFSETS_DEG[name])
                line = next(line for line in limit_lines if f"name={name} " in line)
                self.assertIn(f"raw [{low}, {high}]", line)
                ends = sorted((lower * sign - offset, upper * sign - offset))
                self.assertIn(f"[{ends[0]:+.4f}, {ends[1]:+.4f}]", line)
                for logical in (math.pi, -math.pi, 2.0, -2.0, 0.3, -0.3):
                    unbounded = (logical + offset) * sign
                    bus.goal_writes.clear()
                    write([motor_id], [logical])
                    with self.subTest(path=path, name=name, logical=logical):
                        if lower <= unbounded <= upper:
                            self.assertEqual(bus.goals_for(motor_id), [unbounded])
                            self.assertEqual(controller.last_goal_targets[name], logical)
                        else:
                            edge = upper if unbounded > upper else lower
                            self.assertEqual(bus.goals_for(motor_id), [edge])
                            self.assertEqual(
                                controller.last_goal_targets[name], edge * sign - offset
                            )
            # Joints with factory limits keep the full range.
            knee = MOTOR_TO_ID["left_knee"]
            bus.goal_writes.clear()
            write([knee], [math.pi])
            self.assertEqual(bus.goals_for(knee), [SERVO_GOAL_MAX_RAD])

    def test_rejoin_seed_saturates_a_reading_outside_narrow_limits(self) -> None:
        name = "left_hip_yaw"
        motor_id = MOTOR_TO_ID[name]
        low, high = self.NARROW[name]
        lower, upper = rc_module.servo_goal_bounds(low, high)
        for raw, expected in ((2000, raw_to_rad(2000)), (high, upper), (high + 50, upper),
                              (low - 50, lower)):
            controller, bus = make_controller(TEST_OFFSETS_DEG, configure=self._narrow)
            controller.sync_read_present_position(list(MOTOR_TO_ID.values()))
            controller.sync_write_torque_enable([motor_id], [True])
            bus.torque[motor_id] = 0
            bus.raw_position[motor_id] = raw
            bus.goal_writes.clear()
            controller._check_torque_state(motor_id)
            with self.subTest(raw=raw):
                self.assertEqual(bus.torque[motor_id], 1)
                self.assertEqual(
                    [value.hex() for value in bus.goals_for(motor_id)], [expected.hex()]
                )
                self.assertEqual(
                    controller.last_goal_targets[name],
                    controller._from_servo(motor_id, expected),
                )


if __name__ == "__main__":
    unittest.main()
