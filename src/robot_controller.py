# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from rustypot import Xl330PyController
import numpy as np
import math
import time

from constants import MOTOR_TO_ID, MOTOR_SIGN, NEUTRAL_POSE, IMU_I2C_BUS, PRESENT_CURRENT_UNIT_A
from imu_reader import ThreadedIMUReader


class RobotController:
    """Wraps Xl330PyController."""

    def __init__(self, serial_port: str = "/dev/ttyAMA0", baudrate: int = 1_000_000, timeout: float = 0.001) -> None:
        self._controller = Xl330PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        self._id_to_sign: dict[int, float] = {MOTOR_TO_ID[name]: MOTOR_SIGN[name] for name in MOTOR_TO_ID}
        self._id_to_name = {motor_id: name for name, motor_id in MOTOR_TO_ID.items()}
        self._core_groups = [
            [motor_id for motor_id in MOTOR_TO_ID.values() if motor_id // 10 == decade]
            for decade in (1, 2, 3, 4)
        ]
        # Split a group after a failed response so one silent servo cannot
        # block feedback and goal writes for its responsive neighbours.
        self._state_groups = [group[:] for group in self._core_groups if group]
        self._state_retry_after: dict[tuple[int, ...], float] = {}
        self._state_failures: dict[tuple[int, ...], int] = {}
        self._state_last_failure: dict[int, float] = {}
        self._head_ids = [motor_id for motor_id in MOTOR_TO_ID.values() if motor_id // 10 == 5]
        self._head_cursor = 0
        self._retry_after: dict[tuple[str, tuple[int, ...]], float] = {}
        self._head_retry_after: dict[int, float] = {}
        self._head_failures: dict[int, int] = {}
        self._stale_ids = set(MOTOR_TO_ID.values())
        self._pending_enable: set[int] = set()
        self._last_positions = {MOTOR_TO_ID[name]: float(NEUTRAL_POSE[name]) for name in MOTOR_TO_ID}
        self._last_velocities = {motor_id: 0.0 for motor_id in MOTOR_TO_ID.values()}
        self._last_currents = {motor_id: 0.0 for motor_id in MOTOR_TO_ID.values()}
        self._last_voltages = {motor_id: 0.0 for motor_id in MOTOR_TO_ID.values()}
        self._last_goals = dict(self._last_positions)
        self._proxy_ignore_until: dict[int, float] = {}
        self._head_last_read_s: dict[int, float] = {}
        self._imu_reader = ThreadedIMUReader(i2c_bus=IMU_I2C_BUS, frequency_hz=200.0)
        self._imu_reader.start()

    @property
    def stale_motor_names(self) -> frozenset[str]:
        return frozenset(self._id_to_name[motor_id] for motor_id in self._stale_ids)

    @property
    def state_group_count(self) -> int:
        return len(self._state_groups)

    @property
    def proxy_ignored_motor_names(self) -> frozenset[str]:
        now = time.monotonic()
        return frozenset(
            self._id_to_name[motor_id]
            for motor_id in MOTOR_TO_ID.values()
            if motor_id in self._stale_ids or now < self._proxy_ignore_until.get(motor_id, 0.0)
        )

    @property
    def last_goal_targets(self) -> dict[str, float]:
        """Goal values actually written, including joints held through read faults."""
        return {self._id_to_name[motor_id]: value for motor_id, value in self._last_goals.items()}

    @staticmethod
    def _scalar(raw) -> float:
        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) != 1:
                raise ValueError("servo response must contain one value")
            raw = raw[0]
        return float(raw)

    def _position_recovered(self, motor_id: int, value: float) -> None:
        was_stale = motor_id in self._stale_ids
        self._last_positions[motor_id] = value
        if motor_id in self._pending_enable:
            # A previously silent joint must receive its measured pose before
            # torque is enabled. A failed write leaves it pending for retry.
            try:
                self._controller.sync_write_goal_position(
                    [motor_id], [value * self._id_to_sign[motor_id]]
                )
                self._controller.sync_write_torque_enable([motor_id], [True])
            except (RuntimeError, OSError):
                self._stale_ids.add(motor_id)
                return
            self._last_goals[motor_id] = value
            self._pending_enable.discard(motor_id)
        self._stale_ids.discard(motor_id)
        if was_stale:
            # Current estimation aligns feedback with delayed goal history.
            self._proxy_ignore_until[motor_id] = time.monotonic() + 0.12

    def _read_core(self, kind: str, method: str, cache: dict[int, float], convert) -> None:
        now = time.monotonic()
        for group in self._core_groups:
            if not group:
                continue
            key = (kind, tuple(group))
            if now < self._retry_after.get(key, 0.0):
                self._stale_ids.update(group)
                continue
            if kind != "position" and any(motor_id in self._stale_ids for motor_id in group):
                continue
            try:
                raw = getattr(self._controller, method)(group)
                if len(raw) != len(group):
                    raise RuntimeError(f"{method} returned incomplete feedback")
                values = [convert(motor_id, item) for motor_id, item in zip(group, raw)]
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError(f"{method} returned non-finite feedback")
            except (RuntimeError, OSError, TypeError, ValueError):
                self._stale_ids.update(group)
                self._retry_after[key] = now + 0.08
                continue
            for motor_id, value in zip(group, values):
                if kind == "position":
                    self._position_recovered(motor_id, value)
                else:
                    cache[motor_id] = value

    def _read_core_state(self) -> None:
        """Read contiguous velocity and position registers in one bus request."""
        now = time.monotonic()
        next_groups: list[list[int]] = []
        for group in self._state_groups:
            key = tuple(group)
            if now < self._state_retry_after.get(key, 0.0):
                next_groups.append(group)
                continue
            try:
                rows = self._controller.sync_read_raw_data(group, 128, 8)
                if len(rows) != len(group) or any(len(row) != 8 for row in rows):
                    raise RuntimeError("incomplete position/velocity feedback")
                values = []
                for motor_id, row in zip(group, rows):
                    raw_velocity = int.from_bytes(row[:4], "little", signed=True)
                    raw_position = int.from_bytes(row[4:8], "little", signed=True)
                    sign = self._id_to_sign[motor_id]
                    position = (2.0 * math.pi * raw_position / 4096.0 - math.pi) * sign
                    velocity = raw_velocity * 0.229 * math.pi / 30.0 * sign
                    values.append((position, velocity))
            except (RuntimeError, OSError, TypeError, ValueError):
                self._stale_ids.update(group)
                for motor_id in group:
                    self._state_last_failure[motor_id] = now
                failures = self._state_failures.get(key, 0) + 1
                self._state_failures[key] = failures
                if len(group) > 1 and failures >= 3:
                    middle = len(group) // 2
                    next_groups.extend((group[:middle], group[middle:]))
                    self._state_failures.pop(key, None)
                else:
                    if len(group) == 1:
                        motor_id = group[0]
                        max_delay = 0.2 if motor_id // 10 in (1, 2) else 1.0
                        self._state_retry_after[key] = now + min(
                            max_delay, 0.2 * 2 ** min(failures - 1, 3)
                        )
                    next_groups.append(group)
                continue
            self._state_failures.pop(key, None)
            for motor_id, (position, velocity) in zip(group, values):
                self._last_velocities[motor_id] = velocity
                self._position_recovered(motor_id, position)
            next_groups.append(group)
        # Restore the cheaper original grouping after every member has been
        # answering steadily. A newly silent member will be split again.
        for original in self._core_groups:
            if not original or any(motor_id in self._stale_ids for motor_id in original):
                continue
            if any(now - self._state_last_failure.get(motor_id, 0.0) < 5.0 for motor_id in original):
                continue
            members = set(original)
            parts = [group for group in next_groups if set(group).issubset(members)]
            if len(parts) > 1:
                first = next_groups.index(parts[0])
                next_groups = [group for group in next_groups if group not in parts]
                next_groups.insert(first, original[:])
        self._state_groups = next_groups

    def _poll_one_head_position(self) -> None:
        now = time.monotonic()
        for _ in self._head_ids:
            motor_id = self._head_ids[self._head_cursor]
            self._head_cursor = (self._head_cursor + 1) % len(self._head_ids)
            if now < self._head_retry_after.get(motor_id, 0.0):
                continue
            try:
                raw = self._controller.read_present_position(motor_id)
                value = self._scalar(raw) * self._id_to_sign[motor_id]
                if not math.isfinite(value):
                    raise RuntimeError("non-finite head position")
            except (RuntimeError, OSError, TypeError, ValueError):
                self._stale_ids.add(motor_id)
                failures = self._head_failures.get(motor_id, 0) + 1
                self._head_failures[motor_id] = failures
                self._head_retry_after[motor_id] = now + min(
                    1.0, 0.2 * 2 ** min(failures - 1, 3)
                )
                return
            self._head_failures.pop(motor_id, None)
            previous_s = self._head_last_read_s.get(motor_id)
            if previous_s is not None and now > previous_s:
                self._last_velocities[motor_id] = (value - self._last_positions[motor_id]) / (now - previous_s)
            self._head_last_read_s[motor_id] = now
            self._position_recovered(motor_id, value)
            return

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        if len(ids) != len(values):
            raise ValueError("motor ID and torque value counts differ")
        write_ids, write_values = [], []
        for motor_id, enabled in zip(ids, values):
            if enabled and motor_id in self._stale_ids:
                self._pending_enable.add(motor_id)
            else:
                write_ids.append(motor_id)
                write_values.append(enabled)
                if not enabled:
                    self._pending_enable.discard(motor_id)
        if write_ids:
            self._controller.sync_write_torque_enable(write_ids, write_values)

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        self._controller.sync_write_status_return_level(ids, levels)

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        if len(ids) != len(positions):
            raise ValueError("motor ID and goal counts differ")
        live = [(motor_id, float(pos)) for motor_id, pos in zip(ids, positions) if motor_id not in self._stale_ids]
        if live:
            self._controller.sync_write_goal_position(
                [motor_id for motor_id, _ in live],
                [pos * self._id_to_sign[motor_id] for motor_id, pos in live],
            )
            self._last_goals.update(live)

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        self._read_core_state()
        if self._head_ids:
            self._poll_one_head_position()
        return [self._last_positions[motor_id] for motor_id in ids]

    def read_present_position(self, motor_id: int) -> float:
        try:
            value = self._scalar(self._controller.read_present_position(motor_id)) * self._id_to_sign[motor_id]
            if not math.isfinite(value):
                raise RuntimeError("non-finite position")
        except (RuntimeError, OSError, TypeError, ValueError):
            self._stale_ids.add(motor_id)
            return self._last_positions[motor_id]
        self._position_recovered(motor_id, value)
        return value

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        return [self._last_velocities[motor_id] for motor_id in ids]
    
    def read_present_velocity(self, motor_id: int) -> float:
        try:
            raw = self._scalar(self._controller.read_present_velocity(motor_id))
            value = raw * 0.229 * np.pi / 30 * self._id_to_sign[motor_id]
            if not math.isfinite(value):
                raise RuntimeError("non-finite velocity")
        except (RuntimeError, OSError, TypeError, ValueError):
            self._stale_ids.add(motor_id)
            return self._last_velocities[motor_id]
        self._last_velocities[motor_id] = value
        return value

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        """Present current per motor, in Amps (signed). Magnitude is what matters for the BMS budget."""
        self._read_core(
            "current", "sync_read_present_current", self._last_currents,
            lambda _motor_id, value: self._scalar(value) * PRESENT_CURRENT_UNIT_A,
        )
        return [self._last_currents[motor_id] for motor_id in ids]

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        self._read_core(
            "voltage", "sync_read_present_input_voltage", self._last_voltages,
            lambda _motor_id, value: self._scalar(value),
        )
        return [self._last_voltages[motor_id] for motor_id in ids]

    def read_present_input_voltage(self, motor_id: int) -> float:
        try:
            value = self._scalar(self._controller.read_present_input_voltage(motor_id))
            if not math.isfinite(value):
                raise RuntimeError("non-finite voltage")
        except (RuntimeError, OSError, TypeError, ValueError):
            self._stale_ids.add(motor_id)
            return self._last_voltages[motor_id]
        self._last_voltages[motor_id] = value
        return value

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        return [int(self._scalar(v)) for v in self._controller.sync_read_position_p_gain(ids)]

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._controller.sync_write_position_p_gain(ids, gains)

    def read_acc(self) -> tuple[float, float, float]:
        """Return raw accelerometer (ax, ay, az) in g."""
        return self._imu_reader.get_latest().acc

    def read_gyro(self) -> tuple[float, float, float]:
        """Return (gx, gy, gz) in rad/s."""
        return self._imu_reader.get_latest().gyro

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        """Return orientation quaternion (w, x, y, z)."""
        _ = dt
        return self._imu_reader.get_latest().quat

    def get_imu_status(self) -> dict[str, float | int | bool]:
        return self._imu_reader.get_status()

    def shutdown(self) -> None:
        self._imu_reader.stop()

    def close(self) -> None:
        self.shutdown()
