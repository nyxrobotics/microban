# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from rustypot import Xl330PyController
import numpy as np
import math
import time

from constants import MOTOR_TO_ID, MOTOR_SIGN, NEUTRAL_POSE, IMU_I2C_BUS, PRESENT_CURRENT_UNIT_A, KP_DEFAULT
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
        self._requested_torque_ids: set[int] = set()
        self._torque_unconfirmed_ids: set[int] = set()
        self._torque_check_ids = tuple(MOTOR_TO_ID.values())
        self._torque_check_cursor = 0
        self._torque_check_next_s = time.monotonic() + 0.1
        self._torque_warn_after_s: dict[int, float] = {}
        # A servo that reappears after losing torque must join a moving target
        # gradually, even if the scheduler has already finished its A slew.
        self._torque_rejoin_last_s: dict[int, float] = {}
        self._last_positions = {MOTOR_TO_ID[name]: float(NEUTRAL_POSE[name]) for name in MOTOR_TO_ID}
        self._last_velocities = {motor_id: 0.0 for motor_id in MOTOR_TO_ID.values()}
        self._last_currents = {motor_id: 0.0 for motor_id in MOTOR_TO_ID.values()}
        self._last_voltages = {motor_id: 0.0 for motor_id in MOTOR_TO_ID.values()}
        self._last_goals = dict(self._last_positions)
        self._last_kp = {motor_id: KP_DEFAULT for motor_id in MOTOR_TO_ID.values()}
        self._proxy_ignore_until: dict[int, float] = {}
        self._head_last_read_s: dict[int, float] = {}
        self._imu_reader = ThreadedIMUReader(i2c_bus=IMU_I2C_BUS, frequency_hz=100.0)
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
            if motor_id in self._stale_ids
            or motor_id in self._torque_unconfirmed_ids
            or now < self._proxy_ignore_until.get(motor_id, 0.0)
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
        if motor_id in self._pending_enable and motor_id not in self._requested_torque_ids:
            self._pending_enable.discard(motor_id)
        if motor_id in self._pending_enable and motor_id not in self._torque_unconfirmed_ids:
            # A previously silent joint must receive its measured pose before
            # torque is enabled. A failed write leaves it pending for retry.
            try:
                self._controller.sync_write_goal_position(
                    [motor_id], [value * self._id_to_sign[motor_id]]
                )
                self._controller.sync_write_position_p_gain(
                    [motor_id], [self._last_kp[motor_id]]
                )
                self._controller.sync_write_torque_enable([motor_id], [True])
            except (RuntimeError, OSError):
                self._stale_ids.add(motor_id)
                return
            self._last_goals[motor_id] = value
            self._pending_enable.discard(motor_id)
            self._torque_rejoin_last_s[motor_id] = time.monotonic()
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
            if not enabled:
                # Clear the desired state before touching the bus so a failed
                # B write can never cause the feedback loop to re-enable it.
                self._requested_torque_ids.discard(motor_id)
                self._torque_unconfirmed_ids.discard(motor_id)
                self._torque_rejoin_last_s.pop(motor_id, None)
            if enabled and motor_id in self._stale_ids:
                self._pending_enable.add(motor_id)
            else:
                write_ids.append(motor_id)
                write_values.append(enabled)
                if not enabled:
                    self._pending_enable.discard(motor_id)
        if write_ids:
            self._controller.sync_write_torque_enable(write_ids, write_values)
        for motor_id, enabled in zip(ids, values):
            if enabled:
                self._requested_torque_ids.add(motor_id)

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        self._controller.sync_write_status_return_level(ids, levels)

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        if len(ids) != len(positions):
            raise ValueError("motor ID and goal counts differ")
        now = time.monotonic()
        live = []
        rejoining = []
        for motor_id, raw_pos in zip(ids, positions):
            if motor_id in self._stale_ids or motor_id in self._torque_unconfirmed_ids:
                continue
            pos = float(raw_pos)
            last_rejoin_s = self._torque_rejoin_last_s.get(motor_id)
            if last_rejoin_s is not None:
                dt = max(0.001, min(0.1, now - last_rejoin_s))
                max_step = 0.5 * dt
                previous = self._last_goals[motor_id]
                delta = max(-max_step, min(max_step, pos - previous))
                sent = previous + delta
                rejoining.append((motor_id, abs(sent - pos) <= 1e-9))
                pos = sent
            live.append((motor_id, pos))
        if live:
            self._controller.sync_write_goal_position(
                [motor_id for motor_id, _ in live],
                [pos * self._id_to_sign[motor_id] for motor_id, pos in live],
            )
            self._last_goals.update(live)
            for motor_id, finished in rejoining:
                if finished:
                    self._torque_rejoin_last_s.pop(motor_id, None)
                else:
                    self._torque_rejoin_last_s[motor_id] = now

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        self._read_core_state()
        if self._head_ids:
            self._poll_one_head_position()
        self._poll_torque_state()
        return [self._last_positions[motor_id] for motor_id in ids]

    def _torque_warn(self, motor_id: int, detail: str) -> None:
        now = time.monotonic()
        if now < self._torque_warn_after_s.get(motor_id, 0.0):
            return
        self._torque_warn_after_s[motor_id] = now + 10.0
        print(
            f"Servo torque feedback: id={motor_id} "
            f"name={self._id_to_name[motor_id]} {detail}",
            end="\r\n", flush=True,
        )

    def _seed_rejoin_goal(self, motor_id: int) -> float:
        """Hold a freshly measured pose before restoring this servo's drive."""
        motor_position = self._scalar(
            self._controller.read_present_position(motor_id)
        )
        measured = motor_position * self._id_to_sign[motor_id]
        if not math.isfinite(measured):
            raise ValueError("non-finite measured position")
        self._controller.sync_write_goal_position([motor_id], [motor_position])
        self._last_goals[motor_id] = measured
        # RAM gains may have reset if the servo rebooted.  Preserve the A or
        # policy gain that the scheduler most recently requested for this ID.
        self._controller.sync_write_position_p_gain(
            [motor_id], [self._last_kp[motor_id]]
        )
        return measured

    def _poll_torque_state(self) -> None:
        if not self._requested_torque_ids:
            return
        now = time.monotonic()
        if now < self._torque_check_next_s:
            return
        self._torque_check_next_s = now + 0.1
        motor_id = self._torque_check_ids[self._torque_check_cursor]
        self._torque_check_cursor = (self._torque_check_cursor + 1) % len(self._torque_check_ids)
        if motor_id not in self._requested_torque_ids:
            return

        try:
            enabled = self._scalar(self._controller.read_torque_enable(motor_id))
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            self._torque_warn(motor_id, f"read failed: {exc}")
            return
        if enabled == 1.0:
            if motor_id in self._torque_unconfirmed_ids:
                try:
                    measured = self._seed_rejoin_goal(motor_id)
                except (RuntimeError, OSError, TypeError, ValueError) as exc:
                    self._torque_warn(motor_id, f"ON, rejoin deferred: {exc}")
                    return
                self._finish_torque_rejoin(motor_id, measured, "ON again")
            return
        if enabled != 0.0:
            self._torque_warn(motor_id, f"invalid register value: {enabled!r}")
            return

        self._torque_unconfirmed_ids.add(motor_id)
        self._torque_rejoin_last_s.pop(motor_id, None)
        try:
            hardware_error = self._scalar(
                self._controller.read_hardware_error_status(motor_id)
            )
            if hardware_error != 0.0:
                self._torque_warn(
                    motor_id,
                    f"OFF, hardware error={hardware_error}; enable deferred",
                )
                return
            # Never revive a servo against an old goal register.  The next
            # scheduler command is slew-limited from this fresh pose.
            measured = self._seed_rejoin_goal(motor_id)
            if motor_id not in self._requested_torque_ids:
                return
            self._controller.sync_write_torque_enable([motor_id], [True])
            verified = self._scalar(
                self._controller.read_torque_enable(motor_id)
            )
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            self._torque_warn(motor_id, f"OFF, recovery deferred: {exc}")
            return
        if verified != 1.0:
            self._torque_warn(
                motor_id,
                f"OFF, retry did not take effect (register={verified!r})",
            )
            return

        self._finish_torque_rejoin(motor_id, measured, "was OFF")

    def _finish_torque_rejoin(
        self, motor_id: int, measured: float, previous_status: str
    ) -> None:
        was_stale = motor_id in self._stale_ids
        self._last_positions[motor_id] = measured
        self._pending_enable.discard(motor_id)
        self._stale_ids.discard(motor_id)
        self._torque_unconfirmed_ids.discard(motor_id)
        self._torque_rejoin_last_s[motor_id] = time.monotonic()
        if was_stale:
            self._proxy_ignore_until[motor_id] = time.monotonic() + 0.12
        self._torque_warn_after_s.pop(motor_id, None)
        print(
            f"Servo torque feedback: id={motor_id} "
            f"name={self._id_to_name[motor_id]} {previous_status}; "
            "seeded measured goal and restored torque with 0.5 rad/s rejoin",
            end="\r\n", flush=True,
        )

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
        self._last_kp.update(zip(ids, gains))

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
