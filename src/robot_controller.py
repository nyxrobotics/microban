# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from rustypot import Xl330PyController
import numpy as np
import math
import time

from constants import (
    HARDWARE_JOINT_OFFSET_MAX_RAD,
    HARDWARE_JOINT_OFFSET_RAD,
    IMU_I2C_BUS,
    KP_DEFAULT,
    MOTOR_SIGN,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    PRESENT_CURRENT_UNIT_A,
)
from imu_reader import ThreadedIMUReader


TORQUE_VERIFY_TIMEOUT_S = 1.0


def validated_hardware_offsets(offsets: dict[str, float]) -> dict[str, float]:
    """Check a per-joint real-robot offset table (logical radians)."""
    missing = sorted(set(MOTOR_TO_ID) - set(offsets))
    unknown = sorted(set(offsets) - set(MOTOR_TO_ID))
    if missing or unknown:
        raise ValueError(
            f"hardware joint offsets must list every joint: missing={missing}, unknown={unknown}"
        )
    checked = {}
    for name in MOTOR_TO_ID:
        value = float(offsets[name])
        if not math.isfinite(value) or abs(value) > HARDWARE_JOINT_OFFSET_MAX_RAD:
            raise ValueError(
                f"hardware joint offset {name}={offsets[name]!r} rad must be finite "
                f"and within +-{HARDWARE_JOINT_OFFSET_MAX_RAD} rad"
            )
        checked[name] = value
    return checked


class RobotController:
    """Wraps Xl330PyController.

    Every value above this class (goals, measurements, caches, last goal
    targets) is in the logical joint coordinate used by the policies and
    training. ``_to_servo``/``_from_servo`` are the only conversions to and from
    the servo coordinate: MOTOR_SIGN plus the real-robot calibration offset
    (constants.HARDWARE_JOINT_OFFSET_RAD).
    """

    def __init__(
        self,
        serial_port: str = "/dev/ttyAMA0",
        baudrate: int = 1_000_000,
        timeout: float = 0.001,
        joint_offsets_rad: dict[str, float] | None = None,
    ) -> None:
        self._controller = Xl330PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        self._id_to_sign: dict[int, float] = {MOTOR_TO_ID[name]: MOTOR_SIGN[name] for name in MOTOR_TO_ID}
        offsets = validated_hardware_offsets(
            HARDWARE_JOINT_OFFSET_RAD if joint_offsets_rad is None else joint_offsets_rad
        )
        self._id_to_offset: dict[int, float] = {
            MOTOR_TO_ID[name]: offset for name, offset in offsets.items()
        }
        nonzero = [
            f"{name}={math.degrees(offset):+.3f}deg"
            for name, offset in offsets.items() if offset != 0.0
        ]
        if nonzero:
            print(
                "Hardware joint offsets (servo = logical + offset): " + ", ".join(nonzero),
                end="\r\n", flush=True,
            )
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
        # A neutral-return goal was written before ON. For an unreadable joint
        # that goal is neutral; for a readable joint it is the measured pose.
        # Either can be enabled without waiting for another position reply.
        self._neutral_fallback_enable_ids: set[int] = set()
        # OFF remains eligible for background retry if its broadcast write
        # could not be confirmed within the one-second A/B transaction.
        self._torque_off_unconfirmed_ids: set[int] = set()
        self._torque_check_ids = tuple(MOTOR_TO_ID.values())
        self._torque_check_cursor = 0
        self._torque_recovery_cursor = 0
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

    def _to_servo(self, motor_id: int, logical: float) -> float:
        """Logical joint radians -> servo goal radians (sign and calibration offset)."""
        offset = self._id_to_offset[motor_id]
        if offset:
            logical = logical + offset
        return logical * self._id_to_sign[motor_id]

    def _from_servo(self, motor_id: int, servo: float) -> float:
        """Servo position radians -> logical joint radians (inverse of _to_servo)."""
        return servo * self._id_to_sign[motor_id] - self._id_to_offset[motor_id]

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
                    [motor_id], [self._to_servo(motor_id, value)]
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
                    # Same raw->rad mapping as rustypot: raw 0..4095 is [-pi, pi).
                    position = self._from_servo(
                        motor_id, 2.0 * math.pi * raw_position / 4096.0 - math.pi
                    )
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
                value = self._from_servo(motor_id, self._scalar(raw))
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
        if not ids:
            return
        all_joint_transition = (
            len(ids) == len(self._torque_check_ids)
            and set(ids) == set(self._torque_check_ids)
            and all(value == values[0] for value in values)
        )
        write_ids, write_values = [], []
        for motor_id, enabled in zip(ids, values):
            if not enabled:
                # Clear the desired state before touching the bus so a failed
                # B write can never cause the feedback loop to re-enable it.
                self._requested_torque_ids.discard(motor_id)
                self._torque_unconfirmed_ids.discard(motor_id)
                self._torque_off_unconfirmed_ids.add(motor_id)
                self._torque_rejoin_last_s.pop(motor_id, None)
                self._pending_enable.discard(motor_id)
                self._neutral_fallback_enable_ids.discard(motor_id)
            else:
                # Cancel previous OFF retries immediately. A failed ON write
                # is retried by the scheduler with a fresh measured goal.
                self._torque_off_unconfirmed_ids.discard(motor_id)
            if (
                enabled
                and motor_id in self._stale_ids
                and motor_id not in self._neutral_fallback_enable_ids
            ):
                self._pending_enable.add(motor_id)
            else:
                write_ids.append(motor_id)
                write_values.append(enabled)
        write_error = None
        transition_started_s = time.monotonic()
        if write_ids:
            try:
                # Sync-write uses Dynamixel's broadcast packet ID. No other
                # scheduler motor command runs during this transaction.
                self._controller.sync_write_torque_enable(write_ids, write_values)
            except (RuntimeError, OSError, ValueError) as exc:
                if not all_joint_transition:
                    raise
                write_error = exc
        if all_joint_transition:
            self._verify_torque_transition(
                ids, values[0], write_error, transition_started_s
            )
        for motor_id, enabled in zip(ids, values):
            if enabled:
                self._requested_torque_ids.add(motor_id)

    def _verify_torque_transition(
        self,
        ids: list[int],
        enabled: bool,
        initial_write_error: Exception | None,
        started_s: float,
    ) -> None:
        """Confirm a broadcast A/B transition before normal goals resume."""

        deadline = started_s + TORQUE_VERIFY_TIMEOUT_S
        pending = set(ids)
        last_status: dict[int, float] = {}
        seeded_goals: dict[int, float] = {}
        last_on_retry_s: dict[int, float] = {}
        group_read_available = True
        last_write_error = initial_write_error
        expected = 1.0 if enabled else 0.0
        while pending and time.monotonic() < deadline:
            ordered = [motor_id for motor_id in ids if motor_id in pending]
            status: dict[int, float] = {}
            if group_read_available and len(ordered) > 1:
                try:
                    values = self._controller.sync_read_torque_enable(ordered)
                    if len(values) != len(ordered):
                        raise RuntimeError("incomplete torque state read")
                    status = {
                        motor_id: self._scalar(value)
                        for motor_id, value in zip(ordered, values)
                    }
                except (RuntimeError, OSError, TypeError, ValueError):
                    # One silent servo can invalidate a group response. Check
                    # IDs individually for the rest of this transition.
                    group_read_available = False
            if not status:
                for motor_id in ordered:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        status[motor_id] = self._scalar(
                            self._controller.read_torque_enable(motor_id)
                        )
                    except (RuntimeError, OSError, TypeError, ValueError):
                        pass

            resend = []
            for motor_id in ordered:
                actual = status.get(motor_id)
                if actual is None:
                    if enabled:
                        if motor_id in self._neutral_fallback_enable_ids:
                            # A neutral goal was written before ON. A missing
                            # status reply must not block the ON resend.
                            self._torque_unconfirmed_ids.add(motor_id)
                            if time.monotonic() - last_on_retry_s.get(motor_id, 0.0) >= 0.05:
                                last_on_retry_s[motor_id] = time.monotonic()
                                resend.append(motor_id)
                            continue
                        # A missing torque reply is not proof of OFF. A fresh
                        # position read and goal seed make an ON resend safe.
                        self._pending_enable.add(motor_id)
                        self._stale_ids.add(motor_id)
                        if time.monotonic() - last_on_retry_s.get(motor_id, 0.0) < 0.05:
                            continue
                        try:
                            hardware_error = self._scalar(
                                self._controller.read_hardware_error_status(motor_id)
                            )
                            if hardware_error != 0.0:
                                continue
                            seeded_goals[motor_id] = self._seed_rejoin_goal(motor_id)
                        except (RuntimeError, OSError, TypeError, ValueError):
                            self._pending_enable.add(motor_id)
                            self._stale_ids.add(motor_id)
                            continue
                        self._torque_unconfirmed_ids.add(motor_id)
                        last_on_retry_s[motor_id] = time.monotonic()
                        resend.append(motor_id)
                    else:
                        resend.append(motor_id)
                    continue
                if not math.isfinite(actual) or actual not in (0.0, 1.0):
                    if enabled:
                        self._pending_enable.add(motor_id)
                        self._stale_ids.add(motor_id)
                    else:
                        resend.append(motor_id)
                    continue
                last_status[motor_id] = actual
                if actual == expected:
                    if enabled and motor_id in self._neutral_fallback_enable_ids:
                        self._neutral_fallback_enable_ids.discard(motor_id)
                        self._pending_enable.discard(motor_id)
                        self._torque_unconfirmed_ids.discard(motor_id)
                        pending.discard(motor_id)
                        continue
                    if enabled and (
                        motor_id in self._torque_unconfirmed_ids
                        or motor_id in self._pending_enable
                    ):
                        try:
                            measured = seeded_goals.get(motor_id)
                            if measured is None:
                                measured = self._seed_rejoin_goal(motor_id)
                        except (RuntimeError, OSError, TypeError, ValueError):
                            continue
                        self._finish_torque_rejoin(motor_id, measured, "ON again")
                    if not enabled:
                        self._torque_off_unconfirmed_ids.discard(motor_id)
                    pending.discard(motor_id)
                    continue
                if not enabled:
                    resend.append(motor_id)
                    continue
                # A confirmed OFF servo needs a fresh goal before an ON retry.
                self._torque_unconfirmed_ids.add(motor_id)
                if motor_id in self._neutral_fallback_enable_ids:
                    # The feedforward neutral goal replaced the old register
                    # before A, so a missing position reply is not a blocker.
                    if time.monotonic() - last_on_retry_s.get(motor_id, 0.0) >= 0.05:
                        last_on_retry_s[motor_id] = time.monotonic()
                        resend.append(motor_id)
                    continue
                if time.monotonic() - last_on_retry_s.get(motor_id, 0.0) < 0.05:
                    continue
                try:
                    hardware_error = self._scalar(
                        self._controller.read_hardware_error_status(motor_id)
                    )
                    if hardware_error != 0.0:
                        continue
                    seeded_goals[motor_id] = self._seed_rejoin_goal(motor_id)
                except (RuntimeError, OSError, TypeError, ValueError):
                    continue
                last_on_retry_s[motor_id] = time.monotonic()
                resend.append(motor_id)
            if resend:
                try:
                    self._controller.sync_write_torque_enable(
                        resend, [enabled] * len(resend)
                    )
                except (RuntimeError, OSError, ValueError) as exc:
                    last_write_error = exc
                else:
                    last_write_error = None
            if pending:
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

        if pending:
            unconfirmed = sorted(pending)
            if enabled:
                # Let confirmed joints start their neutral return. Remaining
                # joints are retried by the background torque-state poll.
                self._torque_unconfirmed_ids.update(pending)
            mismatched = sorted(
                motor_id for motor_id in pending
                if motor_id in last_status and last_status[motor_id] != expected
            )
            print(
                f"Servo torque {'ON' if enabled else 'OFF'} unconfirmed after 1 s: "
                f"IDs={unconfirmed}, confirmed mismatches={mismatched}",
                end="\r\n", flush=True,
            )
            if not enabled and (
                mismatched or (last_write_error is not None and len(unconfirmed) == len(ids))
            ):
                raise RuntimeError(
                    f"torque {'ON' if enabled else 'OFF'} not confirmed for IDs {unconfirmed}"
                )
        else:
            elapsed_ms = (time.monotonic() - started_s) * 1000.0
            print(
                f"Servo torque {'ON' if enabled else 'OFF'} confirmed "
                f"{len(ids)}/{len(ids)} in {elapsed_ms:.1f} ms",
                end="\r\n", flush=True,
            )

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        self._controller.sync_write_status_return_level(ids, levels)

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        self._write_goal_position(ids, positions, advance_stale=False)

    def sync_write_neutral_goal_position(self, ids: list[int], positions: list[float]) -> None:
        """Continue feedforward neutral return even without position replies."""
        self._write_goal_position(ids, positions, advance_stale=True)

    def _write_goal_position(
        self, ids: list[int], positions: list[float], *, advance_stale: bool
    ) -> None:
        if len(ids) != len(positions):
            raise ValueError("motor ID and goal counts differ")
        now = time.monotonic()
        live = []
        rejoining = []
        for motor_id, raw_pos in zip(ids, positions):
            if advance_stale:
                pos = float(raw_pos)
            else:
                if (
                    motor_id in self._pending_enable
                    or motor_id in self._torque_unconfirmed_ids
                ):
                    continue
                if motor_id in self._stale_ids:
                    if motor_id not in self._requested_torque_ids:
                        continue
                    pos = self._last_goals[motor_id]
                else:
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
                [self._to_servo(motor_id, pos) for motor_id, pos in live],
            )
            self._last_goals.update(live)
            if advance_stale:
                self._neutral_fallback_enable_ids.update(
                    motor_id for motor_id, _ in live
                    if (
                        motor_id not in self._requested_torque_ids
                        or motor_id in self._torque_unconfirmed_ids
                    )
                )
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
        measured = self._from_servo(motor_id, motor_position)
        if not math.isfinite(measured):
            raise ValueError("non-finite measured position")
        # The servo reading goes back unchanged, so the goal is exactly the
        # measured physical pose; only the cached goal is logical.
        self._controller.sync_write_goal_position([motor_id], [motor_position])
        self._last_goals[motor_id] = measured
        # RAM gains may have reset if the servo rebooted.  Preserve the A or
        # policy gain that the scheduler most recently requested for this ID.
        self._controller.sync_write_position_p_gain(
            [motor_id], [self._last_kp[motor_id]]
        )
        return measured

    def _poll_torque_state(self) -> None:
        now = time.monotonic()
        if not self._requested_torque_ids and not self._torque_off_unconfirmed_ids:
            return
        if now < self._torque_check_next_s:
            return
        self._torque_check_next_s = now + 0.1
        # A failed recovery check must not leave a joint without goal writes
        # until a complete 21-servo round robin finishes. Keep checking the
        # recovering joint on each 100 ms poll while retaining the normal scan.
        recovering = [
            motor_id for motor_id in self._torque_check_ids
            if (
                motor_id in self._torque_unconfirmed_ids
                and motor_id in self._requested_torque_ids
            ) or motor_id in self._torque_off_unconfirmed_ids
        ]
        recovery_id = None
        if recovering:
            recovery_id = recovering[self._torque_recovery_cursor % len(recovering)]
            self._torque_recovery_cursor += 1
            self._check_torque_state(recovery_id)
        motor_id = self._torque_check_ids[self._torque_check_cursor]
        self._torque_check_cursor = (self._torque_check_cursor + 1) % len(self._torque_check_ids)
        if motor_id != recovery_id:
            self._check_torque_state(motor_id)

    def _check_torque_state(self, motor_id: int) -> None:
        requested_on = motor_id in self._requested_torque_ids
        if (
            not requested_on
            and motor_id not in self._torque_off_unconfirmed_ids
        ):
            return

        try:
            enabled = self._scalar(self._controller.read_torque_enable(motor_id))
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            self._torque_warn(motor_id, f"read failed: {exc}")
            if requested_on and motor_id in self._neutral_fallback_enable_ids:
                # A neutral goal is already in the register. A missing status
                # packet must not prevent the ON request from being retried.
                try:
                    self._controller.sync_write_torque_enable([motor_id], [True])
                except (RuntimeError, OSError, TypeError, ValueError) as write_exc:
                    self._torque_warn(motor_id, f"ON retry failed: {write_exc}")
            if not requested_on:
                # OFF is idempotent. Retry it even when the status read is
                # missing, since a failed broadcast write has no other ACK.
                try:
                    self._controller.sync_write_torque_enable([motor_id], [False])
                except (RuntimeError, OSError, TypeError, ValueError) as write_exc:
                    self._torque_warn(motor_id, f"OFF retry failed: {write_exc}")
            return
        if not requested_on:
            if enabled == 0.0:
                self._torque_off_unconfirmed_ids.discard(motor_id)
                return
            if enabled != 1.0:
                self._torque_warn(motor_id, f"invalid register value: {enabled!r}")
                try:
                    self._controller.sync_write_torque_enable([motor_id], [False])
                except (RuntimeError, OSError, TypeError, ValueError) as exc:
                    self._torque_warn(motor_id, f"OFF retry failed: {exc}")
                return
            try:
                # Sync write still uses the broadcast packet ID, with only
                # this failed servo listed in its payload.
                self._controller.sync_write_torque_enable([motor_id], [False])
            except (RuntimeError, OSError, TypeError, ValueError) as exc:
                self._torque_warn(motor_id, f"OFF, torque-disable retry failed: {exc}")
            return
        if enabled == 1.0:
            if motor_id in self._torque_unconfirmed_ids:
                if motor_id in self._neutral_fallback_enable_ids:
                    self._finish_neutral_fallback_enable(motor_id)
                    return
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
        if motor_id in self._neutral_fallback_enable_ids:
            try:
                self._controller.sync_write_torque_enable([motor_id], [True])
                verified = self._scalar(self._controller.read_torque_enable(motor_id))
            except (RuntimeError, OSError, TypeError, ValueError) as exc:
                self._torque_warn(motor_id, f"OFF, neutral ON retry unconfirmed: {exc}")
                return
            if verified == 1.0:
                self._finish_neutral_fallback_enable(motor_id)
            else:
                self._torque_warn(
                    motor_id,
                    f"OFF, neutral ON retry did not take effect (register={verified!r})",
                )
            return
        try:
            hardware_error = self._scalar(
                self._controller.read_hardware_error_status(motor_id)
            )
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            self._torque_warn(motor_id, f"OFF, hardware error read failed: {exc}")
            return
        if hardware_error != 0.0:
            self._torque_warn(
                motor_id,
                f"OFF, hardware error={hardware_error}; enable deferred",
            )
            return
        # Never revive a servo against an old goal register.  The next
        # scheduler command is slew-limited from this fresh pose.
        try:
            measured = self._seed_rejoin_goal(motor_id)
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            self._torque_warn(motor_id, f"OFF, measured-goal seed failed: {exc}")
            return
        if motor_id not in self._requested_torque_ids:
            return
        try:
            self._controller.sync_write_torque_enable([motor_id], [True])
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            self._torque_warn(motor_id, f"OFF, torque-enable write failed: {exc}")
            return
        try:
            verified = self._scalar(
                self._controller.read_torque_enable(motor_id)
            )
        except (RuntimeError, OSError, TypeError, ValueError) as exc:
            self._torque_warn(motor_id, f"OFF, torque-enable verify failed: {exc}")
            return
        if verified != 1.0:
            self._torque_warn(
                motor_id,
                f"OFF, retry did not take effect (register={verified!r})",
            )
            return

        self._finish_torque_rejoin(motor_id, measured, "was OFF")

    def _finish_neutral_fallback_enable(self, motor_id: int) -> None:
        """Accept an ON reply after a feedforward neutral goal was seeded."""
        self._pending_enable.discard(motor_id)
        self._torque_unconfirmed_ids.discard(motor_id)
        self._neutral_fallback_enable_ids.discard(motor_id)
        self._torque_rejoin_last_s[motor_id] = time.monotonic()
        self._torque_warn_after_s.pop(motor_id, None)
        print(
            f"Servo torque feedback: id={motor_id} "
            f"name={self._id_to_name[motor_id]} ON; continuing neutral goal",
            end="\r\n", flush=True,
        )

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
            value = self._from_servo(
                motor_id, self._scalar(self._controller.read_present_position(motor_id))
            )
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
