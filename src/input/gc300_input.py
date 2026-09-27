# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Raspberry Pi Bluetooth input for the MSI FORCE GC300 WIRELESS.

This source reads the kernel joystick device directly.  It never powers the
motor bus down because a controller disappears: while disconnected it replays
the last operation, then zeros only the velocity target after one second so
policy feedback keeps running.  The game's A, B,
R3, left trigger, and sticks have the same roles as the PICO controls.  Y toggles
robot-local face following after the walking policy is enabled.  Right
trigger and grip have no robot action because the gamepad has no 6DoF poses.
"""

from __future__ import annotations

import array
from copy import deepcopy
import errno
import fcntl
import glob
import math
import os
import struct
import time
from pathlib import Path
from typing import Mapping

from input.input_source import InputSource, UserInput
from input.person_follow import PersonFollower


# Linux joystick API and input-event code values from linux/joystick.h and
# linux/input-event-codes.h.  The ioctls map joystick indices back to stable
# input-event codes, so a Bluetooth layout is never guessed from button order.
_JS_EVENT = struct.Struct("=IhBB")
_JS_BUTTON = 0x01
_JS_AXIS = 0x02
_JS_INIT = 0x80
_ABS_X = 0x00
_ABS_Y = 0x01
_ABS_Z = 0x02
_ABS_RZ = 0x05
_ABS_GAS = 0x09
_ABS_BRAKE = 0x0A
_BTN_SOUTH = 0x130  # A
_BTN_EAST = 0x131  # B
_BTN_NORTH = 0x133  # Y
_BTN_TL2 = 0x138   # digital LT fallback
_BTN_THUMBR = 0x13E  # R3
_ABS_COUNT = 0x40
_BTN_MISC = 0x100
_BUTTON_MAP_COUNT = 0x2FF - _BTN_MISC + 1
_IOC_READ = 2
_JSIOCGAXES = (_IOC_READ << 30) | (1 << 16) | (ord("j") << 8) | 0x11
_JSIOCGBUTTONS = (_IOC_READ << 30) | (1 << 16) | (ord("j") << 8) | 0x12
_JSIOCGAXMAP = (_IOC_READ << 30) | (_ABS_COUNT << 16) | (ord("j") << 8) | 0x32
_JSIOCGBTNMAP = (
    (_IOC_READ << 30) | (_BUTTON_MAP_COUNT * 2 << 16) | (ord("j") << 8) | 0x34
)
_PROBE_INTERVAL_NS = 500_000_000
_STICK_DEADZONE = 0.14
_LT_PRESS = 0.65
_LT_RELEASE = 0.45
_DISCONNECT_VELOCITY_TIMEOUT_NS = 1_000_000_000


def _ioctl_byte(fd: int, request: int) -> int:
    result = bytearray(1)
    fcntl.ioctl(fd, request, result, True)
    return result[0]


def _joystick_mapping(fd: int) -> tuple[dict[str, int | None], dict[str, int | None]]:
    axis_count = _ioctl_byte(fd, _JSIOCGAXES)
    button_count = _ioctl_byte(fd, _JSIOCGBUTTONS)
    axis_codes = bytearray(_ABS_COUNT)
    button_codes = array.array("H", [0]) * _BUTTON_MAP_COUNT
    fcntl.ioctl(fd, _JSIOCGAXMAP, axis_codes, True)
    fcntl.ioctl(fd, _JSIOCGBTNMAP, button_codes, True)
    axes_by_code = {
        int(code): index for index, code in enumerate(axis_codes[:axis_count])
    }
    buttons_by_code = {
        int(code): index for index, code in enumerate(button_codes[:button_count])
    }
    # Verified on the robot's GC300 over Bluetooth: ABS_Z/RZ are right stick
    # X/Y, ABS_BRAKE is LT, and ABS_GAS is the separate RT.  In particular,
    # ABS_Z must never be used as the LT fallback: moving the right stick
    # would then start walking.  A different HID layout needs explicit index
    # overrides after its controls have been identified on that device.
    lt_axis = (
        axes_by_code.get(_ABS_BRAKE) if _ABS_GAS in axes_by_code else None
    )
    axes: dict[str, int | None] = {
        "lx": axes_by_code.get(_ABS_X),
        "ly": axes_by_code.get(_ABS_Y),
        "rx": axes_by_code.get(_ABS_Z),
        "ry": axes_by_code.get(_ABS_RZ),
        "lt": lt_axis,
    }
    buttons: dict[str, int | None] = {
        "a": buttons_by_code.get(_BTN_SOUTH),
        "b": buttons_by_code.get(_BTN_EAST),
        "y": buttons_by_code.get(_BTN_NORTH),
        "r3": buttons_by_code.get(_BTN_THUMBR),
        "lt": buttons_by_code.get(_BTN_TL2),
    }
    return axes, buttons


def _radial_deadzone(x: float, y: float) -> tuple[float, float]:
    magnitude = min(1.0, math.hypot(x, y))
    if magnitude <= _STICK_DEADZONE:
        return 0.0, 0.0
    scale = (magnitude - _STICK_DEADZONE) / (1.0 - _STICK_DEADZONE)
    return x / magnitude * scale, y / magnitude * scale


class Gc300InputSource(InputSource):
    """Read a GC300 over Bluetooth without delaying the 50 Hz control loop.

    The pad may connect after the robot starts.  A/B/R3/Y only react to real
    edges, not the synthetic state replay sent when ``/dev/input/js*`` opens.
    B is also effective as a held level.  Reconnection preserves the prior
    torque/policy latch but requires LT to be released before walking resumes.
    """

    controls_motor_power = True

    def __init__(
        self,
        *,
        device_path: str | None = None,
        axis_indices: Mapping[str, int] | None = None,
        button_indices: Mapping[str, int] | None = None,
    ) -> None:
        self._device_path = device_path
        self._axis_override = dict(axis_indices or {})
        self._button_override = dict(button_indices or {})
        self._fd: int | None = None
        self._path = ""
        self._pending = b""
        self._axes: dict[str, int | None] = {}
        self._buttons: dict[str, int | None] = {}
        self._next_probe_ns = 0
        self._torque = False
        self._policy = False
        self._follow_enabled = False
        self._person_follower = PersonFollower()
        self._motion_inhibited = False
        self._gate_rearmed = False
        self._walk_rearmed = False
        self._walking = False
        self._a_edge_pending = False
        self._r3_edge_pending = False
        self._y_edge_pending = False
        self._b_press_pending = False
        self._off_after_disconnect = False
        self._disconnected_since_ns: int | None = None
        self._velocity_zeroed_after_disconnect = False
        self._resume_walk_if_quick = False
        self._sticks = {name: 0.0 for name in ("lx", "ly", "rx", "ry")}
        self._held = {name: False for name in ("a", "b", "r3", "y", "lt")}
        self._lt_axis = 0.0
        self.last_error: str | None = None
        self._last_output = UserInput(
            locomotion_policy="walk",
            torque_enabled=False,
            policy_enabled=False,
        )

    @staticmethod
    def _name(path: str) -> str | None:
        sysfs = Path("/sys/class/input") / Path(path).resolve().name / "device/name"
        try:
            return sysfs.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    def _connect(self) -> None:
        now_ns = time.monotonic_ns()
        if now_ns < self._next_probe_ns:
            return
        self._next_probe_ns = now_ns + _PROBE_INTERVAL_NS
        paths = (
            [self._device_path]
            if self._device_path is not None
            else sorted(glob.glob("/dev/input/js[0-9]*"))
        )
        for path in paths:
            name = self._name(path)
            if name is None or (self._device_path is None and "gc300" not in name.lower()):
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            except OSError as exc:
                message = f"Cannot open GC300 {path}: {exc}"
                if message != self.last_error:
                    print(message, end="\r\n", flush=True)
                self.last_error = message
                continue
            try:
                axes, buttons = _joystick_mapping(fd)
                axes.update(self._axis_override)
                buttons.update(self._button_override)
                missing = [
                    f"axis {key}" for key in ("lx", "ly", "rx", "ry")
                    if axes.get(key) is None
                ] + [
                    f"button {key}" for key in ("a", "b", "r3")
                    if buttons.get(key) is None
                ]
                if axes.get("lt") is None and buttons.get("lt") is None:
                    missing.append("left trigger")
                if missing:
                    raise ValueError(
                        f"missing {', '.join(missing)}; "
                        f"axis indices={axes}, button indices={buttons}"
                    )
            except (OSError, ValueError) as exc:
                message = f"Unsupported GC300 joystick {path} ({name}): {exc}"
                if message != self.last_error:
                    print(message, end="\r\n", flush=True)
                self.last_error = message
                os.close(fd)
                continue
            self._fd = fd
            self._path = path
            self._axes = axes
            self._buttons = buttons
            self._pending = b""
            short_disconnect = (
                self._disconnected_since_ns is not None
                and now_ns - self._disconnected_since_ns
                < _DISCONNECT_VELOCITY_TIMEOUT_NS
            )
            quick_reconnect = self._resume_walk_if_quick and short_disconnect
            self._reset_physical_state()
            if not short_disconnect and self._follow_enabled:
                self._follow_enabled = False
                self._person_follower.reset()
                print("GC300 follow: disarmed after long disconnect", flush=True)
            if quick_reconnect:
                self._walk_rearmed = True
                self._walking = True
            self._resume_walk_if_quick = False
            self._disconnected_since_ns = None
            self._velocity_zeroed_after_disconnect = False
            self.last_error = None
            print(f"GC300 connected: {name} ({path})", end="\r\n", flush=True)
            if buttons.get("y") is None:
                print("GC300 Y button is unavailable; manual controls remain active", flush=True)
            return

    def _reset_physical_state(self) -> None:
        self._sticks = {name: 0.0 for name in ("lx", "ly", "rx", "ry")}
        self._held = {name: False for name in ("a", "b", "r3", "y", "lt")}
        self._lt_axis = 0.0
        self._gate_rearmed = False
        self._walk_rearmed = False
        self._walking = False
        self._a_edge_pending = False
        self._r3_edge_pending = False
        self._y_edge_pending = False
        self._b_press_pending = False

    def _disconnect(self) -> None:
        if self._fd is not None:
            # A B press already sampled from the device remains an explicit
            # OFF request even if USB/Bluetooth vanishes in the same tick.
            if self._b_press_pending or self._held["b"]:
                self._off_after_disconnect = True
            self._resume_walk_if_quick = self._walk_rearmed and self._walking
            try:
                os.close(self._fd)
            except OSError:
                pass
            print("GC300 disconnected; replaying previous operation", end="\r\n", flush=True)
            self._disconnected_since_ns = time.monotonic_ns()
            self._velocity_zeroed_after_disconnect = False
        self._fd = None
        self._path = ""
        self._pending = b""
        self._next_probe_ns = 0
        self._reset_physical_state()

    def start(self) -> None:
        # Connection is discovered in read(), so the runtime may start before
        # the user turns the Bluetooth controller on.
        print("Waiting for GC300 Bluetooth joystick", end="\r\n", flush=True)

    def stop(self) -> None:
        self._disconnect()

    def set_motion_inhibited(self, inhibited: bool) -> None:
        if not isinstance(inhibited, bool):
            raise TypeError("inhibited must be boolean")
        self._motion_inhibited = inhibited
        if inhibited:
            self._follow_enabled = False
            self._person_follower.reset()
            self._walk_rearmed = False
            self._walking = False
            self._last_output.active_moves.clear()
            self._last_output.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            self._last_output.balance_only = False

    @staticmethod
    def _unit(value: int) -> float:
        return max(-1.0, min(1.0, value / 32767.0))

    def _event(self, value: int, event_type: int, number: int) -> None:
        initial = bool(event_type & _JS_INIT)
        kind = event_type & ~_JS_INIT
        if kind == _JS_AXIS:
            for name in ("lx", "ly", "rx", "ry"):
                if number == self._axes[name]:
                    self._sticks[name] = self._unit(value)
                    return
            if number == self._axes.get("lt"):
                # GC300's observed LT range is -32767 released to +32767
                # pressed. This remains valid if it reconnects while held.
                self._lt_axis = max(0.0, min(1.0, (value + 32767) / 65534.0))
            return
        if kind != _JS_BUTTON:
            return
        for name in ("a", "b", "r3", "y", "lt"):
            if number != self._buttons.get(name):
                continue
            pressed = value != 0
            rising = pressed and not self._held[name] and not initial
            self._held[name] = pressed
            if name == "b" and rising:
                self._b_press_pending = True
            elif name == "y" and rising:
                self._y_edge_pending = True
            elif self._gate_rearmed and rising:
                if name == "a":
                    self._a_edge_pending = True
                elif name == "r3":
                    self._r3_edge_pending = True
            if not any(self._held[key] for key in ("a", "b", "r3")):
                self._gate_rearmed = True
            return

    def _drain(self) -> None:
        fd = self._fd
        assert fd is not None
        try:
            # Bound the work per scheduler tick even if the HID driver floods
            # events; all joystick reads are nonblocking.
            for _ in range(8):
                try:
                    chunk = os.read(fd, _JS_EVENT.size * 64)
                except BlockingIOError:
                    break
                if not chunk:
                    self._disconnect()
                    return
                self._pending += chunk
                complete = len(self._pending) // _JS_EVENT.size * _JS_EVENT.size
                for offset in range(0, complete, _JS_EVENT.size):
                    _time_ms, value, event_type, number = _JS_EVENT.unpack_from(
                        self._pending, offset
                    )
                    self._event(value, event_type, number)
                self._pending = self._pending[complete:]
        except OSError as exc:
            if exc.errno not in (errno.ENODEV, errno.EIO, errno.EBADF):
                self.last_error = f"GC300 read failed: {exc}"
            self._disconnect()

    def _disconnected_output(self) -> UserInput:
        previous = deepcopy(self._last_output)
        if (
            self._disconnected_since_ns is not None
            and time.monotonic_ns() - self._disconnected_since_ns
            >= _DISCONNECT_VELOCITY_TIMEOUT_NS
        ):
            previous.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            if not self._velocity_zeroed_after_disconnect:
                print(
                    "GC300 disconnected for 1 s; velocity target set to zero",
                    end="\r\n",
                    flush=True,
                )
                self._velocity_zeroed_after_disconnect = True
        return previous

    def read(self) -> UserInput:
        if self._fd is not None and not os.path.exists(self._path):
            self._disconnect()
        if self._off_after_disconnect:
            self._off_after_disconnect = False
            self._torque = False
            self._policy = False
            self._follow_enabled = False
            self._person_follower.reset()
            self._last_output = UserInput(
                locomotion_policy="walk",
                torque_enabled=False,
                policy_enabled=False,
            )
            return deepcopy(self._last_output)
        if self._fd is None:
            self._connect()
        if self._fd is None:
            return self._disconnected_output()
        self._drain()
        if self._fd is None:
            if self._off_after_disconnect:
                self._off_after_disconnect = False
                self._torque = False
                self._policy = False
                self._follow_enabled = False
                self._person_follower.reset()
                self._last_output = UserInput(
                    locomotion_policy="walk",
                    torque_enabled=False,
                    policy_enabled=False,
                )
            return self._disconnected_output()

        # A physical B level is explicit OFF even if it arrived in a synthetic
        # init snapshot.  A or R3 init snapshot never raises their gates.
        was_policy_active = self._torque and self._policy
        if self._held["b"] or self._b_press_pending:
            self._torque = False
            self._policy = False
            self._follow_enabled = False
        elif self._a_edge_pending:
            self._torque = True
            self._policy = False
            self._follow_enabled = False
            self._walk_rearmed = False
            self._walking = False
        elif self._r3_edge_pending and self._torque:
            self._policy = not self._policy
            if not self._policy:
                self._follow_enabled = False
        elif self._y_edge_pending and was_policy_active:
            self._follow_enabled = not self._follow_enabled
            self._person_follower.reset()
            print(
                "GC300 follow: ON" if self._follow_enabled else "GC300 follow: OFF",
                flush=True,
            )
        if not self._follow_enabled:
            self._person_follower.reset()
        self._a_edge_pending = False
        self._r3_edge_pending = False
        self._y_edge_pending = False
        self._b_press_pending = False
        lt = max(self._lt_axis, float(self._held["lt"]))
        if lt <= _LT_RELEASE and not self._motion_inhibited:
            self._walk_rearmed = True
        if self._walk_rearmed and not self._motion_inhibited:
            self._walking = lt >= (_LT_RELEASE if self._walking else _LT_PRESS)
        else:
            self._walking = False

        active_policy = self._torque and self._policy and not self._motion_inhibited
        walking = active_policy and self._walking
        following = active_policy and self._follow_enabled
        lx, ly = _radial_deadzone(self._sticks["lx"], self._sticks["ly"])
        rx, _ry = _radial_deadzone(self._sticks["rx"], self._sticks["ry"])
        velocity = {
            "vx": -ly if walking else 0.0,
            "vy": -lx if walking else 0.0,
            "vtheta": -rx if walking else 0.0,
        }
        head_orientation = None
        if following:
            follow_command = self._person_follower.command()
            head_orientation = follow_command.head_orientation
            if not walking:
                velocity = follow_command.velocity
        self._last_output = UserInput(
            active_moves=(
                {"walk", "hmd_head"} if following else {"walk"}
            ) if active_policy else set(),
            velocity=velocity,
            locomotion_policy="walk",
            balance_only=active_policy and not walking and not following,
            head_orientation=head_orientation,
            arm_tracking_enabled=False,
            torque_enabled=self._torque,
            policy_enabled=self._policy,
        )
        return deepcopy(self._last_output)

    def set_head_telemetry(
        self,
        *,
        head: float,
        neck_roll: float,
        neck_pitch: float,
        trunk_roll: float,
        trunk_pitch: float,
    ) -> None:
        _ = (neck_roll, trunk_roll)
        self._person_follower.set_head_telemetry(
            head=head, neck_pitch=neck_pitch, trunk_pitch=trunk_pitch
        )
