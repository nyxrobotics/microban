# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Receives UserInput over UDP from the teleop bridge (microban_teleop, a separate PC).

Wire format: one JSON object per UDP packet, all fields optional —
    {
      "velocity": {"vx": 0.3, "vy": 0.0, "vtheta": 0.0},
      "active_moves": ["walk"],
      "head_orientation": {"roll": 0.0, "pitch": -0.2} | null,
      "foot_target": {"left": [dx, dy, dz], "right": [dx, dy, dz]} | null,
      "hand_target": {"left": [dx, dy, dz] | null, "right": [dx, dy, dz] | null} | null
    }

A field that's missing or null falls back to UserInput's own default (neutral), per
this project's rule that a missing target degrades gracefully rather than requiring
every field present. If no packet arrives within `stale_after_s`, read() falls back to
a fully-neutral UserInput (velocity zero, no targets) instead of repeating a stale
command — a dead network link must not leave the robot walking on its last order.
"""

import json
import socket
import threading
import time

from input.input_source import InputSource, UserInput


def _tuple3(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


class NetworkInputSource(InputSource):
    """Non-blocking UDP JSON input source. See module docstring for the wire format."""

    def __init__(self, port: int = 5555, stale_after_s: float = 0.3) -> None:
        self._port = port
        self._stale_after_s = stale_after_s

        self._state = UserInput()
        self._last_recv_s = 0.0
        self._lock = threading.Lock()

        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"NetworkInputSource listening on UDP :{self._port}", end="\r\n", flush=True)

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def read(self) -> UserInput:
        with self._lock:
            if (time.monotonic() - self._last_recv_s) > self._stale_after_s:
                return UserInput()
            return UserInput(
                active_moves=set(self._state.active_moves),
                velocity=dict(self._state.velocity),
                show_imu=self._state.show_imu,
                head_orientation=dict(self._state.head_orientation) if self._state.head_orientation else None,
                foot_target=dict(self._state.foot_target) if self._state.foot_target else None,
                hand_target=dict(self._state.hand_target) if self._state.hand_target else None,
            )

    # ------------------------------------------------------------------
    # Internal

    def _read_loop(self) -> None:
        sock = self._sock
        assert sock is not None
        while self._running:
            try:
                data, _addr = sock.recvfrom(4096)
            except (socket.timeout, OSError):
                continue
            try:
                packet = json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            self._apply(packet)

    def _apply(self, packet: dict) -> None:
        velocity = packet.get("velocity")
        active_moves = packet.get("active_moves")
        head_orientation = packet.get("head_orientation")
        foot_target = packet.get("foot_target")
        hand_target = packet.get("hand_target")

        with self._lock:
            if velocity is not None:
                self._state.velocity = {
                    "vx": float(velocity.get("vx", 0.0)),
                    "vy": float(velocity.get("vy", 0.0)),
                    "vtheta": float(velocity.get("vtheta", 0.0)),
                }
            if active_moves is not None:
                self._state.active_moves = set(active_moves)
            if head_orientation is not None:
                self._state.head_orientation = {
                    "roll": float(head_orientation.get("roll", 0.0)),
                    "pitch": float(head_orientation.get("pitch", 0.0)),
                }
            else:
                self._state.head_orientation = None
            if foot_target is not None:
                self._state.foot_target = {
                    "left": _tuple3(foot_target.get("left")) or (0.0, 0.0, 0.0),
                    "right": _tuple3(foot_target.get("right")) or (0.0, 0.0, 0.0),
                }
            else:
                self._state.foot_target = None
            if hand_target is not None:
                self._state.hand_target = {
                    "left": _tuple3(hand_target.get("left")),
                    "right": _tuple3(hand_target.get("right")),
                }
            else:
                self._state.hand_target = None
            self._last_recv_s = time.monotonic()
