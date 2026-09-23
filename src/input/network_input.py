# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Receives UserInput over UDP from the teleop bridge (microban_teleop, a separate PC).

Wire format: one complete JSON snapshot per UDP packet —
    {
      "version": 1,
      "session_id": "random-per-bridge-start",
      "seq": 42,
      "velocity": {"vx": 0.3, "vy": 0.0, "vtheta": 0.0},
      "active_moves": ["walk"],
      "head_orientation": {"roll": 0.0, "pitch": -0.2, "yaw": 0.3} | null,
      "head_yaw_front": false,
      "foot_target": {"left": [dx, dy, dz], "right": [dx, dy, dz]} | null,
      "hand_target": {"left": [dx, dy, dz] | null, "right": [dx, dy, dz] | null} | null
    }

Every packet replaces the previous state; omitted fields are neutral. If no packet
arrives within `stale_after_s`, read() returns a fully-neutral UserInput. After start
or a timeout, walking stays disarmed until at least one released-trigger snapshot
(``"walk"`` absent) arrives. A dead/reconnecting link therefore cannot resume walking
from a trigger that was held before the interruption.
"""

import json
import math
import socket
import threading
import time

from input.input_source import InputSource, UserInput


_NETWORK_MOVES = frozenset({"walk", "hmd_head"})


def _finite_float(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite numeric input")
    return result


def _unit_float(value) -> float:
    return max(-1.0, min(1.0, _finite_float(value)))


def _tuple3(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return (_finite_float(value[0]), _finite_float(value[1]), _finite_float(value[2]))


class NetworkInputSource(InputSource):
    """Non-blocking UDP JSON input source. See module docstring for the wire format."""

    def __init__(
        self,
        port: int = 5555,
        stale_after_s: float = 0.3,
        allowed_remote: str | None = None,
    ) -> None:
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not math.isfinite(stale_after_s) or not 0.05 <= stale_after_s <= 0.5:
            raise ValueError("stale_after_s must be finite and between 0.05 and 0.5 seconds")
        self._port = port
        self._stale_after_s = stale_after_s
        self._allowed_remote = (
            socket.gethostbyname(allowed_remote) if allowed_remote else None
        )

        self._state = UserInput()
        self._last_recv_s = 0.0
        self._lock = threading.Lock()

        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._walk_armed = False
        self._motion_inhibited = False
        self._session_id: str | None = None
        self._last_seq = -1
        # Remember prior bridge instances so delayed datagrams cannot switch control
        # back to an older session after a clean restart/takeover.
        self._retired_sessions: list[str] = []

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        peer = self._allowed_remote or "any sender"
        print(
            f"NetworkInputSource listening on UDP :{self._port} ({peer})",
            end="\r\n",
            flush=True,
        )

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.3)
        self._thread = None

    def read(self) -> UserInput:
        with self._lock:
            if (time.monotonic() - self._last_recv_s) > self._stale_after_s:
                self._state = UserInput()
                self._walk_armed = False
                return UserInput()
            return UserInput(
                active_moves=set(self._state.active_moves),
                velocity=dict(self._state.velocity),
                show_imu=self._state.show_imu,
                head_orientation=dict(self._state.head_orientation) if self._state.head_orientation else None,
                head_yaw_front=self._state.head_yaw_front,
                foot_target=dict(self._state.foot_target) if self._state.foot_target else None,
                hand_target=dict(self._state.hand_target) if self._state.hand_target else None,
            )

    def set_motion_inhibited(self, inhibited: bool) -> None:
        """Suppress walking and require a post-recovery trigger release.

        The flag is checked under the same lock as packet application, closing the
        race where a held-trigger packet could re-arm walking between the scheduler's
        safety decision and its next input read.
        """
        if not isinstance(inhibited, bool):
            raise TypeError("inhibited must be boolean")
        with self._lock:
            if inhibited == self._motion_inhibited:
                return
            self._motion_inhibited = inhibited
            self._walk_armed = False
            self._state.active_moves.discard("walk")
            self._state.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

    # ------------------------------------------------------------------
    # Internal

    def _read_loop(self) -> None:
        sock = self._sock
        assert sock is not None
        while self._running:
            try:
                data, addr = sock.recvfrom(16384)
            except (socket.timeout, OSError):
                continue
            if self._allowed_remote is not None and addr[0] != self._allowed_remote:
                continue
            try:
                packet = json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            try:
                self._apply(packet)
            except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                # A malformed datagram must never kill the receiver thread and leave
                # the scheduler unknowingly running on its previous command.
                continue

    def _apply(self, packet: dict) -> None:
        if not isinstance(packet, dict):
            raise TypeError("packet must be a JSON object")
        if packet.get("version") != 1:
            raise ValueError("unsupported protocol version")

        session_id = packet.get("session_id")
        seq = packet.get("seq")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise TypeError("session_id must be a non-empty string of at most 128 characters")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise TypeError("seq must be a non-negative integer")

        velocity = packet.get("velocity") or {}
        if not isinstance(velocity, dict):
            raise TypeError("velocity must be an object")
        parsed_velocity = {
            "vx": _unit_float(velocity.get("vx", 0.0)),
            "vy": _unit_float(velocity.get("vy", 0.0)),
            "vtheta": _unit_float(velocity.get("vtheta", 0.0)),
        }

        move_values = packet.get("active_moves") or []
        if not isinstance(move_values, list) or not all(isinstance(v, str) for v in move_values):
            raise TypeError("active_moves must be a list of strings")
        requested_moves = set(move_values)
        if not requested_moves <= _NETWORK_MOVES:
            raise ValueError("network packet requested an unsupported move")

        orientation = packet.get("head_orientation")
        if orientation is not None:
            if not isinstance(orientation, dict):
                raise TypeError("head_orientation must be an object or null")
            parsed_orientation = {
                axis: _finite_float(orientation.get(axis, 0.0))
                for axis in ("roll", "pitch", "yaw")
            }
        else:
            parsed_orientation = None

        head_yaw_front = packet.get("head_yaw_front", False)
        if not isinstance(head_yaw_front, bool):
            raise TypeError("head_yaw_front must be boolean")

        foot_target = packet.get("foot_target")
        if foot_target is not None:
            if not isinstance(foot_target, dict):
                raise TypeError("foot_target must be an object or null")
            parsed_foot_target = {
                "left": _tuple3(foot_target.get("left")) or (0.0, 0.0, 0.0),
                "right": _tuple3(foot_target.get("right")) or (0.0, 0.0, 0.0),
            }
        else:
            parsed_foot_target = None

        hand_target = packet.get("hand_target")
        if hand_target is not None:
            if not isinstance(hand_target, dict):
                raise TypeError("hand_target must be an object or null")
            parsed_hand_target = {
                "left": _tuple3(hand_target.get("left")),
                "right": _tuple3(hand_target.get("right")),
            }
        else:
            parsed_hand_target = None

        with self._lock:
            if session_id != self._session_id:
                if session_id in self._retired_sessions:
                    return
                # A bridge handover must begin from a released walk deadman. This
                # permits restart at any sequence number while rejecting held-trigger
                # takeover packets.
                if "walk" in requested_moves:
                    return
                if self._session_id is not None:
                    self._retired_sessions.append(self._session_id)
                    del self._retired_sessions[:-16]
                self._session_id = session_id
                self._last_seq = -1
                self._walk_armed = False
            if seq <= self._last_seq:
                return
            self._last_seq = seq

            if self._motion_inhibited:
                self._walk_armed = False
                requested_moves.discard("walk")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            elif "walk" not in requested_moves:
                self._walk_armed = True
            elif not self._walk_armed:
                requested_moves.discard("walk")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

            self._state = UserInput(
                active_moves=requested_moves,
                velocity=parsed_velocity,
                head_orientation=parsed_orientation,
                head_yaw_front=head_yaw_front,
                foot_target=parsed_foot_target,
                hand_target=parsed_hand_target,
            )
            self._last_recv_s = time.monotonic()
