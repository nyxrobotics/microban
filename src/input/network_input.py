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
      "locomotion_policy": "walk" | "pico_teleop",
      "head_orientation": {"roll": 0.0, "pitch": -0.2, "yaw": 0.3} | null,
      "head_yaw_front": false,
      "body_target_contract": "microban_pico_offsets_v1",
      "body_target_safety_margin": 0.8,
      "foot_target": {"left": [dx, dy, dz], "right": [dx, dy, dz]} | null,
      "hand_target": {"left": [dx, dy, dz] | null, "right": [dx, dy, dz] | null} | null
    }

Every packet replaces the previous state; omitted fields are neutral. If no packet
arrives within `stale_after_s`, read() returns a fully-neutral UserInput. After start
or a timeout, walking stays disarmed until at least one released-trigger snapshot
(``"walk"`` absent) arrives. A dead/reconnecting link therefore cannot resume walking
from a trigger that was held before the interruption.
Hybrid ``pico_teleop`` walk snapshots use fixed policy-session calibration
offsets in the robot trunk frame (+X forward, +Y left, +Z up), in metres. They
must declare the exact contract and 0.8 safety margin above, provide complete
paired feet and hands, and stay inside the bridge's 80% training envelope.
Any mismatch stops and disarms walking immediately, clears both target pairs,
and requires another trigger release.
"""

import json
import math
import socket
import threading
import time

from input.input_source import InputSource, UserInput


_NETWORK_MOVES = frozenset({"walk", "hmd_head"})
_LOCOMOTION_POLICIES = frozenset({"walk", "pico_teleop"})
_PICO_BODY_TARGET_CONTRACT = "microban_pico_offsets_v1"
_PICO_BODY_TARGET_SAFETY_MARGIN = 0.8
_PICO_FOOT_TARGET_LOWER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (-0.03, -0.03, 0.0)
)
_PICO_FOOT_TARGET_UPPER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (0.03, 0.03, 0.05)
)
_PICO_HAND_TARGET_LOWER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (-0.08, -0.08, -0.08)
)
_PICO_HAND_TARGET_UPPER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (0.08, 0.08, 0.08)
)


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
    if (
        not isinstance(value, (list, tuple))
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 3
    ):
        raise TypeError("target vector must contain exactly three numbers")
    if any(
        not isinstance(component, (int, float)) or isinstance(component, bool)
        for component in value
    ):
        raise TypeError("target vector components must be JSON numbers")
    return (_finite_float(value[0]), _finite_float(value[1]), _finite_float(value[2]))


def _target_pair_in_bounds(
    value: dict[str, tuple[float, float, float] | None] | None,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> bool:
    if value is None or set(value) != {"left", "right"}:
        return False
    for side in ("left", "right"):
        vector = value[side]
        if vector is None or any(
            component < lower[index] or component > upper[index]
            for index, component in enumerate(vector)
        ):
            return False
    return True


def _matches_pico_safety_margin(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(numeric_value)
        and numeric_value == _PICO_BODY_TARGET_SAFETY_MARGIN
    )


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
                locomotion_policy=self._state.locomotion_policy,
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

        locomotion_policy = packet.get("locomotion_policy", "walk")
        if locomotion_policy not in _LOCOMOTION_POLICIES:
            raise ValueError("unsupported locomotion_policy")
        incoming_pico_packet = locomotion_policy == "pico_teleop"
        pico_walk_requested = (
            incoming_pico_packet and "walk" in requested_moves
        )

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

        target_payload_valid = True
        try:
            foot_target = packet.get("foot_target")
            if foot_target is not None:
                if not isinstance(foot_target, dict):
                    raise TypeError("foot_target must be an object or null")
                left_foot_target = _tuple3(foot_target.get("left"))
                right_foot_target = _tuple3(foot_target.get("right"))
                complete_foot_target = (
                    set(foot_target) == {"left", "right"}
                    and left_foot_target is not None
                    and right_foot_target is not None
                )
                parsed_foot_target = {
                    "left": left_foot_target or (0.0, 0.0, 0.0),
                    "right": right_foot_target or (0.0, 0.0, 0.0),
                }
            else:
                complete_foot_target = False
                parsed_foot_target = None

            hand_target = packet.get("hand_target")
            if hand_target is not None:
                if not isinstance(hand_target, dict):
                    raise TypeError("hand_target must be an object or null")
                left_hand_target = _tuple3(hand_target.get("left"))
                right_hand_target = _tuple3(hand_target.get("right"))
                complete_hand_target = (
                    set(hand_target) == {"left", "right"}
                    and left_hand_target is not None
                    and right_hand_target is not None
                )
                parsed_hand_target = {
                    "left": left_hand_target,
                    "right": right_hand_target,
                }
            else:
                complete_hand_target = False
                parsed_hand_target = None
        except (IndexError, TypeError, ValueError, OverflowError):
            if not incoming_pico_packet:
                raise
            # Contract failures must become a new neutral/disarmed state below;
            # raising here would leave the previous walking state latched.
            target_payload_valid = False
            complete_foot_target = False
            complete_hand_target = False
            parsed_foot_target = None
            parsed_hand_target = None

        pico_metadata_valid = (
            packet.get("body_target_contract") == _PICO_BODY_TARGET_CONTRACT
            and _matches_pico_safety_margin(
                packet.get("body_target_safety_margin")
            )
        )
        if pico_walk_requested:
            pico_body_target_valid = (
                pico_metadata_valid
                and target_payload_valid
                and complete_foot_target
                and complete_hand_target
                and _target_pair_in_bounds(
                    parsed_foot_target,
                    _PICO_FOOT_TARGET_LOWER,
                    _PICO_FOOT_TARGET_UPPER,
                )
                and _target_pair_in_bounds(
                    parsed_hand_target,
                    _PICO_HAND_TARGET_LOWER,
                    _PICO_HAND_TARGET_UPPER,
                )
            )
        else:
            # The native bridge sends target-null snapshots while the deadman is
            # released. Requiring the same metadata keeps release/re-arm packets
            # inside the negotiated contract without requiring impossible pairs.
            pico_body_target_valid = (
                pico_metadata_valid
                and target_payload_valid
                and parsed_foot_target is None
                and parsed_hand_target is None
            )

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

            mode_changed = locomotion_policy != self._state.locomotion_policy
            force_disarmed = mode_changed
            if mode_changed:
                self._walk_armed = False
            if mode_changed and "walk" in requested_moves:
                # A policy may never change while either policy owns the joints.
                # Ignore the requested mode and force a deadman release instead.
                locomotion_policy = self._state.locomotion_policy
                requested_moves.discard("walk")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
                parsed_foot_target = None
                parsed_hand_target = None
                mode_changed = False

            if incoming_pico_packet and not pico_body_target_valid:
                # Never raise and retain an older walking snapshot for a bad
                # body-target contract. Apply this packet as an immediate stop
                # and require an explicit later release before re-arming.
                self._walk_armed = False
                force_disarmed = True
                requested_moves.discard("walk")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
                parsed_foot_target = None
                parsed_hand_target = None

            if self._motion_inhibited:
                self._walk_armed = False
                requested_moves.discard("walk")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            elif "walk" not in requested_moves and not force_disarmed:
                self._walk_armed = True
            elif not self._walk_armed:
                requested_moves.discard("walk")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

            self._state = UserInput(
                active_moves=requested_moves,
                velocity=parsed_velocity,
                locomotion_policy=locomotion_policy,
                head_orientation=parsed_orientation,
                head_yaw_front=head_yaw_front,
                foot_target=parsed_foot_target,
                hand_target=parsed_hand_target,
            )
            self._last_recv_s = time.monotonic()
