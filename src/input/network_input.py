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
      "head_orientation": {"roll": 0.0, "pitch": -0.2, "yaw": 0.3} | null,  # gated by the right-trigger deadman below, same as hand_target/arm_joint_target
      "head_yaw_front": false,
      "body_target_contract": "microban_pico_offsets_v2_both_feet_stationary",
      "body_target_safety_margin": 0.8,
      "foot_target": {"left": [dx, dy, dz], "right": [dx, dy, dz]} | null,
      "hand_target": {"left": [dx, dy, dz] | null, "right": [dx, dy, dz] | null} | null,
      "arm_tracking_enabled": false,
      "arm_joint_target": {"left": [pitch, roll, elbow], "right": [...]},
      "torque_enabled": false,
      "torque_off_requested": false,
      "policy_enabled": false
    }

Every packet replaces the previous motion state; omitted fields are neutral. Before
the first authenticated packet, read() leaves the startup hardware gate disabled.
During a UDP gap, it replays the last accepted operator input so the policy keeps
its feedback loop running. A new bridge session still requires a released-trigger
snapshot (``"walk"`` absent) before walking can arm.
Hybrid ``pico_teleop`` walk snapshots use fixed policy-session calibration
offsets in the robot trunk frame (+X forward, +Y left, +Z up), in metres. They
must declare the exact contract and 0.8 safety margin above, provide complete
paired feet and hands, and stay inside the bridge's 80% training envelope.
After the bridge projects a support foot into its floor band, two active foot
offsets use the narrower simultaneous-foot envelope and require an exactly zero
twist command.
Any mismatch clears both target pairs and downgrades that snapshot to the proven
``walk`` policy without discarding its trigger or joystick command. The optional
tracking channel can therefore degrade without disabling basic locomotion.

A second, unrelated packet shape is a stateless bridge-side latency probe,
handled before session/sequence validation and never touching UserInput:
    {"type": "clock_ping", "nonce": "<opaque string, at most 64 chars>"}
which is echoed back verbatim as ``{"type": "clock_pong", "nonce": ...}`` to
the sender's address. The bridge times its own round trip; the robot reports
no timestamp of its own and needs no synchronized clock. When the scheduler
has supplied recent head/neck telemetry (see ``set_head_telemetry``), the
reply also carries it so the bridge can measure real camera-reprojection lag
instead of simulating it:
    {"type": "clock_pong", "nonce": ..., "head": 0.1, "neck_roll": 0.0,
     "neck_pitch": -0.05, "trunk_roll": 0.0, "trunk_pitch": 0.01}
Telemetry fields are omitted entirely if none has been supplied yet.
"""

import json
import ipaddress
import math
from pathlib import Path
import socket
import threading
import time

from input.input_source import InputSource, UserInput
from pico_arm_contract import (
    PICO_ARM_HOME_RAD,
    PICO_ARM_SIDES,
    parse_pico_arm_joint_target,
)

_NETWORK_MOVES = frozenset({"walk", "hmd_head", "pico_arms"})
_LOCOMOTION_POLICIES = frozenset({"walk", "pico_teleop"})
_PICO_BODY_TARGET_CONTRACT = "microban_pico_offsets_v2_both_feet_stationary"
_PICO_BODY_TARGET_SAFETY_MARGIN = 0.8
_PICO_SUPPORT_FOOT_FLOOR_BAND_M = 0.0025
_PICO_FOOT_TARGET_LOWER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (-0.03, -0.03, 0.0)
)
_PICO_FOOT_TARGET_UPPER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (0.03, 0.03, 0.05)
)
_PICO_SIMULTANEOUS_BOTH_FEET_LOWER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (-0.01, -0.01, 0.0)
)
_PICO_SIMULTANEOUS_BOTH_FEET_UPPER = tuple(
    value * _PICO_BODY_TARGET_SAFETY_MARGIN for value in (0.01, 0.01, 0.02)
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


def _simultaneous_both_feet_valid(
    foot_target: dict[str, tuple[float, float, float] | None] | None,
    velocity: dict[str, float],
) -> bool:
    """Apply the narrow stationary contract when both offsets are non-zero."""

    if foot_target is None:
        return False
    left = foot_target.get("left")
    right = foot_target.get("right")
    if left is None or right is None:
        return False
    both_active = all(
        any(abs(component) > 1.0e-12 for component in vector)
        for vector in (left, right)
    )
    if not both_active:
        return True
    if any(velocity[axis] != 0.0 for axis in ("vx", "vy", "vtheta")):
        return False
    return _target_pair_in_bounds(
        foot_target,
        _PICO_SIMULTANEOUS_BOTH_FEET_LOWER,
        _PICO_SIMULTANEOUS_BOTH_FEET_UPPER,
    )


def _support_foot_floor_projection_valid(
    foot_target: dict[str, tuple[float, float, float] | None] | None,
) -> bool:
    """Require the bridge's support-floor band to arrive as exact XYZ zero."""

    if foot_target is None:
        return False
    for side in ("left", "right"):
        vector = foot_target.get(side)
        if vector is None:
            return False
        if vector[2] <= _PICO_SUPPORT_FOOT_FLOOR_BAND_M and vector != (
            0.0,
            0.0,
            0.0,
        ):
            return False
    return True


class NetworkInputSource(InputSource):
    """Non-blocking UDP JSON input source. See module docstring for the wire format."""

    # A network-controlled real robot must never energize itself at process
    # startup.  main.py uses this marker to start limp and Scheduler uses the
    # packet's B/A/R3 state as the only way out of that state.
    controls_motor_power = True

    def __init__(
        self,
        port: int = 5555,
        stale_after_s: float = 0.3,
        allowed_remote: str | None = None,
        allowed_remote_file: str | None = None,
    ) -> None:
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise ValueError("port must be between 1 and 65535")
        if not math.isfinite(stale_after_s) or not 0.05 <= stale_after_s <= 0.5:
            raise ValueError(
                "stale_after_s must be finite and between 0.05 and 0.5 seconds"
            )
        self._port = port
        self._stale_after_s = stale_after_s
        self._allowed_remote = (
            socket.gethostbyname(allowed_remote) if allowed_remote else None
        )
        self._allowed_remote_file = Path(allowed_remote_file) if allowed_remote_file else None
        if self._allowed_remote_file is not None and not self._allowed_remote_file.is_absolute():
            raise ValueError("allowed_remote_file must be an absolute path")
        self._allowed_remote_file_error = False

        self._state = UserInput()
        self._last_recv_s = 0.0
        self._has_operator_state = False
        self._hold_once = False
        self._lock = threading.Lock()
        self._head_telemetry: dict[str, float] | None = None

        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._walk_armed = False
        self._arm_armed = False
        self._motion_inhibited = False
        self._session_id: str | None = None
        self._last_seq = -1
        # Remember prior bridge instances so delayed datagrams cannot switch control
        # back to an older session after a clean restart/takeover.
        self._retired_sessions: list[str] = []

    def start(self) -> None:
        if self._allowed_remote_file is not None:
            self._reload_allowed_remote()
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
            # Force one scheduler tick to retain the physical goal when the
            # allowed sender changes, even if a new UDP packet arrives first.
            # An explicit B request from the new sender still takes priority.
            force_hold = self._hold_once
            self._hold_once = False
            if force_hold and self._state.torque_enabled is False and not self._state.hold_last_targets:
                force_hold = False
            if force_hold:
                # A new allowed sender must begin with released triggers.  A
                # temporary packet gap from the same sender is different: it
                # preserves the operator's deadman latches and motion state.
                self._state = UserInput()
                self._last_recv_s = 0.0
                self._has_operator_state = False
                self._walk_armed = False
                self._arm_armed = False
            if force_hold or self._last_recv_s == 0.0:
                # Before the first authenticated packet (or after authority
                # changes), leave the startup hardware gate alone. Once a
                # packet has arrived, read() replays its operator input during
                # UDP loss so the feedback policy keeps running.
                return UserInput(
                    torque_enabled=None,
                    policy_enabled=None,
                    hold_last_targets=True,
                )
            return UserInput(
                active_moves=set(self._state.active_moves),
                velocity=dict(self._state.velocity),
                show_imu=self._state.show_imu,
                locomotion_policy=self._state.locomotion_policy,
                learned_policy_degraded=self._state.learned_policy_degraded,
                balance_only=self._state.balance_only,
                head_orientation=dict(self._state.head_orientation)
                if self._state.head_orientation
                else None,
                head_yaw_front=self._state.head_yaw_front,
                foot_target=dict(self._state.foot_target)
                if self._state.foot_target
                else None,
                hand_target=dict(self._state.hand_target)
                if self._state.hand_target
                else None,
                arm_tracking_enabled=self._state.arm_tracking_enabled,
                arm_joint_target=dict(self._state.arm_joint_target)
                if self._state.arm_joint_target
                else None,
                torque_enabled=self._state.torque_enabled,
                policy_enabled=self._state.policy_enabled,
                hold_last_targets=self._state.hold_last_targets,
                getup_armed=self._state.getup_armed,
            )

    def set_motion_inhibited(self, inhibited: bool) -> None:
        """Suppress walking/arms and require post-recovery trigger releases.

        The flag is checked under the same lock as packet application, closing the
        race where a held-trigger packet could re-arm motion between the scheduler's
        safety decision and its next input read.
        """
        if not isinstance(inhibited, bool):
            raise TypeError("inhibited must be boolean")
        with self._lock:
            if inhibited == self._motion_inhibited:
                return
            self._motion_inhibited = inhibited
            self._walk_armed = False
            self._arm_armed = False
            self._state.active_moves.discard("walk")
            self._state.active_moves.discard("pico_arms")
            self._state.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            self._state.arm_tracking_enabled = False
            self._state.arm_joint_target = None

    def set_head_telemetry(
        self,
        *,
        head: float,
        neck_roll: float,
        neck_pitch: float,
        trunk_roll: float,
        trunk_pitch: float,
    ) -> None:
        """Latest measured head/neck joint angles and trunk tilt.

        Read-only cache for the next clock_pong reply (see module docstring);
        never touches UserInput or control state. Silently ignored if any
        value is non-finite so a transient IMU/read glitch cannot poison the
        bridge's camera-reprojection lag estimate with a bogus telemetry
        sample.
        """
        values = {
            "head": float(head),
            "neck_roll": float(neck_roll),
            "neck_pitch": float(neck_pitch),
            "trunk_roll": float(trunk_roll),
            "trunk_pitch": float(trunk_pitch),
        }
        if not all(math.isfinite(value) for value in values.values()):
            return
        with self._lock:
            self._head_telemetry = values

    # ------------------------------------------------------------------
    # Internal

    def _reload_allowed_remote(self) -> None:
        """Reload the single robot sender without interrupting motor power."""
        path = self._allowed_remote_file
        if path is None:
            return
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing or linked allowlist file")
            if path.stat().st_size > 8192:
                raise ValueError("allowlist file is too large")
            entries = [
                line.removeprefix("MICROBAN_NETWORK_ALLOWED_IP=").strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("MICROBAN_NETWORK_ALLOWED_IP=")
            ]
            if len(entries) != 1:
                raise ValueError("expected exactly one allowed IP entry")
            address = entries[0]
            parsed = ipaddress.IPv4Address(address)
            if (
                str(parsed) != address
                or parsed.is_unspecified
                or parsed.is_loopback
                or parsed.is_multicast
            ):
                raise ValueError("allowed IP is not a usable canonical IPv4 address")
        except (OSError, UnicodeError, ValueError) as exc:
            if not self._allowed_remote_file_error:
                print(
                    f"Warning: retaining prior allowed sender; cannot reload {path}: {exc}",
                    end="\r\n",
                    flush=True,
                )
                self._allowed_remote_file_error = True
            return

        self._allowed_remote_file_error = False
        with self._lock:
            previous = self._allowed_remote
            if address == previous:
                return
            self._allowed_remote = address
            self._state = UserInput()
            self._last_recv_s = 0.0
            self._has_operator_state = False
            self._hold_once = True
            self._walk_armed = False
            self._arm_armed = False
        print(
            f"NetworkInputSource allowed sender changed: {previous or 'any sender'} -> {address}; holding motor goals",
            end="\r\n",
            flush=True,
        )

    def _read_loop(self) -> None:
        sock = self._sock
        assert sock is not None
        next_allowlist_check_s = 0.0
        while self._running:
            if self._allowed_remote_file is not None and time.monotonic() >= next_allowlist_check_s:
                self._reload_allowed_remote()
                next_allowlist_check_s = time.monotonic() + 0.25
            try:
                data, addr = sock.recvfrom(16384)
            except (TimeoutError, OSError):
                continue
            if self._allowed_remote is not None and addr[0] != self._allowed_remote:
                continue
            try:
                packet = json.loads(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(packet, dict) and packet.get("type") == "clock_ping":
                self._reply_clock_ping(packet, addr)
                continue
            try:
                self._apply(packet)
            except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                # A bad tracking or motion field must not discard an otherwise
                # valid B/A/R3 command from the same authenticated snapshot.
                # Re-parse just the session, sequence and hardware gate with
                # optional motion fields omitted.  _apply() still validates the
                # sender's session/sequence and the three boolean gate fields.
                if not isinstance(packet, dict):
                    continue
                gate_packet = {
                    key: packet[key]
                    for key in (
                        "version", "session_id", "seq", "torque_enabled",
                        "torque_off_requested", "policy_enabled",
                    )
                    if key in packet
                }
                try:
                    self._apply(gate_packet, gate_only=True)
                except (KeyError, IndexError, TypeError, ValueError, OverflowError):
                    # An invalid gate/session is not an authenticated command.
                    continue

    def _reply_clock_ping(self, packet: dict, addr: tuple[str, int]) -> None:
        """Echo a bridge-side latency probe. Stateless: never touches UserInput.

        The bridge measures its own round-trip time from this echo alone (send
        T1, receive reply at T3, RTT = T3-T1); the robot does not need a
        synchronized clock and reports none. ``nonce`` is capped and
        type-checked so a malformed probe cannot wedge the receive thread or
        grow without bound.
        """
        nonce = packet.get("nonce")
        if not isinstance(nonce, str) or len(nonce) > 64:
            return
        reply_obj = {"type": "clock_pong", "nonce": nonce}
        with self._lock:
            telemetry = self._head_telemetry
        if telemetry is not None:
            reply_obj.update(telemetry)
        reply = json.dumps(
            reply_obj,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            assert self._sock is not None
            self._sock.sendto(reply, addr)
        except OSError:
            pass

    def _apply(self, packet: dict, *, gate_only: bool = False) -> None:
        if not isinstance(packet, dict):
            raise TypeError("packet must be a JSON object")
        if packet.get("version") != 1:
            raise ValueError("unsupported protocol version")

        session_id = packet.get("session_id")
        seq = packet.get("seq")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
            raise TypeError(
                "session_id must be a non-empty string of at most 128 characters"
            )
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
        if not isinstance(move_values, list) or not all(
            isinstance(v, str) for v in move_values
        ):
            raise TypeError("active_moves must be a list of strings")
        requested_moves = set(move_values)
        if not requested_moves <= _NETWORK_MOVES:
            raise ValueError("network packet requested an unsupported move")

        locomotion_policy = packet.get("locomotion_policy", "walk")
        if locomotion_policy not in _LOCOMOTION_POLICIES:
            raise ValueError("unsupported locomotion_policy")
        incoming_pico_packet = locomotion_policy == "pico_teleop"
        pico_walk_requested = incoming_pico_packet and "walk" in requested_moves

        orientation = packet.get("head_orientation")
        if orientation is not None:
            try:
                if not isinstance(orientation, dict):
                    raise TypeError("head_orientation must be an object or null")
                parsed_orientation = {
                    axis: _finite_float(orientation.get(axis, 0.0))
                    for axis in ("roll", "pitch", "yaw")
                }
            except (TypeError, ValueError, OverflowError):
                # Head tracking is optional; keep this snapshot's buttons,
                # joystick and other independently valid tracking channels.
                parsed_orientation = None
        else:
            parsed_orientation = None

        head_yaw_front = packet.get("head_yaw_front", False)
        if not isinstance(head_yaw_front, bool):
            head_yaw_front = False

        # Only an explicit B event requests torque OFF. A bridge restart or
        # missing tracking frame emits false/false, which holds the current
        # physical goal and torque state instead of collapsing the robot.
        torque_enabled = packet.get("torque_enabled", False)
        if not isinstance(torque_enabled, bool):
            raise TypeError("torque_enabled must be boolean")
        torque_off_requested = packet.get("torque_off_requested", False)
        if not isinstance(torque_off_requested, bool):
            raise TypeError("torque_off_requested must be boolean")
        policy_enabled = packet.get("policy_enabled", False)
        if not isinstance(policy_enabled, bool):
            raise TypeError("policy_enabled must be boolean")
        hold_last_targets = not torque_enabled and not torque_off_requested
        if torque_off_requested:
            torque_enabled = False
            policy_enabled = False
        elif hold_last_targets:
            torque_enabled = None
            policy_enabled = None
        else:
            policy_enabled = policy_enabled and torque_enabled

        # Defence in depth: the bridge also sends neutral motion fields while
        # gated, but the robot independently discards them before any deadman
        # or learned-policy parsing can retain a previous target.
        if not policy_enabled:
            requested_moves.clear()
            parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

        # Parse the direct-arm channel independently from the learned-policy
        # Cartesian targets.  A malformed arm snapshot is consumed as a
        # fail-closed release below instead of raising out of _apply(), because
        # ignoring the datagram would retain an older joint target until the
        # network watchdog expired.
        arm_fields_present = (
            "arm_tracking_enabled" in packet and "arm_joint_target" in packet
        )
        arm_payload_valid = True
        arm_tracking_enabled = packet.get("arm_tracking_enabled", False)
        parsed_arm_joint_target = None
        try:
            if not isinstance(arm_tracking_enabled, bool):
                raise TypeError("arm_tracking_enabled must be boolean")
            parsed_arm_joint_target = parse_pico_arm_joint_target(
                packet.get("arm_joint_target")
            )
            if not arm_tracking_enabled and any(
                parsed_arm_joint_target[side] != PICO_ARM_HOME_RAD[side]
                for side in PICO_ARM_SIDES
            ):
                raise ValueError(
                    "released arm tracking requires the exact authenticated HOME target"
                )
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            arm_payload_valid = False
            arm_tracking_enabled = False
            parsed_arm_joint_target = None

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

        pico_metadata_valid = packet.get(
            "body_target_contract"
        ) == _PICO_BODY_TARGET_CONTRACT and _matches_pico_safety_margin(
            packet.get("body_target_safety_margin")
        )
        pico_arm_requested = "pico_arms" in requested_moves
        pico_arm_snapshot_valid = (
            incoming_pico_packet
            and pico_metadata_valid
            and pico_arm_requested
            and arm_fields_present
            and arm_payload_valid
            and parsed_arm_joint_target is not None
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
                and _support_foot_floor_projection_valid(parsed_foot_target)
                and _simultaneous_both_feet_valid(
                    parsed_foot_target,
                    parsed_velocity,
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

        if not policy_enabled:
            # Keep a disabled-policy snapshot completely inert even if a
            # sender accidentally includes held-trigger/body data alongside
            # the hardware-state fields.
            requested_moves.clear()
            parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            parsed_foot_target = None
            parsed_hand_target = None
            arm_tracking_enabled = False
            parsed_arm_joint_target = None
            pico_arm_requested = False
            pico_arm_snapshot_valid = False

        # R3 keeps the standing actor active at zero velocity even while the
        # left trigger is released. PICO arm authentication remains independent.
        balance_only = bool(policy_enabled and "walk" not in requested_moves)
        if balance_only:
            requested_moves.add("walk")
            locomotion_policy = "walk"
            parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            parsed_foot_target = None
            parsed_hand_target = None

        with self._lock:
            if hold_last_targets and self._has_operator_state:
                # An empty bridge snapshot means its PICO input is temporarily
                # unavailable. Preserve the last authenticated operator state,
                # including policy feedback, across UDP and bridge restarts.
                # An explicit B packet still takes the normal path below.
                if session_id != self._session_id:
                    if session_id in self._retired_sessions:
                        return
                    if self._session_id is not None:
                        self._retired_sessions.append(self._session_id)
                        del self._retired_sessions[:-16]
                    self._session_id = session_id
                    self._last_seq = -1
                if seq <= self._last_seq:
                    return
                self._last_seq = seq
                self._last_recv_s = time.monotonic()
                return
            if session_id != self._session_id:
                if session_id in self._retired_sessions:
                    return
                # A bridge handover must begin from a released walk deadman. This
                # permits restart at any sequence number while rejecting held-trigger
                # takeover packets.  The independent right-trigger arm deadman
                # follows the same rule so a restarted bridge cannot jump to a
                # cached arm pose while the physical trigger is already held.
                if ("walk" in requested_moves and not balance_only) or arm_tracking_enabled:
                    return
                if self._session_id is not None:
                    self._retired_sessions.append(self._session_id)
                    del self._retired_sessions[:-16]
                self._session_id = session_id
                self._last_seq = -1
                self._walk_armed = False
                self._arm_armed = False
            if seq <= self._last_seq:
                return
            self._last_seq = seq

            learned_policy_degraded = bool(
                policy_enabled and pico_walk_requested and not pico_body_target_valid
            )
            if learned_policy_degraded:
                # Body tracking is an enhancement, not the locomotion deadman.
                # Replace the learned-policy request with the proven velocity actor
                # while retaining this packet's trigger and joystick values. The
                # selector latches that fallback until a trigger release so recovery
                # cannot hot-swap policies mid-stride.
                locomotion_policy = "walk"
                parsed_foot_target = None
                parsed_hand_target = None

            if self._motion_inhibited:
                self._walk_armed = False
                self._arm_armed = False
                requested_moves.discard("walk")
                requested_moves.discard("pico_arms")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
                arm_tracking_enabled = False
                parsed_arm_joint_target = None
            elif balance_only:
                # A complete R3 snapshot proves the left trigger is released.
                # A gate-only retry has no trigger data, so keep the balance
                # actor active at zero velocity without arming later walking.
                self._walk_armed = not gate_only
            elif "walk" not in requested_moves:
                self._walk_armed = True
            elif not self._walk_armed:
                requested_moves.discard("walk")
                parsed_velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

            if self._motion_inhibited:
                pass
            elif not pico_arm_requested:
                # Legacy/non-PICO snapshots do not own the overlay and cannot
                # implicitly arm a later right-trigger press.
                self._arm_armed = False
                arm_tracking_enabled = False
                parsed_arm_joint_target = None
            elif not pico_arm_snapshot_valid:
                # Clear the complete arm channel atomically.  In particular,
                # never keep one valid side or a previous target after a bad
                # packet.
                self._arm_armed = False
                requested_moves.discard("pico_arms")
                arm_tracking_enabled = False
                parsed_arm_joint_target = None
            elif not arm_tracking_enabled:
                # A valid released snapshot keeps the overlay active at the
                # exact PICO home and arms the next physical press.
                self._arm_armed = True
            elif not self._arm_armed:
                requested_moves.discard("pico_arms")
                arm_tracking_enabled = False
                parsed_arm_joint_target = None

            self._state = UserInput(
                active_moves=requested_moves,
                velocity=parsed_velocity,
                locomotion_policy=locomotion_policy,
                learned_policy_degraded=learned_policy_degraded,
                balance_only=balance_only,
                # Head tracking shares the arm/hand deadman: both only take effect
                # while the right trigger is held (arm_tracking_enabled reflects its
                # latched, re-arm-safe state at this point).
                head_orientation=parsed_orientation if arm_tracking_enabled else None,
                head_yaw_front=head_yaw_front,
                foot_target=parsed_foot_target,
                hand_target=parsed_hand_target,
                arm_tracking_enabled=arm_tracking_enabled,
                arm_joint_target=parsed_arm_joint_target,
                torque_enabled=torque_enabled,
                policy_enabled=policy_enabled,
                hold_last_targets=hold_last_targets,
                getup_armed=False,
            )
            self._last_recv_s = time.monotonic()
            if not hold_last_targets:
                self._has_operator_state = True
