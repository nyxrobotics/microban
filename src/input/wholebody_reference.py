# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""The whole-body reference stream: retargeted frames from the PC over UDP.

The PC retargets the operator's body (PICO) to Microban and sends one frame
per datagram at 50 Hz to UDP 5556, the format of mjlab_microban
docs/wholebody_reference_stream.md (``src/mjlab_microban/live/stream.py``):
a little-endian header followed by float32 arrays in the order of
``wholebody/data/hdf5.py`` FRAME_KEYS::

    magic        4s   b"MBWB"
    version      u16  1
    joint_count  u16  21 (head, neck_roll, neck_pitch, then OBSERVATION_DOF_ORDER)
    body_count   u16  22 (robot.xml's bodies; body 0 is the trunk, the anchor)
    seq          u32  +1 per frame from 0 at the PC's start, modulo 2^32
    sent_time_s  f64  the PC's Unix time
    flags        u16  CALIBRATED, ACTIVE, RESTART, STALE, HELD, LATE (bits 0-5)
    joint_pos       float32[21]     rad, absolute
    joint_vel       float32[21]     rad/s
    body_pos_w      float32[22, 3]  m, world (z up, floor at 0)
    body_quat_w     float32[22, 4]  w, x, y, z
    body_lin_vel_w  float32[22, 3]  m/s, world
    body_ang_vel_w  float32[22, 3]  rad/s, world

The receiver keeps the newest frame with its arrival time, the count of
frames lost (sequence gaps) since it started, the frame's latency above the
lowest latency of the last LATENCY_WINDOW_S (the clocks need not agree) and an
activation count: +1 whenever the PC starts a reference (CALIBRATED and ACTIVE
set after a frame without them, or a RESTART frame).  A frame older by
sequence number is dropped unless it was sent more than RESTART_TIME_S later
than the newest one (the PC restarted).  Freshness itself is judged by the
whole-body move (moves/wholebody.py).
"""

from __future__ import annotations

import collections
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"MBWB"
VERSION = 1
HEADER = struct.Struct("<4sHHHIdH")
NUM_JOINTS = 21
NUM_BODIES = 22
CALIBRATED = 1 << 0
ACTIVE = 1 << 1
RESTART = 1 << 2
STALE = 1 << 3
HELD = 1 << 4
LATE = 1 << 5
RESTART_TIME_S = 0.5
# Joint and body arrays as FRAME_KEYS: (name, per-body width or None for joints).
FRAME_LAYOUT = (
    ("joint_pos", None),
    ("joint_vel", None),
    ("body_pos_w", 3),
    ("body_quat_w", 4),
    ("body_lin_vel_w", 3),
    ("body_ang_vel_w", 3),
)
QUAT_NORM_TOLERANCE = 0.05
MAX_ABS_JOINT_POS_RAD = 2.0 * math.pi
LATENCY_WINDOW_S = 10.0


class ReferenceDecodeError(ValueError):
    """A datagram is not a whole-body reference frame."""


@dataclass(frozen=True)
class ReferenceFrame:
    """One reference frame; the anchor (trunk) arrays are body 0's."""

    seq: int
    time_s: float
    flags: int
    joint_pos: np.ndarray  # (21,) robot joint order
    joint_vel: np.ndarray  # (21,)
    anchor_pos: np.ndarray  # (3,)
    anchor_quat: np.ndarray  # (4,) w, x, y, z, unit
    anchor_lin_vel: np.ndarray  # (3,)
    anchor_ang_vel: np.ndarray  # (3,)

    @property
    def engaged(self) -> bool:
        """The PC follows a calibrated operator who started tracking."""
        return self.flags & (CALIBRATED | ACTIVE) == CALIBRATED | ACTIVE


def frame_size(num_joints: int, num_bodies: int) -> int:
    floats = sum(num_joints if width is None else num_bodies * width for _, width in FRAME_LAYOUT)
    return HEADER.size + 4 * floats


def encode_frame(seq: int, time_s: float, arrays: dict[str, np.ndarray], flags: int = 0) -> bytes:
    """One datagram from FRAME_KEYS arrays of one frame (the sender's and the tests' side)."""

    joints = np.asarray(arrays["joint_pos"], dtype=np.float32).reshape(-1)
    bodies = np.asarray(arrays["body_pos_w"], dtype=np.float32).reshape(-1, 3).shape[0]
    payload = []
    for name, width in FRAME_LAYOUT:
        value = np.asarray(arrays[name], dtype=np.float32).reshape(-1)
        expected = len(joints) if width is None else bodies * width
        if value.size != expected:
            raise ValueError(f"{name} has {value.size} values, expected {expected}")
        payload.append(value)
    header = HEADER.pack(MAGIC, VERSION, len(joints), bodies, int(seq) & 0xFFFFFFFF, float(time_s), int(flags))
    return header + np.concatenate(payload).astype("<f4").tobytes()


def decode_frame(data: bytes) -> ReferenceFrame:
    if len(data) < HEADER.size:
        raise ReferenceDecodeError("datagram shorter than the header")
    magic, version, joints, bodies, seq, time_s, flags = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ReferenceDecodeError("bad magic")
    if version != VERSION:
        raise ReferenceDecodeError(f"unsupported version {version}")
    if joints != NUM_JOINTS or bodies != NUM_BODIES:
        raise ReferenceDecodeError(f"{joints} joints / {bodies} bodies")
    if len(data) != frame_size(joints, bodies):
        raise ReferenceDecodeError(f"datagram is {len(data)} bytes, expected {frame_size(joints, bodies)}")
    if not math.isfinite(time_s):
        raise ReferenceDecodeError("non-finite time")
    values = np.frombuffer(data, dtype="<f4", offset=HEADER.size).astype(np.float64)
    if not np.isfinite(values).all():
        raise ReferenceDecodeError("non-finite value")
    arrays, start = {}, 0
    for name, width in FRAME_LAYOUT:
        count = joints if width is None else bodies * width
        arrays[name] = values[start : start + count] if width is None else values[start : start + count].reshape(
            bodies, width
        )
        start += count
    if np.any(np.abs(arrays["joint_pos"]) > MAX_ABS_JOINT_POS_RAD):
        raise ReferenceDecodeError("joint angle out of range")
    quat = arrays["body_quat_w"][0]
    norm = float(np.linalg.norm(quat))
    if abs(norm - 1.0) > QUAT_NORM_TOLERANCE:
        raise ReferenceDecodeError("trunk quaternion is not a unit quaternion")
    return ReferenceFrame(
        seq=int(seq),
        time_s=float(time_s),
        flags=int(flags),
        joint_pos=arrays["joint_pos"].copy(),
        joint_vel=arrays["joint_vel"].copy(),
        anchor_pos=arrays["body_pos_w"][0].copy(),
        anchor_quat=quat / norm,
        anchor_lin_vel=arrays["body_lin_vel_w"][0].copy(),
        anchor_ang_vel=arrays["body_ang_vel_w"][0].copy(),
    )


@dataclass(frozen=True)
class ReferenceSnapshot:
    """What the controller reads once per tick."""

    frame: ReferenceFrame | None
    received_s: float  # time.perf_counter() at arrival of ``frame``
    stream: int  # +1 whenever the PC restarted (its sequence began again)
    lost: int  # frames missing between consecutive frames, counted since the start
    latency_excess_s: float  # this frame's latency above the window's lowest
    activation: int  # +1 whenever the PC starts a reference
    stale_flag_s: float  # how long (PC time) the STALE flag has been set, 0 when clear
    held_flag_s: float  # how long (PC time) the HELD flag has been set, 0 when clear


class ReferenceReceiver:
    """Receives reference frames on a UDP port (a thread) and keeps the newest."""

    def __init__(
        self,
        port: int = 5556,
        allowed_remote: str | None = None,
        allowed_remote_file: str | None = None,
        clock=time.perf_counter,
    ) -> None:
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        self._port = port
        self._allowed_remote = socket.gethostbyname(allowed_remote) if allowed_remote else None
        self._allowed_remote_file = Path(allowed_remote_file) if allowed_remote_file else None
        self._allowed_checked_s = -math.inf
        self._clock = clock
        self._lock = threading.Lock()
        self._frame: ReferenceFrame | None = None
        self._received_s = -math.inf
        self._stream = 0
        self._lost = 0
        self._latency_excess_s = 0.0
        self._activation = 0
        self._flag_since: dict[int, float] = {}
        self._offsets: collections.deque[tuple[float, float]] = collections.deque()
        self.rejected = 0
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # Packet handling (also called directly by tests and the simulation).
    def accept(self, data: bytes, received_s: float | None = None) -> bool:
        try:
            frame = decode_frame(data)
        except ReferenceDecodeError:
            self.rejected += 1
            return False
        now = self._clock() if received_s is None else received_s
        with self._lock:
            previous = self._frame
            if previous is not None:
                step = (frame.seq - previous.seq) % (1 << 32)
                if step == 0 or step >= 1 << 31:
                    if frame.time_s <= previous.time_s + RESTART_TIME_S:
                        # A late duplicate or reordered datagram.
                        return False
                    # Sent well after the newest frame: the PC started again.
                    self._stream += 1
                    self._offsets.clear()
                else:
                    self._lost += step - 1
            if frame.engaged and (frame.flags & RESTART or previous is None or not previous.engaged):
                self._activation += 1
            for flag in (STALE, HELD):
                if not frame.flags & flag:
                    self._flag_since.pop(flag, None)
                else:
                    self._flag_since.setdefault(flag, frame.time_s)
            offset = now - frame.time_s
            while self._offsets and self._offsets[0][0] < now - LATENCY_WINDOW_S:
                self._offsets.popleft()
            while self._offsets and self._offsets[-1][1] >= offset:
                self._offsets.pop()
            self._offsets.append((now, offset))
            self._latency_excess_s = offset - self._offsets[0][1]
            self._frame = frame
            self._received_s = now
        return True

    def read(self) -> ReferenceSnapshot:
        with self._lock:
            snapshot = ReferenceSnapshot(
                frame=self._frame,
                received_s=self._received_s,
                stream=self._stream,
                lost=self._lost,
                latency_excess_s=self._latency_excess_s,
                activation=self._activation,
                stale_flag_s=self._flag_duration(STALE),
                held_flag_s=self._flag_duration(HELD),
            )
        return snapshot

    def _flag_duration(self, flag: int) -> float:
        since = self._flag_since.get(flag)
        return 0.0 if since is None or self._frame is None else self._frame.time_s - since

    def now(self) -> float:
        return self._clock()

    # Socket thread.
    def start(self) -> None:
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", self._port))
        self._sock.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, name="wholebody-reference", daemon=True)
        self._thread.start()
        print(
            f"Whole-body reference listening on UDP :{self._port} ({self._allowed_remote or 'any sender'})",
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

    def _sender_allowed(self, host: str) -> bool:
        if self._allowed_remote_file is not None:
            now = time.monotonic()
            if now - self._allowed_checked_s >= 1.0:
                self._allowed_checked_s = now
                try:
                    text = self._allowed_remote_file.read_text(encoding="ascii").strip()
                    self._allowed_remote = socket.gethostbyname(text) if text else None
                except (OSError, UnicodeError, socket.gaierror):
                    self._allowed_remote = None
                    return False
            if self._allowed_remote is None:
                return False
        return self._allowed_remote is None or host == self._allowed_remote

    def _read_loop(self) -> None:
        while self._running:
            sock = self._sock
            if sock is None:
                return
            try:
                data, (host, _port) = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if self._sender_allowed(host):
                self.accept(data)
