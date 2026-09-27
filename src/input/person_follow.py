"""Small, nonblocking bridge from the robot-local face detector to GC300 input.

The detector runs in a separate low-priority process.  This class only reads
its latest tiny /run snapshot and converts a fresh face location to bounded
walking and head targets.  Missing detection affects this mode alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time


DEFAULT_TARGET_PATH = Path("/run/microban-person-tracking/target.json")
TARGET_MAX_AGE_NS = 1_500_000_000
_MAX_TARGET_BYTES = 512
_HORIZONTAL_HALF_FOV_RAD = math.radians(56.0)
_VERTICAL_HALF_FOV_RAD = math.atan(math.tan(_HORIZONTAL_HALF_FOV_RAD) * 3.0 / 4.0)
# The camera sits about 32 cm above the floor.  A level view cannot see a
# nearby adult's face; an upward aim of 40 degrees covers useful person heights
# throughout the near field without moving the head between detector updates.
_FACE_SEARCH_PITCH_RAD = math.radians(-40.0)
_STOP_FACE_WIDTH = 0.055


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class FaceTarget:
    captured_monotonic_ns: int
    center_x: float
    center_y: float
    face_width: float


@dataclass(frozen=True)
class FollowCommand:
    velocity: dict[str, float]
    head_orientation: dict[str, float]
    target_visible: bool


class PersonFollower:
    def __init__(self, path: Path = DEFAULT_TARGET_PATH) -> None:
        self._path = path
        self._file_identity: tuple[int, int, int] | None = None
        self._latest: FaceTarget | None = None
        self._last_applied_capture_ns: int | None = None
        self._head_yaw = 0.0
        self._head_pitch = _FACE_SEARCH_PITCH_RAD
        self._measured_head_yaw = 0.0
        self._measured_neck_pitch = 0.0
        self._trunk_pitch = 0.0
        self._last_logged_visible: bool | None = None

    def reset(self) -> None:
        self._last_applied_capture_ns = None
        self._head_yaw = 0.0
        self._head_pitch = _FACE_SEARCH_PITCH_RAD
        self._last_logged_visible = None

    def set_head_telemetry(
        self, *, head: float, neck_pitch: float, trunk_pitch: float
    ) -> None:
        if math.isfinite(head):
            self._measured_head_yaw = head
        if math.isfinite(neck_pitch):
            self._measured_neck_pitch = neck_pitch
        if math.isfinite(trunk_pitch):
            self._trunk_pitch = trunk_pitch

    def _load_latest(self) -> None:
        try:
            status = self._path.stat()
            identity = (status.st_ino, status.st_mtime_ns, status.st_size)
            if identity == self._file_identity:
                return
            self._file_identity = identity
            if status.st_size > _MAX_TARGET_BYTES:
                self._latest = None
                return
            with self._path.open("rb") as stream:
                raw = stream.read(_MAX_TARGET_BYTES + 1)
            if len(raw) > _MAX_TARGET_BYTES:
                raise ValueError("target snapshot is too large")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("target snapshot is not an object")
            if payload.get("version") != 1 or payload.get("detected") is not True:
                self._latest = None
                return
            timestamp = payload["captured_monotonic_ns"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                raise ValueError("invalid capture timestamp")
            values = [float(payload[key]) for key in ("center_x", "center_y", "face_width")]
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                raise ValueError("invalid face location")
            self._latest = FaceTarget(timestamp, *values)
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            self._latest = None
            self._file_identity = None

    def command(self) -> FollowCommand:
        self._load_latest()
        target = self._latest
        now_ns = time.monotonic_ns()
        visible = (
            target is not None
            and 0 <= now_ns - target.captured_monotonic_ns <= TARGET_MAX_AGE_NS
        )
        if visible != self._last_logged_visible:
            print("GC300 follow: face visible" if visible else "GC300 follow: waiting for face", flush=True)
            self._last_logged_visible = visible
        if not visible:
            self._last_applied_capture_ns = None
            self._head_yaw = 0.0
            self._head_pitch = _FACE_SEARCH_PITCH_RAD
            return FollowCommand(
                velocity={"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                head_orientation={"roll": 0.0, "pitch": self._head_pitch, "yaw": 0.0},
                target_visible=False,
            )

        assert target is not None
        if target.captured_monotonic_ns != self._last_applied_capture_ns:
            # Positive robot yaw points left.  The image x axis points right.
            pixel_yaw = -math.atan((2.0 * target.center_x - 1.0) * math.tan(_HORIZONTAL_HALF_FOV_RAD))
            pixel_pitch = math.atan((2.0 * target.center_y - 1.0) * math.tan(_VERTICAL_HALF_FOV_RAD))
            self._head_yaw = _clamp(self._measured_head_yaw + pixel_yaw, -1.2, 1.2)
            self._head_pitch = _clamp(
                self._trunk_pitch + self._measured_neck_pitch + pixel_pitch,
                -1.2, 0.35,
            )
            self._last_applied_capture_ns = target.captured_monotonic_ns

        bearing = self._head_yaw
        turn = _clamp(0.4 * bearing, -0.22, 0.22) if abs(bearing) > 0.06 else 0.0
        # Face width is only a proximity proxy, not calibrated distance.  A
        # 15 cm face reaches about 0.055 of the 640 px left eye at 1 m with
        # the measured fx=234 px.  Stop there, while turning before advancing.
        size_error = _STOP_FACE_WIDTH - target.face_width
        forward = _clamp(size_error * 5.0, 0.0, 0.16)
        forward *= _clamp(1.0 - abs(bearing) / 0.30, 0.0, 1.0)
        return FollowCommand(
            velocity={"vx": forward, "vy": 0.0, "vtheta": turn},
            head_orientation={"roll": 0.0, "pitch": self._head_pitch, "yaw": self._head_yaw},
            target_visible=True,
        )
