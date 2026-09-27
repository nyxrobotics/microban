"""Small, nonblocking bridge from the robot-local face detector to GC300 input.

The detector runs in a separate low-priority process.  This class only reads
its latest tiny /run snapshot and converts a fresh face location to bounded
walking and head targets.  Missing detection affects this mode alone.
"""

from __future__ import annotations

from collections import deque
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
# Face width measures depth along the camera ray, not ground distance.  Use the
# measured focal length and a deliberately small face-size assumption, then
# project the sight line onto the ground before allowing forward motion.
_FX_PER_IMAGE_WIDTH = 233.8976224959923 / 640.0
_ASSUMED_FACE_WIDTH_M = 0.12
_STOP_HORIZONTAL_DISTANCE_M = 1.1
_MAX_APPROACH_ELEVATION_RAD = math.radians(55.0)
_POSE_MATCH_MAX_AGE_NS = 120_000_000
_LOST_FACE_HEAD_HOLD_NS = 1_000_000_000


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


@dataclass(frozen=True)
class HeadPose:
    measured_monotonic_ns: int
    head: float
    neck_pitch: float
    trunk_pitch: float


class PersonFollower:
    def __init__(self, path: Path = DEFAULT_TARGET_PATH) -> None:
        self._path = path
        self._file_identity: tuple[int, int, int] | None = None
        self._latest: FaceTarget | None = None
        self._last_applied_capture_ns: int | None = None
        self._head_yaw = 0.0
        self._head_pitch = _FACE_SEARCH_PITCH_RAD
        self._face_elevation_rad = 0.0
        self._horizontal_distance_m = 0.0
        self._measured_head_yaw = 0.0
        self._measured_neck_pitch = 0.0
        self._trunk_pitch = 0.0
        self._pose_history: deque[HeadPose] = deque(maxlen=128)
        self._last_face_seen_ns: int | None = None
        self._last_logged_visible: bool | None = None

    def reset(self) -> None:
        self._last_applied_capture_ns = None
        self._head_yaw = 0.0
        self._head_pitch = _FACE_SEARCH_PITCH_RAD
        self._face_elevation_rad = 0.0
        self._horizontal_distance_m = 0.0
        self._last_face_seen_ns = None
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
        self._pose_history.append(
            HeadPose(
                time.monotonic_ns(),
                self._measured_head_yaw,
                self._measured_neck_pitch,
                self._trunk_pitch,
            )
        )

    def _pose_at_capture(self, captured_ns: int) -> HeadPose | None:
        if not self._pose_history:
            return None
        closest = min(
            self._pose_history,
            key=lambda pose: abs(pose.measured_monotonic_ns - captured_ns),
        )
        if abs(closest.measured_monotonic_ns - captured_ns) > _POSE_MATCH_MAX_AGE_NS:
            return None
        return closest

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
            if (
                not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
                or values[2] < 0.01
            ):
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
        capture_pose = None
        if visible and target is not None and target.captured_monotonic_ns != self._last_applied_capture_ns:
            capture_pose = self._pose_at_capture(target.captured_monotonic_ns)
            visible = capture_pose is not None
        if visible != self._last_logged_visible:
            print("GC300 follow: face visible" if visible else "GC300 follow: waiting for face", flush=True)
            self._last_logged_visible = visible
        if not visible:
            self._last_applied_capture_ns = None
            if (
                self._last_face_seen_ns is not None
                and now_ns - self._last_face_seen_ns <= _LOST_FACE_HEAD_HOLD_NS
            ):
                return FollowCommand(
                    velocity={"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                    head_orientation={
                        "roll": 0.0,
                        "pitch": self._head_pitch,
                        "yaw": self._head_yaw,
                    },
                    target_visible=False,
                )
            self._head_yaw = 0.0
            self._head_pitch = _FACE_SEARCH_PITCH_RAD
            self._face_elevation_rad = 0.0
            self._horizontal_distance_m = 0.0
            return FollowCommand(
                velocity={"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                head_orientation={"roll": 0.0, "pitch": self._head_pitch, "yaw": 0.0},
                target_visible=False,
            )

        assert target is not None
        if target.captured_monotonic_ns != self._last_applied_capture_ns:
            assert capture_pose is not None
            # Positive robot yaw points left.  The image x axis points right.
            pixel_yaw = -math.atan((2.0 * target.center_x - 1.0) * math.tan(_HORIZONTAL_HALF_FOV_RAD))
            pixel_pitch = math.atan((2.0 * target.center_y - 1.0) * math.tan(_VERTICAL_HALF_FOV_RAD))
            self._head_yaw = _clamp(capture_pose.head + pixel_yaw, -1.2, 1.2)
            face_pitch = capture_pose.trunk_pitch + capture_pose.neck_pitch + pixel_pitch
            self._head_pitch = _clamp(
                face_pitch,
                -1.2, 0.35,
            )
            optical_depth_m = (
                _FX_PER_IMAGE_WIDTH * _ASSUMED_FACE_WIDTH_M / target.face_width
            )
            ray_scale = math.sqrt(
                1.0 + math.tan(pixel_yaw) ** 2 + math.tan(pixel_pitch) ** 2
            )
            self._face_elevation_rad = -face_pitch
            self._horizontal_distance_m = max(
                0.0, optical_depth_m * ray_scale * math.cos(face_pitch)
            )
            self._last_applied_capture_ns = target.captured_monotonic_ns
        self._last_face_seen_ns = now_ns

        bearing = self._head_yaw
        turn = _clamp(0.4 * bearing, -0.22, 0.22) if abs(bearing) > 0.06 else 0.0
        # Face width alone underestimates how close the feet are when the low
        # camera looks steeply up.  The projected range is approximate because
        # real face sizes vary, so keep a margin and stop on steep elevation.
        forward = _clamp(
            (self._horizontal_distance_m - _STOP_HORIZONTAL_DISTANCE_M) * 0.24,
            0.0, 0.16,
        )
        if self._face_elevation_rad >= _MAX_APPROACH_ELEVATION_RAD:
            forward = 0.0
        forward *= _clamp(1.0 - abs(bearing) / 0.30, 0.0, 1.0)
        return FollowCommand(
            velocity={"vx": forward, "vy": 0.0, "vtheta": turn},
            head_orientation={"roll": 0.0, "pitch": self._head_pitch, "yaw": self._head_yaw},
            target_visible=True,
        )
