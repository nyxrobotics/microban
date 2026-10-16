# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Whole-body policy: it stands the robot on its own reference and follows the operator's retargeted body.

The policy (contract microban-wholebody-1, wholebody_contract.py) drives the
18 body joints, arms included: target = clip(HOME + raw, -pi, +pi).  Its
observation is built exactly as in training (mjlab_microban
``wholebody/observations.py`` and ``env_cfg.actor_terms``): 118 values per
control tick, the last 10 ticks oldest first, every row the first frame after
a reset.  The ``*_b`` terms use the IMU trunk orientation with its yaw.

From R3 on, the policy drives the robot the whole time; the PICO policy does
not stand it.  The reference is the move's own until the PC's start joins:

1. enter: from the measured 21 joints and the IMU's attitude (velocities
   zero, HOME's height) to the HOME stance in ENTER_S (smoothstep), its
   heading the IMU's at the start;
2. stand: the HOME stance (HOME's 21 joints, trunk attitude and height,
   velocities zero), its heading following the IMU's at most
   HEADING_STEP_RAD per tick (not while the trunk is tilted more than
   HEADING_FREEZE_TILT_DEG from HOME's attitude); its angular velocity stays
   zero, so a robot turned by a push is not told the reference turns too;
3. join: the PC's reference, turned about z to the stand's heading
   (``reference_yaw: aligned_to_imu_yaw_at_start``), replaces the stand's in
   JOIN_S (smootherstep; the trunk's orientation offset fades as a rotation
   vector, never a slerp between two moving attitudes);
4. track: the PC's reference;
5. coast: on a stop, the last reference followed goes on at its speed, which
   fades to zero in COAST_S (the PC's Carry; its height does not rise);
6. hold: that pose, still, until STOP_HOLD_S after the stop;
7. return: to the HOME stance in RETURN_S (smoothstep, the heading fixed at
   its start), then stand again.

The join starts when, for STILL_TICKS ticks, the robot stands still (trunk
within STILL_TILT_DEG of HOME's attitude, every leg joint slower than
STILL_LEG_SPEED_RAD_S) and the PC's reference stands quietly (as every piece
of the training's benchmark starts: its trunk near upright, at standing
height, slow, not turning, and not the PC's interruption reference), and the
PC's start (activation) is new and live (CALIBRATED and ACTIVE, fresh, no
STALE, HELD or LATE flag) with ``walk`` not asked for on the wire.  A start
joined once is never joined again, whatever ends it.

Stop (track or join): the PC ends tracking (ACTIVE or CALIBRATED off), a
stale reference (no frame for MAX_REFERENCE_AGE_S, a jump of more than
MAX_SEQUENCE_GAP frames, a frame later than MAX_LATENCY_EXCESS_S, a restarted
PC, the PC's STALE flag for more than MAX_STALE_FLAG_S or its HELD flag for
more than MAX_HELD_FLAG_S), a new start from the PC, or ``walk`` on the wire:
coast, hold, return, stand.

A trunk off the reference's attitude (gravity's z in the two trunk frames
differs by more than ORIENTATION_TERMINATION, the training's ``anchor_ori``
end) while joining, tracking, coasting, holding or returning enters again
from the measured pose, as the training starts its next episode on the
reference: history and previous action rebuilt.  The history is also rebuilt
by every start of the move (R3, after a get-up) and after the scheduler's
safety hold.

A trunk tilted more than 60 deg from vertical holds the measured pose: the
scheduler's fall detection then starts the get-up.  The exception is a trunk
following the reference's while joining or tracking, or after a stop in
mid-bow: while that reference is itself past the fall angle, or was within
FALL_LAG_S (the robot follows late), and the robot keeps to its attitude, the
policy keeps running and the scheduler does not count a fall
(``follows_fallen_reference``).  A stop on a reference lying on the floor
(its trunk lower than FLOOR_REF_HEIGHT_M) counts as a fall: the get-up takes
the robot.

Neck: while joining and tracking its targets are the reference's neck angles
(the PC fills them from the operator's head), clamped to the contract's
``neck_limits_rad_json`` and slewed at ``neck_slew_rates_rad_s_json``, as
training's ReferenceNeckTarget does, and the HMD head move yields
(``WholebodyClaim.neck``).  Otherwise the move slews the neck toward its own
reference's (HOME standing), and the HMD head move, when it runs, drives it
as it does on the PICO policy.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from constants import (
    HOME_PROJECTED_GRAVITY,
    HOME_ROOT_POS_Z_M,
    HOME_ROOT_QUAT_WXYZ,
    KP_HARDWARE_NEUTRAL,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    SERVO_TARGET_RANGE_RAD,
)
from controller import ControllerProtocol
from input.wholebody_reference import HELD, LATE, STALE, ReferenceFrame, ReferenceReceiver
from moves.move import MotorCommand, Move, MoveState
from observer import Observation
from policy_contract import CONTROL_HZ, HEAD_JOINTS, SOFT_JOINT_POS_LIMITS
from wholebody_contract import (
    ACTION_WIDTH,
    HISTORY_LENGTH,
    INPUT_NAME,
    OBSERVATION_JOINT_NAMES,
    OBSERVATION_WIDTH,
    POLICY_FILE,
    WholebodyContract,
    load_wholebody_policy,
    parse_wholebody_policy,
)

ROBOT_JOINT_ORDER = OBSERVATION_JOINT_NAMES  # the wire's and the observation's joint order
ACTION_COLUMNS = [ROBOT_JOINT_ORDER.index(name) for name in OBSERVATION_DOF_ORDER]
NECK_COLUMNS = [ROBOT_JOINT_ORDER.index(name) for name in HEAD_JOINTS]
LEG_JOINTS = tuple(name for name in OBSERVATION_DOF_ORDER if any(p in name for p in ("hip", "knee", "ankle")))
ALL_MOTOR_IDS = list(MOTOR_TO_ID.values())
GRAVITY_W = np.array([0.0, 0.0, -1.0])
DT = 1.0 / CONTROL_HZ


def _mjcf_range(name: str) -> tuple[float, float]:
    """A joint's MJCF range: the soft limits divided back by 0.9 (wholebody_contract.frame_problem)."""
    if name not in SOFT_JOINT_POS_LIMITS:
        return -math.inf, math.inf
    lower, upper = SOFT_JOINT_POS_LIMITS[name]
    middle, half = 0.5 * (lower + upper), 0.5 * (upper - lower) / 0.9
    return middle - half, middle + half


JOINT_LOWER = np.array([_mjcf_range(name)[0] for name in ROBOT_JOINT_ORDER])
JOINT_UPPER = np.array([_mjcf_range(name)[1] for name in ROBOT_JOINT_ORDER])

ENTER_S = 1.0
JOIN_S = 1.0
COAST_S = 0.2
STOP_HOLD_S = 0.5  # from the stop, the coast included
RETURN_S = 1.5
# The stand's heading follows the IMU's (1 rad/s), not while the trunk is tilted.
HEADING_STEP_RAD = 0.02
HEADING_FREEZE_TILT_DEG = 30.0
# The robot stands still: the trunk near HOME's attitude, no leg joint moving
# (the PICO policy's start of preflight-fixes: its legs move at 1.7-4.6 rad/s
# stepping and below 0.4 rad/s standing).
STILL_TILT_DEG = 12.0
STILL_LEG_SPEED_RAD_S = 1.0
STILL_TICKS = 25
# The reference stands quietly, as every piece of the training's benchmark
# starts (mjlab_microban wholebody/benchmark.py standing_frames, its trunk
# checks: near upright, at standing height, slow; the soles are not in the
# stream's trunk), and does not turn (a turning operator is not joined: the
# join would carry the turn's offset).
REFERENCE_STAND_TILT_DEG = 30.0
REFERENCE_STAND_DROP_M = 0.02
REFERENCE_STAND_SPEED_M_S = 0.05
REFERENCE_STAND_TURN_RAD_S = 0.3
MAX_REFERENCE_AGE_S = 0.1
MAX_SEQUENCE_GAP = 5
MAX_LATENCY_EXCESS_S = 0.15
MAX_STALE_FLAG_S = 0.25  # the PC sets STALE after 0.25 s without the operator's body
# HELD is followed for up to 1 s: through it the PC sends its interruption
# reference and comes back to the operator with no frame changing more than
# the training reference's 99.9th percentile per tick (mjlab_microban
# docs/wholebody_reference_stream.md, stream version 2; a version-1 PC jumps
# from the held pose to the live one in one frame when HELD clears, and the
# receiver refuses its frames).
MAX_HELD_FLAG_S = 1.0
ORIENTATION_TERMINATION = 1.0
FALL_GRAVITY_Z = -0.5
# A trunk past the fall angle is not a fall while the reference's trunk is
# past it too, or was within FALL_LAG_S (the robot follows late), and the robot
# keeps to its attitude (ORIENTATION_TERMINATION).
FALL_LAG_S = 0.5
# A reference whose trunk is lower than this lies on the floor (PICO
# recordings: 0.05-0.09 m on the floor, 0.10-0.18 m bowing): a stop there
# leaves the robot to the get-up.
FLOOR_REF_HEIGHT_M = 0.10


class WholebodyPolicyRuntimeError(RuntimeError):
    """Inference produced a result that is unsafe to send to the servos."""


class Phase(enum.Enum):
    ENTER = "enter"
    STAND = "stand"
    JOIN = "join"
    TRACK = "track"
    COAST = "coast"
    HOLD = "hold"
    RETURN = "return"


FOLLOWING = (Phase.JOIN, Phase.TRACK)
STOPPING = (Phase.COAST, Phase.HOLD, Phase.RETURN)


# ------------------------------------------------------------------ quaternions (w, x, y, z)
def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: Sequence[float]) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_apply(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    u = np.asarray(q[1:], dtype=np.float64)
    t = 2.0 * np.cross(u, v)
    return np.asarray(v, dtype=np.float64) + q[0] * t + np.cross(u, t)


def quat_apply_inverse(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    return quat_apply(quat_conj(q), v)


def matrix_from_quat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of(q: Sequence[float]) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def yaw_alignment(robot_quat: Sequence[float], reference_quat: Sequence[float]) -> np.ndarray:
    """The turn about z that gives the reference trunk the robot trunk's heading."""
    return quat_from_yaw(yaw_of(robot_quat) - yaw_of(reference_quat))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    omega = math.acos(min(1.0, dot))
    out = (math.sin((1.0 - t) * omega) * q0 + math.sin(t * omega) * q1) / math.sin(omega)
    return out / np.linalg.norm(out)


# ------------------------------------------------------------------ reference in the robot's world
class Reference:
    """A reference state in the IMU's world (yaw-aligned), robot joint order."""

    __slots__ = ("joint_pos", "joint_vel", "quat", "lin_vel", "ang_vel", "height")

    def __init__(self, joint_pos, joint_vel, quat, lin_vel, ang_vel, height) -> None:
        self.joint_pos = np.asarray(joint_pos, dtype=np.float64)
        self.joint_vel = np.asarray(joint_vel, dtype=np.float64)
        self.quat = np.asarray(quat, dtype=np.float64)
        self.lin_vel = np.asarray(lin_vel, dtype=np.float64)
        self.ang_vel = np.asarray(ang_vel, dtype=np.float64)
        self.height = float(height)

    @classmethod
    def aligned(cls, frame: ReferenceFrame, yaw_offset: np.ndarray) -> "Reference":
        """``frame`` turned about z by ``yaw_offset`` (Teleopit's align_motion_qpos_yaw)."""
        return cls(
            frame.joint_pos,
            frame.joint_vel,
            quat_mul(yaw_offset, frame.anchor_quat),
            quat_apply(yaw_offset, frame.anchor_lin_vel),
            quat_apply(yaw_offset, frame.anchor_ang_vel),
            frame.anchor_pos[2],
        )

    def still(self) -> "Reference":
        """The same pose standing still (the training's held reference)."""
        zero = np.zeros(3)
        return Reference(self.joint_pos, np.zeros_like(self.joint_vel), self.quat, zero, zero, self.height)


def home_reference(yaw: float) -> Reference:
    """The HOME stance at heading ``yaw``, standing still."""
    joints = np.array([NEUTRAL_POSE[name] for name in ROBOT_JOINT_ORDER])
    quat = quat_mul(quat_from_yaw(yaw), np.asarray(HOME_ROOT_QUAT_WXYZ, dtype=np.float64))
    return Reference(joints, np.zeros_like(joints), quat, np.zeros(3), np.zeros(3), HOME_ROOT_POS_Z_M)


def blend_reference(start: Reference, end: Reference, fraction: float, duration_s: float) -> Reference:
    """Smoothstep from ``start`` to ``end``; velocities are the blend's own."""
    s = fraction * fraction * (3.0 - 2.0 * fraction)
    ds = 6.0 * fraction * (1.0 - fraction) / duration_s
    joints = start.joint_pos + s * (end.joint_pos - start.joint_pos)
    joint_vel = ds * (end.joint_pos - start.joint_pos)
    quat = quat_slerp(start.quat, end.quat, s)
    relative = quat_mul(quat_conj(start.quat), end.quat)
    if relative[0] < 0.0:
        relative = -relative
    angle = 2.0 * math.atan2(float(np.linalg.norm(relative[1:])), float(relative[0]))
    axis = relative[1:] / max(1.0e-9, float(np.linalg.norm(relative[1:])))
    ang_vel = quat_apply(start.quat, axis * angle * ds)
    height = start.height + s * (end.height - start.height)
    lin_vel = np.array([0.0, 0.0, ds * (end.height - start.height)])
    return Reference(joints, joint_vel, quat, lin_vel, ang_vel, height)


# ------------------------------------------------------------------ observation
def build_frame(
    reference: Reference,
    *,
    robot_quat: Sequence[float],
    gyro: Sequence[float],
    projected_gravity: Sequence[float],
    joint_pos: Mapping[str, float],
    joint_vel: Mapping[str, float],
    last_action: Sequence[float],
) -> np.ndarray:
    """The 118 values of one tick, in OBSERVATION_SCHEMA order (training's actor_terms)."""

    robot_quat = np.asarray(robot_quat, dtype=np.float64)
    relative = quat_mul(quat_conj(robot_quat), reference.quat)
    values = [
        reference.joint_pos[ACTION_COLUMNS],
        reference.joint_vel[ACTION_COLUMNS],
        matrix_from_quat(relative)[:, :2].reshape(-1),
        np.asarray(gyro, dtype=np.float64),
        np.array([float(joint_pos[name]) - NEUTRAL_POSE[name] for name in ROBOT_JOINT_ORDER]),
        np.array([float(joint_vel[name]) for name in ROBOT_JOINT_ORDER]),
        np.asarray(last_action, dtype=np.float64),
        np.asarray(projected_gravity, dtype=np.float64),
        quat_apply_inverse(robot_quat, reference.lin_vel),
        quat_apply_inverse(robot_quat, reference.ang_vel),
        quat_apply_inverse(reference.quat, GRAVITY_W),
        np.array([reference.height]),
    ]
    frame = np.concatenate(values)
    if frame.shape != (OBSERVATION_WIDTH,):
        raise WholebodyPolicyRuntimeError(f"observation width {frame.shape} is not {OBSERVATION_WIDTH}")
    return frame


class ObservationHistory:
    """The last HISTORY_LENGTH frames, oldest first; a reset fills every row with the first frame."""

    def __init__(self) -> None:
        self._rows = np.zeros((HISTORY_LENGTH, OBSERVATION_WIDTH), dtype=np.float32)
        self._empty = True

    def reset(self) -> None:
        self._empty = True

    def push(self, frame: np.ndarray) -> np.ndarray:
        frame32 = np.asarray(frame, dtype=np.float32)
        if self._empty:
            self._rows[:] = frame32
            self._empty = False
        else:
            self._rows[:-1] = self._rows[1:]
            self._rows[-1] = frame32
        return self._rows[None].copy()


def orientation_diverged(robot_projected_gravity: Sequence[float], reference: Reference) -> bool:
    """The training's anchor_ori end: gravity's z in the two trunk frames differ by more than the line."""
    ref_z = float(quat_apply_inverse(reference.quat, GRAVITY_W)[2])
    return abs(ref_z - float(robot_projected_gravity[2])) > ORIENTATION_TERMINATION


def tilt_from_home_deg(projected_gravity: Sequence[float]) -> float:
    g = np.asarray(projected_gravity, dtype=np.float64)
    norm = float(np.linalg.norm(g))
    if norm < 1.0e-6:
        return math.inf
    cosine = float(np.dot(g / norm, np.asarray(HOME_PROJECTED_GRAVITY, dtype=np.float64)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_from_rotvec(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(0.5 * angle)], math.sin(0.5 * angle) * v / angle])


def rotvec_from_quat(q: Sequence[float]) -> np.ndarray:
    """The rotation vector of ``q``, the short way round."""
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    norm = float(np.linalg.norm(q[1:]))
    if norm < 1.0e-12:
        return np.zeros(3)
    return q[1:] / norm * 2.0 * math.atan2(norm, float(q[0]))


def smootherstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def smootherstep_rate(u: float) -> float:
    """d smootherstep / du."""
    u = min(max(u, 0.0), 1.0)
    return 30.0 * u * u * (1.0 - u) * (1.0 - u)


def join_reference(start: Reference, live: Reference, offset: np.ndarray, u: float) -> Reference:
    """From ``start`` to ``live`` at ``u`` (0..1) of JOIN_S: joints and height
    blended, the trunk's orientation ``exp((1-s) offset) * live``; velocities
    are the blend's own."""
    s, ds = smootherstep(u), smootherstep_rate(u) / JOIN_S
    dj, dh = live.joint_pos - start.joint_pos, live.height - start.height
    turn = quat_from_rotvec((1.0 - s) * offset)
    return Reference(
        start.joint_pos + s * dj,
        s * live.joint_vel + ds * dj,
        quat_mul(turn, live.quat),
        s * live.lin_vel + np.array([0.0, 0.0, ds * dh]),
        -ds * offset + quat_apply(turn, live.ang_vel),
        start.height + s * dh,
    )


def coast_reference(last: Reference, tau: float) -> Reference:
    """``last`` going on at its speed, which fades to zero in COAST_S (the PC's
    Carry): the joints stop at their MJCF ranges and the height does not rise
    above the higher of its own and HOME's."""
    u = min(max(tau / COAST_S, 0.0), 1.0)
    dist = COAST_S * (u - u**3 + 0.5 * u**4)
    fade = 1.0 - u * u * (3.0 - 2.0 * u)
    free = last.joint_pos + last.joint_vel * dist
    joints = np.clip(free, JOINT_LOWER, JOINT_UPPER)
    joint_vel = np.where(joints == free, last.joint_vel * fade, 0.0)
    ceiling = max(last.height, HOME_ROOT_POS_Z_M)
    height = last.height + last.lin_vel[2] * dist
    lin_vel = last.lin_vel * fade
    if height > ceiling:
        height, lin_vel = ceiling, np.array([lin_vel[0], lin_vel[1], 0.0])
    return Reference(
        joints, joint_vel, quat_mul(quat_from_rotvec(last.ang_vel * dist), last.quat), lin_vel,
        last.ang_vel * fade, height,
    )


def trunk_tilt_deg(quat: Sequence[float]) -> float:
    """The trunk's tilt from vertical."""
    _, x, y, _ = (float(v) for v in quat)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


def reference_unsteady(snapshot) -> str | None:
    """Why the newest reference frame does not stand quietly (None: it does).

    The benchmark's standing frame on the trunk (tilted at most
    REFERENCE_STAND_TILT_DEG from vertical, at most REFERENCE_STAND_DROP_M
    below HOME's height, slower than REFERENCE_STAND_SPEED_M_S), not turning
    faster than REFERENCE_STAND_TURN_RAD_S, and the operator's own pose (not
    the PC's interruption reference).  ``snapshot``: the receiver's.
    """
    frame = snapshot.frame
    if frame is None:
        return "no reference frame"
    if frame.flags & (STALE | HELD | LATE):
        return "the PC sends its interruption reference (STALE, HELD or LATE), not the operator"
    tilt = trunk_tilt_deg(frame.anchor_quat)
    if tilt > REFERENCE_STAND_TILT_DEG:
        return f"the operator's trunk is tilted {tilt:.0f} deg"
    height = float(frame.anchor_pos[2])
    if height < HOME_ROOT_POS_Z_M - REFERENCE_STAND_DROP_M:
        return f"the operator's trunk is {100.0 * (HOME_ROOT_POS_Z_M - height):.0f} cm below standing"
    speed = float(np.linalg.norm(frame.anchor_lin_vel))
    if speed > REFERENCE_STAND_SPEED_M_S:
        return f"the operator's trunk moves at {speed:.2f} m/s"
    turn = float(np.linalg.norm(frame.anchor_ang_vel))
    if turn > REFERENCE_STAND_TURN_RAD_S:
        return f"the operator's trunk turns at {turn:.2f} rad/s"
    return None


# ------------------------------------------------------------------ the move
class WholebodyMove(Move):
    """Stand the robot with the whole-body policy and follow the PC's reference stream."""

    def __init__(
        self,
        reference: ReferenceReceiver,
        controller: ControllerProtocol | None = None,
        policy_path: str | Path = Path("src/agents") / POLICY_FILE,
        neutral_return_duration_s: float = 0.8,
        *,
        session: Any | None = None,
        contract: WholebodyContract | None = None,
    ) -> None:
        super().__init__()
        self._reference = reference
        self._controller = controller
        self._neutral_return_duration_s = neutral_return_duration_s
        if session is None:
            loaded = load_wholebody_policy(policy_path)
            session, contract = loaded.session, loaded.contract
            self._self_test_rows = loaded.self_test_rows
        else:
            # An injected session (tests, tools) is checked but not self-tested
            # here; load_wholebody_policy is the deployment path.
            contract = contract if contract is not None else parse_wholebody_policy(session)
            self._self_test_rows = 0
        self._session = session
        self._contract: WholebodyContract = contract
        self._neck_lower = np.array([contract.neck_limits_rad[n][0] for n in HEAD_JOINTS])
        self._neck_upper = np.array([contract.neck_limits_rad[n][1] for n in HEAD_JOINTS])
        self._neck_rate = np.array([contract.neck_slew_rates_rad_s[n] for n in HEAD_JOINTS])
        self._history = ObservationHistory()
        self._last_action = np.zeros(ACTION_WIDTH, dtype=np.float32)
        self.stop_reason: str | None = None
        # The selector clears it while the wire asks for ``walk``: no join.
        self.joins_allowed = True
        # The PC's starts (activations) up to this one are never joined (again).
        self._done_activation = 0
        self._clear_run_state()

    def _clear_run_state(self) -> None:
        self._phase = Phase.ENTER
        self._phase_started_s = 0.0
        self._current: Reference | None = None
        self._start_ref: Reference | None = None
        self._home: Reference | None = None
        self._heading = 0.0
        self._still_ticks = 0
        self._join_start: Reference | None = None
        self._join_offset = np.zeros(3)
        self._yaw_offset = quat_from_yaw(0.0)
        self._stream = -1
        self._activation = -1
        self._lost = 0
        self._last_followed: Reference | None = None
        self._hold_ref: Reference | None = None
        self._stop_s: float | None = None
        self._stop_on_floor = False
        self._neck_target: np.ndarray | None = None
        self._last_time_s: float | None = None
        self._reference_fell_s: float | None = None  # last time the followed reference was past the fall angle
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}
        self._enter_pending: str | None = None
        self._wait_said: int | None = None

    @property
    def contract(self) -> WholebodyContract:
        return self._contract

    @property
    def self_test_rows(self) -> int:
        return self._self_test_rows

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def drives_neck(self) -> bool:
        """The move drives the neck to the PC's reference (the HMD head move yields)."""
        return self.state == MoveState.ACTIVE and self._phase in FOLLOWING

    def can_balance(self, user_input) -> bool:
        _ = user_input
        return True

    def follows_fallen_reference(self, now_s: float, projected_gravity) -> bool:
        """The PC's reference (joined, tracked, or stopped in mid-bow) has its
        trunk past the fall angle (or had within FALL_LAG_S) and the robot
        keeps to the reference's attitude: a trunk past the fall angle is the
        motion followed, not a fall."""
        return (
            self.state == MoveState.ACTIVE
            and self._current is not None
            and (self._phase in FOLLOWING or (self._phase in STOPPING and not self._stop_on_floor))
            and self._reference_fell_s is not None
            and now_s - self._reference_fell_s <= FALL_LAG_S
            and not orientation_diverged(projected_gravity, self._current)
        )

    # Freshness of the stream.
    def _stale_reason(self, snapshot, stream: int | None) -> str | None:
        frame = snapshot.frame
        if frame is None:
            return "no reference frame received"
        if stream is not None and snapshot.stream != stream:
            return "the PC restarted the reference stream"
        if not frame.engaged:
            return "the PC is not tracking (CALIBRATED/ACTIVE off)"
        age = self._reference.now() - snapshot.received_s
        if age > MAX_REFERENCE_AGE_S:
            return f"no reference frame for {age:.2f} s"
        if snapshot.latency_excess_s > MAX_LATENCY_EXCESS_S:
            return f"reference frame {snapshot.latency_excess_s:.2f} s late"
        if snapshot.stale_flag_s > MAX_STALE_FLAG_S:
            return "the PC lost the operator's body (STALE)"
        if snapshot.held_flag_s > MAX_HELD_FLAG_S:
            return "the PC held an excluded pose (HELD)"
        return None

    def join_refusal(self, snapshot) -> str | None:
        """Why the PC's start cannot be joined at all now (None: it can, once
        the robot and the operator stand still)."""
        if not self.joins_allowed:
            return "walk requested"
        reason = self._stale_reason(snapshot, None)
        if reason is None and snapshot.activation <= self._done_activation:
            reason = "waiting for the PC to start tracking again"
        return reason

    def _robot_unsteady(self, obs: Observation) -> str | None:
        """Why the robot does not stand still this tick (None: it does)."""
        if tilt_from_home_deg(obs.robot_state.projected_gravity) >= STILL_TILT_DEG:
            return "the robot's trunk is off the HOME attitude"
        speeds = [obs.robot_state.motor_velocities.get(name) for name in LEG_JOINTS]
        if not all(speed is not None and abs(float(speed)) < STILL_LEG_SPEED_RAD_S for speed in speeds):
            return "the robot's legs move"
        return None

    def request_stop(self, reason: str, now_s: float) -> None:
        """Stop joining or tracking: coast, hold, return, stand (no-op otherwise)."""
        if self.state != MoveState.ACTIVE or self._phase not in FOLLOWING or self._current is None:
            return
        self.stop_reason = reason
        self._last_followed = self._current
        self._stop_s = now_s
        self._stop_on_floor = self._current.height < FLOOR_REF_HEIGHT_M
        self._set_phase(Phase.COAST, now_s)
        print(
            f"Whole-body tracking stopped: {reason}; coasting, holding, then standing at HOME"
            + (" (the reference lies on the floor: a fall is left to the get-up)" if self._stop_on_floor else ""),
            end="\r\n",
            flush=True,
        )

    def _set_phase(self, phase: Phase, now_s: float) -> None:
        self._phase = phase
        self._phase_started_s = now_s

    # Lifecycle.
    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        robot_quat = self._body_quat(obs)
        measured = self._measured_positions(obs)
        now = float(obs.robot_state.time_s)
        self._clear_run_state()
        self.stop_reason = None
        if self._controller is not None:
            # A learned policy runs every joint at the gain it was trained with.
            self._controller.sync_write_kp(ALL_MOTOR_IDS, [KP_RL] * len(ALL_MOTOR_IDS))
        self._enter(measured, robot_quat, now)
        print("Whole-body policy stands the robot (from the measured pose to HOME)", end="\r\n", flush=True)
        for name in OBSERVATION_DOF_ORDER:
            command.target_angles[name] = measured[name]
        self._neck_target = np.clip(
            np.array([measured[name] for name in HEAD_JOINTS]), self._neck_lower, self._neck_upper
        )
        for index, name in enumerate(HEAD_JOINTS):
            command.target_angles[name] = float(self._neck_target[index])
        self._last_time_s = now
        self.state = MoveState.ACTIVE

    def _enter(self, measured: Mapping[str, float], robot_quat: np.ndarray, now: float) -> None:
        """Enter from the measured pose: a fresh history, the previous action zero."""
        joints = np.array([measured[name] for name in ROBOT_JOINT_ORDER])
        zero = np.zeros(3)
        self._start_ref = Reference(joints, np.zeros_like(joints), robot_quat, zero, zero, HOME_ROOT_POS_Z_M)
        self._heading = yaw_of(robot_quat)
        self._home = home_reference(self._heading)
        self._current = self._start_ref
        self._history.reset()
        self._last_action.fill(0.0)
        self._reference_fell_s = None
        self._stop_on_floor = False
        self._still_ticks = 0
        self._set_phase(Phase.ENTER, now)

    def _body_quat(self, obs: Observation) -> np.ndarray:
        quat = np.asarray(obs.robot_state.body_quat, dtype=np.float64)
        if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1.0e-6:
            raise WholebodyPolicyRuntimeError("IMU trunk orientation is invalid")
        return quat / np.linalg.norm(quat)

    def _measured_positions(self, obs: Observation) -> dict[str, float]:
        positions = {}
        for name in ROBOT_JOINT_ORDER:
            value = float(obs.robot_state.motor_positions.get(name, NEUTRAL_POSE[name]))
            if not math.isfinite(value):
                raise WholebodyPolicyRuntimeError(f"motor position for {name} is non-finite")
            positions[name] = value
        return positions

    def _update_reference(
        self, obs: Observation, now: float, robot_quat: np.ndarray, measured: Mapping[str, float]
    ) -> None:
        """Advance the phase and set ``self._current`` for this tick (the reference of the next one)."""
        gravity = obs.robot_state.projected_gravity
        ahead = now + DT
        if self._enter_pending is not None:
            print(f"Whole-body policy: {self._enter_pending}; entering again from the measured pose",
                  end="\r\n", flush=True)
            self._enter_pending = None
            self._enter(measured, robot_quat, now)
        if self._phase in FOLLOWING:
            self._follow(now, ahead)
        if self._phase == Phase.COAST:
            assert self._last_followed is not None and self._stop_s is not None
            tau = ahead - self._stop_s
            if tau >= COAST_S:
                self._hold_ref = coast_reference(self._last_followed, COAST_S).still()
                self._set_phase(Phase.HOLD, now)
            else:
                self._current = coast_reference(self._last_followed, tau)
        if self._phase == Phase.HOLD:
            assert self._hold_ref is not None and self._stop_s is not None
            self._current = self._hold_ref
            if now - self._stop_s >= STOP_HOLD_S:
                # HOME at the robot's heading, or the held one while the trunk is tilted.
                upright = tilt_from_home_deg(gravity) <= HEADING_FREEZE_TILT_DEG
                self._heading = yaw_of(robot_quat if upright else self._hold_ref.quat)
                self._home = home_reference(self._heading)
                self._set_phase(Phase.RETURN, now)
        if self._phase == Phase.RETURN:
            assert self._hold_ref is not None and self._home is not None
            fraction = min(1.0, (ahead - self._phase_started_s) / RETURN_S)
            self._current = blend_reference(self._hold_ref, self._home, fraction, RETURN_S)
            if fraction >= 1.0:
                self._set_phase(Phase.STAND, now)
                self._still_ticks = 0
                print("Whole-body policy: back at the HOME stance, standing", end="\r\n", flush=True)
        if self._phase in FOLLOWING + STOPPING and orientation_diverged(gravity, self._current):
            # The training ends its episode here and starts the next on the
            # reference: enter again from where the robot is.
            self._enter(measured, robot_quat, now)
            print("Whole-body policy: the trunk left the reference's attitude; entering again from the measured pose",
                  end="\r\n", flush=True)
        if self._phase == Phase.ENTER:
            assert self._start_ref is not None and self._home is not None
            fraction = min(1.0, (ahead - self._phase_started_s) / ENTER_S)
            self._current = blend_reference(self._start_ref, self._home, fraction, ENTER_S)
            if fraction >= 1.0:
                self._set_phase(Phase.STAND, now)
                self._still_ticks = 0
        if self._phase == Phase.STAND:
            self._stand(obs, now, ahead, robot_quat, measured)

    def _stand(
        self, obs: Observation, now: float, ahead: float, robot_quat: np.ndarray, measured: Mapping[str, float]
    ) -> None:
        """The HOME stance at the IMU's heading; join the PC's start when everything stands still."""
        if tilt_from_home_deg(obs.robot_state.projected_gravity) <= HEADING_FREEZE_TILT_DEG:
            turn = wrap_angle(yaw_of(robot_quat) - self._heading)
            self._heading = wrap_angle(self._heading + max(-HEADING_STEP_RAD, min(HEADING_STEP_RAD, turn)))
        # The heading moves, the reference does not turn: its angular velocity stays zero.
        self._home = home_reference(self._heading)
        self._current = self._home
        snapshot = self._reference.read()
        wait = self._robot_unsteady(obs) or reference_unsteady(snapshot)
        self._still_ticks = 0 if wait is not None else self._still_ticks + 1
        refusal = self.join_refusal(snapshot)
        if refusal is not None:
            return
        if wait is None and self._still_ticks >= STILL_TICKS:
            self._join(snapshot, now, measured)
            self._follow(now, ahead)
        elif self._wait_said != snapshot.activation:
            self._wait_said = snapshot.activation
            print(
                f"Whole-body tracking: PC start {snapshot.activation} waits for the robot and the operator to stand "
                f"still for {STILL_TICKS * DT:.1f} s (now: {wait or 'still, counting'})",
                end="\r\n",
                flush=True,
            )

    def _join(self, snapshot, now: float, measured: Mapping[str, float]) -> None:
        frame = snapshot.frame
        assert frame is not None and self._current is not None
        # The PC's reference turned about z to the stand's heading.
        self._yaw_offset = quat_from_yaw(self._heading - yaw_of(frame.anchor_quat))
        live = Reference.aligned(frame, self._yaw_offset)
        self._join_start = self._current
        self._join_offset = rotvec_from_quat(quat_mul(self._join_start.quat, quat_conj(live.quat)))
        self._stream, self._lost, self._activation = snapshot.stream, snapshot.lost, snapshot.activation
        self._done_activation = max(self._done_activation, snapshot.activation)
        self.stop_reason = None
        self._set_phase(Phase.JOIN, now)
        # The HMD head move may have had the neck: the neck slews from where it is.
        self._neck_target = np.clip(
            np.array([measured[name] for name in HEAD_JOINTS]), self._neck_lower, self._neck_upper
        )
        print(
            f"Whole-body tracking joins PC start {snapshot.activation} (reference turned "
            f"{math.degrees(2.0 * math.atan2(self._yaw_offset[3], self._yaw_offset[0])):+.0f} deg to the robot's "
            "heading)",
            end="\r\n",
            flush=True,
        )

    def _follow(self, now: float, ahead: float) -> None:
        """Join or track the PC's reference, or stop."""
        snapshot = self._reference.read()
        reason = self._stale_reason(snapshot, self._stream)
        lost, self._lost = snapshot.lost - self._lost, snapshot.lost
        if reason is None and lost > MAX_SEQUENCE_GAP:
            reason = f"{lost} reference frames lost"
        if reason is None and snapshot.activation != self._activation:
            reason = "the PC started tracking again"
        if reason is not None:
            self.request_stop(reason, now)
            return
        assert snapshot.frame is not None
        live = Reference.aligned(snapshot.frame, self._yaw_offset)
        if self._phase == Phase.JOIN:
            assert self._join_start is not None
            u = (ahead - self._phase_started_s) / JOIN_S
            if u < 1.0:
                self._current = join_reference(self._join_start, live, self._join_offset, u)
                return
            self._set_phase(Phase.TRACK, now)
        self._current = live

    def step(self, obs: Observation, command: MotorCommand) -> None:
        gravity = [float(v) for v in obs.robot_state.projected_gravity]
        if len(gravity) != 3 or not all(math.isfinite(v) for v in gravity):
            raise WholebodyPolicyRuntimeError("projected gravity is invalid")
        measured = self._measured_positions(obs)
        now = float(obs.robot_state.time_s)
        if gravity[2] > FALL_GRAVITY_Z and not self.follows_fallen_reference(now, gravity):
            # Falling: the scheduler's fall detection owns what follows; the
            # neck keeps its last target.
            for name in OBSERVATION_DOF_ORDER:
                command.target_angles[name] = measured[name]
            if self._neck_target is not None:
                for index, name in enumerate(HEAD_JOINTS):
                    command.target_angles[name] = float(self._neck_target[index])
            return
        robot_quat = self._body_quat(obs)
        self._update_reference(obs, now, robot_quat, measured)
        assert self._current is not None
        if self._phase in FOLLOWING + STOPPING and float(
            quat_apply_inverse(self._current.quat, GRAVITY_W)[2]
        ) > FALL_GRAVITY_Z:
            self._reference_fell_s = now
        frame = build_frame(
            self._current,
            robot_quat=robot_quat,
            gyro=[float(v) for v in obs.robot_state.gyro],
            projected_gravity=gravity,
            joint_pos=measured,
            joint_vel=obs.robot_state.motor_velocities,
            last_action=self._last_action,
        )
        if not np.isfinite(frame).all():
            raise WholebodyPolicyRuntimeError("observation is not finite")
        history = self._history.push(frame)
        outputs = self._session.run(None, {INPUT_NAME: history})
        action = np.asarray(outputs[0], dtype=np.float64)
        if len(outputs) != 1 or action.shape != (1, ACTION_WIDTH) or not np.isfinite(action).all():
            raise WholebodyPolicyRuntimeError(f"unsafe policy output {action.shape}")
        raw = action[0].astype(np.float32)
        if not np.isfinite(raw).all():
            raise WholebodyPolicyRuntimeError("raw action is not finite in float32")
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            target = NEUTRAL_POSE[name] + float(raw[index])
            command.target_angles[name] = max(-SERVO_TARGET_RANGE_RAD, min(SERVO_TARGET_RANGE_RAD, target))
        self._last_action = raw.copy()
        self._drive_neck(now, command)

    def _drive_neck(self, now: float, command: MotorCommand) -> None:
        """The contract's neck: the reference's neck angles, clamped and slewed."""
        assert self._current is not None and self._neck_target is not None
        goal = np.clip(self._current.joint_pos[NECK_COLUMNS], self._neck_lower, self._neck_upper)
        dt = max(0.0, min(0.1, now - (self._last_time_s if self._last_time_s is not None else now)))
        step = self._neck_rate * dt
        self._neck_target = np.clip(
            self._neck_target + np.clip(goal - self._neck_target, -step, step), self._neck_lower, self._neck_upper
        )
        for index, name in enumerate(HEAD_JOINTS):
            command.target_angles[name] = float(self._neck_target[index])
        self._last_time_s = now

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._phase in FOLLOWING:
            # Interrupted (policy off, get-up): this start was joined, so it is
            # not joined again.
            self.stop_reason = "stopped by the scheduler"
        if "getup" in obs.user_input.active_moves:
            self._finish_stop()
            return
        now = float(obs.robot_state.time_s)
        if self._stop_start_time_s is None:
            self._stop_start_time_s = now
            self._stop_start_angles = self._measured_positions(obs)
        fraction = min(1.0, max(0.0, now - self._stop_start_time_s) / max(1e-6, self._neutral_return_duration_s))
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        for name in ROBOT_JOINT_ORDER:
            start = self._stop_start_angles[name]
            command.target_angles[name] = start + (NEUTRAL_POSE[name] - start) * blend
        if fraction >= 1.0:
            if self._controller is not None:
                self._controller.sync_write_kp(ALL_MOTOR_IDS, [KP_HARDWARE_NEUTRAL] * len(ALL_MOTOR_IDS))
            self._finish_stop()

    def _finish_stop(self) -> None:
        self._last_action.fill(0.0)
        self._history.reset()
        self._clear_run_state()
        self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        # The scheduler held the goals while callbacks were skipped: the next
        # step enters again from the measured pose (a fresh history); a start
        # joined is not joined again.
        _ = obs
        self._history.reset()
        self._last_action.fill(0.0)
        if self.state == MoveState.ACTIVE:
            self._enter_pending = "safety hold"
