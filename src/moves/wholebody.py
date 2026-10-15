# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Whole-body tracking: the robot follows the operator's retargeted body.

The policy (contract microban-wholebody-1, wholebody_contract.py) drives the
18 body joints, arms included: target = clip(HOME + raw, -pi, +pi).  Its
observation is built exactly as in training (mjlab_microban
``wholebody/observations.py`` and ``env_cfg.actor_terms``): 118 values per
control tick, the last 10 ticks oldest first, every row the first frame after
a start.  The reference comes from the PC (input/wholebody_reference.py),
turned about z at the start so that its trunk heading is the IMU's
(``reference_yaw: aligned_to_imu_yaw_at_start``); the ``*_b`` terms use the
IMU trunk orientation with its yaw.

Start: like the training benchmark's teleoperation start, the reference
stands still for SETTLE_S (the newest frame's pose, velocities zero), then
plays live.

Tracking runs while the PC's frames say CALIBRATED and ACTIVE (the operator
started it on the PC).  A RESTART frame starts it again in place: the yaw is
aligned again, the history rebuilt.

Stop: the PC ends tracking (ACTIVE or CALIBRATED off), a stale reference (no
frame for MAX_REFERENCE_AGE_S, a jump of more than MAX_SEQUENCE_GAP frames, a
frame later than MAX_LATENCY_EXCESS_S, a restarted PC, the PC's STALE flag
for more than MAX_STALE_FLAG_S or its HELD flag for more than
MAX_HELD_FLAG_S), a trunk off the reference's attitude (the z components of
gravity in the two trunk frames differ by more than ORIENTATION_TERMINATION,
the training's ``anchor_ori`` end) or the selector's request all lead to the
same sequence, the policy still running:

1. hold: the last reference pose, velocities zero, for STOP_HOLD_S;
2. return: the reference moves to the HOME stance (HOME joints, the HOME
   trunk attitude at the robot's heading, HOME trunk height) in RETURN_S;
3. stand: the HOME reference until the trunk is within HANDOFF_TILT_DEG of
   the HOME attitude for HANDOFF_TICKS ticks (or HANDOFF_MAX_S passed); then
   ``handoff_ready`` asks the selector to give the robot to the standing
   policy.

The stopped activation does not start again: tracking resumes only when the
PC starts a new one (ACTIVE off then on, or RESTART).  A trunk tilted more
than 60 deg from vertical holds the measured pose, as the
PICO policy does: the scheduler's fall detection then starts the get-up.
The exception is a trunk following the reference's: while the followed
reference (live, held or returning) is itself past the fall angle, or was
within FALL_LAG_S (the robot follows late), and the robot keeps to its
attitude (ORIENTATION_TERMINATION), the policy keeps running and the
scheduler does not count a fall (``follows_fallen_reference``).  A robot that
leaves the reference's attitude stops as above and then counts as fallen.

Neck: its targets are the reference's neck angles (the PC fills them from the
operator's head), clamped to the contract's ``neck_limits_rad_json`` and
slewed at ``neck_slew_rates_rad_s_json`` from the measured neck at the start,
as training's ReferenceNeckTarget does; the hold and the return to HOME move
the neck with the rest of the reference.  The HMD head move yields
(``WholebodyClaim``).
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
from input.wholebody_reference import ReferenceFrame, ReferenceReceiver
from moves.move import MotorCommand, Move, MoveState
from observer import Observation
from policy_contract import HEAD_JOINTS
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
ALL_MOTOR_IDS = list(MOTOR_TO_ID.values())
GRAVITY_W = np.array([0.0, 0.0, -1.0])

SETTLE_S = 1.0
MAX_REFERENCE_AGE_S = 0.1
MAX_SEQUENCE_GAP = 5
MAX_LATENCY_EXCESS_S = 0.15
MAX_STALE_FLAG_S = 0.25  # the PC sets STALE after 0.25 s without the operator's body
# HELD is followed for up to 1 s: through it the PC sends its interruption
# reference and comes back to the operator with no frame changing more than
# the training reference's 99.9th percentile per tick (mjlab_microban
# docs/wholebody_reference_stream.md, from its preflight-fixes; an older PC
# jumps from the held pose to the live one in one frame when HELD clears).
MAX_HELD_FLAG_S = 1.0
ORIENTATION_TERMINATION = 1.0
STOP_HOLD_S = 0.5
RETURN_S = 1.5
HANDOFF_TILT_DEG = 12.0
HANDOFF_TICKS = 10
HANDOFF_MAX_S = 3.0
FALL_GRAVITY_Z = -0.5
# A trunk past the fall angle is not a fall while the reference's trunk is
# past it too, or was within FALL_LAG_S (the robot follows late), and the robot
# keeps to its attitude (ORIENTATION_TERMINATION).
FALL_LAG_S = 0.5


class WholebodyPolicyRuntimeError(RuntimeError):
    """Inference produced a result that is unsafe to send to the servos."""


class Phase(enum.Enum):
    SETTLE = "settle"
    TRACK = "track"
    HOLD = "hold"
    RETURN = "return"
    STAND = "stand"


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


# ------------------------------------------------------------------ the move
class WholebodyMove(Move):
    """Run the whole-body tracking policy on the PC's reference stream."""

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
        self._phase = Phase.SETTLE
        self.stop_reason: str | None = None
        # The PC activation that stopped or was refused: never tracked again.
        self._stopped_activation: int | None = None
        self._clear_run_state()

    def _clear_run_state(self) -> None:
        self._yaw_offset = quat_from_yaw(0.0)
        self._stream = -1
        self._activation = -1
        self._lost = 0
        self._started_s = 0.0
        self._phase_started_s = 0.0
        self._current: Reference | None = None
        self._phase_start_ref: Reference | None = None
        self._home: Reference | None = None
        self._settled_ticks = 0
        self._neck_target: np.ndarray | None = None
        self._last_time_s: float | None = None
        self._reference_fell_s: float | None = None  # last time the followed reference was past the fall angle
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}
        self.handoff_ready = False

    @property
    def contract(self) -> WholebodyContract:
        return self._contract

    @property
    def self_test_rows(self) -> int:
        return self._self_test_rows

    @property
    def phase(self) -> Phase:
        return self._phase

    def can_balance(self, user_input) -> bool:
        _ = user_input
        return True

    def follows_fallen_reference(self, now_s: float, projected_gravity) -> bool:
        """The reference's trunk is past the fall angle (or was within
        FALL_LAG_S) and the robot keeps to the reference's attitude: a trunk
        past the fall angle is the motion followed, not a fall."""
        return (
            self.state == MoveState.ACTIVE
            and self._current is not None
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

    def start_refusal(self) -> str | None:
        """Why tracking cannot start now (None: it can)."""
        snapshot = self._reference.read()
        reason = self._stale_reason(snapshot, None)
        # An activation already run (stopped, or interrupted by the scheduler's
        # gate without a stop) is not resumed.
        if reason is None and snapshot.activation in (self._stopped_activation, self._activation):
            reason = f"waiting for the PC to start tracking again (stopped: {self.stop_reason or 'interrupted'})"
        return reason

    def refuse(self, reason: str) -> None:
        """The selector could not run this activation: do not start it again."""
        self._stopped_activation = self._reference.read().activation
        self.stop_reason = reason

    def request_stop(self, reason: str, now_s: float) -> None:
        """Begin the hold -> HOME -> hand-off sequence (no-op once begun)."""
        if self.state == MoveState.ACTIVE and self._phase in (Phase.SETTLE, Phase.TRACK):
            self.stop_reason = reason
            self._stopped_activation = self._activation
            print(f"Whole-body tracking stopped: {reason}; holding, then returning to HOME", end="\r\n", flush=True)
            if self._current is not None:
                self._current = self._current.still()
            self._enter(Phase.HOLD, now_s)

    def _enter(self, phase: Phase, now_s: float) -> None:
        self._phase = phase
        self._phase_started_s = now_s
        self._phase_start_ref = self._current

    # Lifecycle.
    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        reason = self.start_refusal()
        if reason is not None:
            raise WholebodyPolicyRuntimeError(f"cannot start whole-body tracking: {reason}")
        robot_quat = self._body_quat(obs)
        now = float(obs.robot_state.time_s)
        self._clear_run_state()
        self._begin(self._reference.read(), robot_quat, now)
        self.stop_reason = None
        measured = self._measured_positions(obs)
        if self._controller is not None:
            # A learned policy runs every joint at the gain it was trained with.
            self._controller.sync_write_kp(ALL_MOTOR_IDS, [KP_RL] * len(ALL_MOTOR_IDS))
        for name in OBSERVATION_DOF_ORDER:
            command.target_angles[name] = measured[name]
        self._neck_target = np.clip(
            np.array([measured[name] for name in HEAD_JOINTS]), self._neck_lower, self._neck_upper
        )
        for index, name in enumerate(HEAD_JOINTS):
            command.target_angles[name] = float(self._neck_target[index])
        self._last_time_s = now
        self.state = MoveState.ACTIVE

    def _begin(self, snapshot, robot_quat: np.ndarray, now: float) -> None:
        """Start following ``snapshot``'s activation: align, settle, fresh history."""
        frame = snapshot.frame
        assert frame is not None
        # Turn the reference about z so its trunk heading is the IMU's now.
        self._yaw_offset = yaw_alignment(robot_quat, frame.anchor_quat)
        self._stream = snapshot.stream
        self._activation = snapshot.activation
        self._lost = snapshot.lost
        self._started_s = now
        self._phase = Phase.SETTLE
        self._phase_started_s = now
        self._current = Reference.aligned(frame, self._yaw_offset).still()
        self._history.reset()
        self._last_action.fill(0.0)
        print(
            f"Whole-body tracking started (PC activation {snapshot.activation}; reference turned "
            f"{math.degrees(2.0 * math.atan2(self._yaw_offset[3], self._yaw_offset[0])):+.0f} deg to the IMU yaw)",
            end="\r\n",
            flush=True,
        )

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

    def _update_reference(self, obs: Observation, now: float, robot_quat: np.ndarray) -> None:
        """Advance the phase and set ``self._current`` for this tick."""
        gravity = obs.robot_state.projected_gravity
        if self._phase in (Phase.SETTLE, Phase.TRACK):
            snapshot = self._reference.read()
            reason = self._stale_reason(snapshot, self._stream)
            lost, self._lost = snapshot.lost - self._lost, snapshot.lost
            if reason is None and lost > MAX_SEQUENCE_GAP:
                reason = f"{lost} reference frames lost"
            if reason is None and snapshot.activation != self._activation:
                # RESTART (or ACTIVE again) from the PC: start following anew.
                print("Whole-body tracking restarted by the PC", end="\r\n", flush=True)
                self._begin(snapshot, robot_quat, now)
            if reason is None:
                frame = snapshot.frame
                assert frame is not None
                live = Reference.aligned(frame, self._yaw_offset)
                if orientation_diverged(gravity, live):
                    reason = "the trunk left the reference's attitude"
                else:
                    if self._phase == Phase.SETTLE and now - self._started_s >= SETTLE_S:
                        self._enter(Phase.TRACK, now)
                    self._current = live.still() if self._phase == Phase.SETTLE else live
            if reason is not None:
                # The hold keeps the last reference that was followed, never
                # the stale or diverged one.
                self.request_stop(reason, now)
        if self._phase == Phase.HOLD and now - self._phase_started_s >= STOP_HOLD_S:
            self._home = home_reference(yaw_of(robot_quat))
            self._enter(Phase.RETURN, now)
        if self._phase == Phase.RETURN:
            assert self._phase_start_ref is not None and self._home is not None
            fraction = min(1.0, (now - self._phase_started_s) / RETURN_S)
            self._current = blend_reference(self._phase_start_ref, self._home, fraction, RETURN_S)
            if fraction >= 1.0:
                self._current = self._home
                self._enter(Phase.STAND, now)
        if self._phase == Phase.STAND:
            self._current = self._home
            settled = tilt_from_home_deg(gravity) < HANDOFF_TILT_DEG
            self._settled_ticks = self._settled_ticks + 1 if settled else 0
            if not self.handoff_ready and (
                self._settled_ticks >= HANDOFF_TICKS or now - self._phase_started_s >= HANDOFF_MAX_S
            ):
                self.handoff_ready = True
                print(
                    f"Whole-body tracking back at HOME ({tilt_from_home_deg(gravity):.1f} deg from the HOME "
                    "attitude); the standing policy takes over",
                    end="\r\n",
                    flush=True,
                )

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
        self._update_reference(obs, now, robot_quat)
        assert self._current is not None
        if float(quat_apply_inverse(self._current.quat, GRAVITY_W)[2]) > FALL_GRAVITY_Z:
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
        if self._phase in (Phase.SETTLE, Phase.TRACK) and self._activation >= 0:
            # Interrupted (policy off, get-up): this activation is not resumed.
            self.stop_reason = "stopped by the scheduler"
            self._stopped_activation = self._activation
            self._phase = Phase.HOLD
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
        self._phase = Phase.SETTLE
        self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        # The scheduler held the goals while callbacks were skipped: restart the
        # history and stop tracking (the reference moved on meanwhile).
        _ = obs
        self._history.reset()
        self._last_action.fill(0.0)
        if self.state == MoveState.ACTIVE:
            self.request_stop("safety hold", float(obs.robot_state.time_s))
