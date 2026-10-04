# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from constants import (
    HOME_TRUNK_PITCH_RAD,
    KP_DEFAULT,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    POLICY_ACTION_SCALE,
    SERVO_TARGET_RANGE_RAD,
)
from controller import ControllerProtocol
from home_pose import home_pose_stamp_matches
from moves.move import MotorCommand, Move, MoveState
from observer import Observation

# Set to True to log motor positions and voltages during the walk move
# Note: requires to set observe_voltage = True in the Observer to log voltages
LOGGING = False

# Policy name. Every input source (PICO fallback, GC300, keyboard, gamepad)
# runs this one walking model.
AGENT_NAME = "walk.onnx"

# Deployment contract of the walking actor, read from the ONNX metadata and
# checked at construction (fail closed): target = clip(NEUTRAL_POSE + raw *
# 1.0, -pi, +pi) on the 18 OBSERVATION_DOF_ORDER joints -- no software clip,
# only the servo's one-turn goal range (constants.SERVO_TARGET_RANGE_RAD) --
# and the previous action observation is the policy's own raw previous output,
# all at the forward-lean HOME (trunk 10 deg forward; mjlab_microban
# export_walk_onnx.py). The exporter also writes the full HOME (joints and
# root pose) as the ``home_pose`` JSON, which must be the robot's HOME. The
# centered upright HOME actors (v3_centered_home_servo_range) and the older
# v2_centered_home_clip157 (+-1.57) ones are rejected.
WALK_CONTRACT_VERSION = "v4_forward_lean_home_servo_range"
WALK_PREVIOUS_ACTION_SEMANTICS = "raw_policy_output"
# default_joint_pos must reproduce NEUTRAL_POSE to this tolerance; the
# exporter therefore has to write full-precision floats (mjlab's base
# exporter rounds to 3 decimals, which this check rejects on purpose).
WALK_DEFAULT_POSE_TOLERANCE_RAD = 1.0e-6
_SCALE_TOLERANCE = 1.0e-6
_CLIP_TOLERANCE_RAD = 1.0e-6

# Neck roll/pitch joint ranges (rad), from src/model/mjcf/robot.xml. The stabilization
# below clips to these so a large trunk tilt can't request an out-of-range neck target.
NECK_ROLL_RANGE = (-0.436332, 0.436332)
NECK_PITCH_RANGE = (-1.570796, 0.436332)


class WalkPolicyContractError(ValueError):
    """The installed walk.onnx does not carry the deployed walking contract."""


def _body_roll_pitch(body_quat: list[float]) -> tuple[float, float]:
    """Roll (about +X) and pitch (about +Y) of the trunk frame, same convention as
    scheduler.py's IMU display and as the neck_roll/neck_pitch joint axes (robot.xml)."""
    w, x, y, z = body_quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def _metadata_floats(meta: dict[str, str], key: str) -> list[float]:
    value = meta.get(key)
    if not value:
        raise WalkPolicyContractError(f"walk policy metadata lacks {key}")
    try:
        values = [float(part) for part in value.split(",")]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WalkPolicyContractError(f"walk policy metadata {key} is not numeric") from exc
    if not all(math.isfinite(item) for item in values):
        raise WalkPolicyContractError(f"walk policy metadata {key} is not finite")
    return values


def _metadata_exact(meta: dict[str, str], key: str, expected: str) -> None:
    actual = meta.get(key)
    if actual != expected:
        raise WalkPolicyContractError(
            f"walk policy metadata {key}={actual!r}, expected {expected!r}"
        )


def parse_walk_contract(
    session: Any,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], bool]:
    """Validate a walking actor and return (defaults, clip, uses_reference_phase).

    Raises WalkPolicyContractError for every missing or inconsistent field.
    """
    meta = dict(session.get_modelmeta().custom_metadata_map)
    _metadata_exact(meta, "walk_contract_version", WALK_CONTRACT_VERSION)
    _metadata_exact(meta, "previous_action_semantics", WALK_PREVIOUS_ACTION_SEMANTICS)
    if not home_pose_stamp_matches(meta.get("home_pose")):
        raise WalkPolicyContractError(
            "walk policy home_pose is missing or is not the robot's HOME "
            "(joints NEUTRAL_POSE, root HOME_ROOT_POS_Z_M / HOME_ROOT_QUAT_WXYZ)"
        )

    action_count = len(OBSERVATION_DOF_ORDER)
    action_names = meta.get("action_joint_names")
    if action_names is not None and action_names.split(",") != OBSERVATION_DOF_ORDER:
        raise WalkPolicyContractError(
            "walk policy action_joint_names differ from OBSERVATION_DOF_ORDER"
        )

    joint_names = (meta.get("joint_names") or "").split(",")
    defaults = _metadata_floats(meta, "default_joint_pos")
    if (
        len(joint_names) != len(defaults)
        or len(set(joint_names)) != len(joint_names)
        or not set(joint_names) <= set(NEUTRAL_POSE)
        or not set(OBSERVATION_DOF_ORDER) <= set(joint_names)
    ):
        raise WalkPolicyContractError(
            "walk policy joint_names/default_joint_pos are malformed"
        )
    for name, value in zip(joint_names, defaults):
        if abs(value - NEUTRAL_POSE[name]) > WALK_DEFAULT_POSE_TOLERANCE_RAD:
            raise WalkPolicyContractError(
                f"walk policy default_joint_pos[{name}]={value!r} differs from "
                f"NEUTRAL_POSE {NEUTRAL_POSE[name]!r} (trained at another HOME?)"
            )

    scale = _metadata_floats(meta, "action_scale")
    if len(scale) not in (1, action_count) or any(
        abs(value - POLICY_ACTION_SCALE) > _SCALE_TOLERANCE for value in scale
    ):
        raise WalkPolicyContractError(f"walk policy action_scale must be {POLICY_ACTION_SCALE}")

    lower = _metadata_floats(meta, "action_clip_lower")
    upper = _metadata_floats(meta, "action_clip_upper")
    if len(lower) != action_count or len(upper) != action_count or any(
        abs(lo + SERVO_TARGET_RANGE_RAD) > _CLIP_TOLERANCE_RAD
        or abs(hi - SERVO_TARGET_RANGE_RAD) > _CLIP_TOLERANCE_RAD
        for lo, hi in zip(lower, upper)
    ):
        raise WalkPolicyContractError(
            "walk policy action_clip_lower/upper must be the servo range "
            f"+-{SERVO_TARGET_RANGE_RAD!r} rad on all {action_count} action joints"
        )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    # base_obs = gyro(3) + proj_grav(3) + pos(N) + vel(N) + action(N) + cmd(3)
    # phase_obs = base_obs + phase(2)
    base_obs_size = 3 + 3 + 3 * action_count + 3
    if len(inputs) != 1 or len(outputs) != 1:
        raise WalkPolicyContractError("walk policy must have one input and one output")
    input_shape = list(inputs[0].shape)
    if input_shape not in ([1, base_obs_size], [1, base_obs_size + 2]):
        raise WalkPolicyContractError(f"walk policy input shape {input_shape} is unsupported")
    if list(outputs[0].shape) != [1, action_count]:
        raise WalkPolicyContractError(
            f"walk policy output shape {list(outputs[0].shape)} is not [1, {action_count}]"
        )

    # Validated equal to NEUTRAL_POSE above; use the full-precision robot copy.
    default_pose = {name: float(NEUTRAL_POSE[name]) for name in joint_names}
    # Validated equal to +-pi above; apply the exact servo range.
    clip = {
        name: (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)
        for name in OBSERVATION_DOF_ORDER
    }
    return default_pose, clip, input_shape[1] > base_obs_size


class WalkMove(Move):
    """Walk using a RL policy trained in simulation."""

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        neutral_return_duration_s: float = 0.8,
        *,
        policy_path: str | Path | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self._last_safe_targets: dict[str, float] = {}
        self.policy_faulted = False

        # Load ONNX policy
        if session is None:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            path = policy_path if policy_path is not None else Path("src/agents") / AGENT_NAME
            session = ort.InferenceSession(str(path), sess_options=session_options)
        self._ort_session = session

        # Fail closed: an actor trained at another HOME, with another action
        # rule or without the contract stamp never reaches the motors.
        (
            self._default_pose,
            self._action_clip,
            self._use_reference_phase,
        ) = parse_walk_contract(self._ort_session)
        self.action_scale = POLICY_ACTION_SCALE

        # Head stabilization: counter-rotate neck_roll/neck_pitch against trunk tilt so the
        # head stays level while walking. Not part of the RL policy (neck is excluded from
        # its action/observation space) — this is a separate proportional control law layered
        # on top; gain=1.0 is full cancellation of the extracted roll/pitch.
        self._neck_stabilize_gain = 1.0
        self._neutral_return_duration_s = neutral_return_duration_s
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}

        self._phase_step = 0
        self._phase_total_steps = 20

        # Safety parameters. A fall is a physical attitude, so this stays measured from
        # vertical (trunk tilt > 60 deg), like the scheduler's fall debounce; HOME's 10 deg
        # forward lean still leaves 50 deg before it trips.
        self._projected_gravity_z_threshold = -0.5  # Threshold for detecting a fall based on projected gravity

        # Logging
        self.position = {
            "head": [],
            "left_hip_yaw": [],
            "left_hip_roll": [],
            "left_hip_pitch": [],
            "left_knee": [],
            "left_ankle_pitch": [],
            "left_ankle_roll": [],
            "right_hip_yaw": [],
            "right_hip_roll": [],
            "right_hip_pitch": [],
            "right_knee": [],
            "right_ankle_pitch": [],
            "right_ankle_roll": [],
            "left_shoulder_pitch": [],
            "left_shoulder_roll": [],
            "left_elbow": [],
            "right_shoulder_pitch": [],
            "right_shoulder_roll": [],
            "right_elbow": [],
        }
        self.voltage = {
            "head": [],
            "left_hip_yaw": [],
            "left_hip_roll": [],
            "left_hip_pitch": [],
            "left_knee": [],
            "left_ankle_pitch": [],
            "left_ankle_roll": [],
            "right_hip_yaw": [],
            "right_hip_roll": [],
            "right_hip_pitch": [],
            "right_knee": [],
            "right_ankle_pitch": [],
            "right_ankle_roll": [],
            "left_shoulder_pitch": [],
            "left_shoulder_roll": [],
            "left_elbow": [],
            "right_shoulder_pitch": [],
            "right_shoulder_roll": [],
            "right_elbow": [],
        }
        
    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        # A learned-policy runtime fault can hand ownership to this proven actor in
        # one control cycle. Never carry action/phase recurrence from an older walk
        # activation into that emergency handoff.
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self._phase_step = 0
        self.policy_faulted = False
        if self._controller is not None:
            ids = [MOTOR_TO_ID[name] for name in OBSERVATION_DOF_ORDER]
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        # The scheduler's base command is neutral. Hold the measured gait joints on
        # the transition tick so enabling the policy cannot create a one-frame jump.
        for name in OBSERVATION_DOF_ORDER:
            command.target_angles[name] = obs.robot_state.motor_positions.get(
                name, NEUTRAL_POSE[name]
            )
        self._last_safe_targets = {
            name: command.target_angles[name] for name in OBSERVATION_DOF_ORDER
        }
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self.state = MoveState.ACTIVE

    def can_balance(self, user_input) -> bool:
        # The actor is loaded in __init__; it stands in place at zero velocity.
        _ = user_input
        return True

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Update reference phase
        if self._use_reference_phase:
            commanded_vel = np.mean([np.abs(obs.user_input.velocity["vx"]), np.abs(obs.user_input.velocity["vy"]), np.abs(obs.user_input.velocity["vtheta"])])
            if commanded_vel > 0.01:
                self._phase_step += 1
            else:
                self._phase_step = 0

        # Safety check: if the robot is fallen, stop the policy
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            # Fall detection is debounced before GetupMove takes ownership. Hold the
            # measured gait pose during that window instead of letting the scheduler's
            # neutral base command create an instantaneous 18-joint jump.
            for name in OBSERVATION_DOF_ORDER:
                command.target_angles[name] = obs.robot_state.motor_positions.get(
                    name, NEUTRAL_POSE[name]
                )
            self._last_safe_targets = {
                name: command.target_angles[name] for name in OBSERVATION_DOF_ORDER
            }
            return

        if self.policy_faulted:
            self._hold_safe_targets(obs, command)
            return

        # Run policy. Like GetupMove, only an unusable output (wrong width,
        # non-finite value, or an inference failure) is a fault: it latches,
        # and the last bounded targets are held until the move restarts.
        # A large finite raw output is legitimate; the clip below bounds it.
        input_obs: list[float] = []
        try:
            input_obs = self.build_observation(obs)
            ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
            ort_outs = self._ort_session.run(None, ort_inputs)
            action = [float(value) for value in ort_outs[0][0]]
            error = None
        except Exception as exc:  # noqa: BLE001 - inference boundary
            action = []
            error = exc
        if (
            error is not None
            or len(action) != len(OBSERVATION_DOF_ORDER)
            or any(not math.isfinite(value) for value in action)
        ):
            finite_action = [abs(value) for value in action if math.isfinite(value)]
            finite_obs = [abs(value) for value in input_obs if math.isfinite(value)]
            print(
                "Walk actor output invalid; holding last targets: "
                f"error={error!r} count={len(action)} "
                f"nonfinite={len(action) - len(finite_action)} "
                f"max_abs_raw={max(finite_action, default=0.0):.3g} "
                f"obs_nonfinite={len(input_obs) - len(finite_obs)} "
                f"obs_max_abs={max(finite_obs, default=0.0):.3g}",
                end="\r\n",
                flush=True,
            )
            self.policy_faulted = True
            self._hold_safe_targets(obs, command)
            return

        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            lo, hi = self._action_clip[name]
            target = self._default_pose[name] + action[i] * self.action_scale
            command.target_angles[name] = max(lo, min(hi, target))
        # The training observation is the policy's own raw output, not the
        # clipped target.
        self._last_action = action
        self._last_safe_targets = {
            name: command.target_angles[name] for name in OBSERVATION_DOF_ORDER
        }

        # Head stabilization: cancel trunk roll/pitch on the neck. Independent of the RL
        # policy above, so it still runs even though neck_roll/neck_pitch aren't in its
        # action space. Without a VR head command the reference is the HOME attitude
        # (trunk pitched HOME_TRUNK_PITCH_RAD forward): standing at HOME the neck stays
        # at its trained HOME angle 0 (walk training holds it there) and only the gait's
        # sway around HOME is cancelled. With VR teleop active the commanded
        # head_orientation is a world (gravity-levelled) attitude, as in hmd_head.py,
        # so the full trunk tilt is cancelled.
        if obs.robot_state.body_quat:
            head_orientation = obs.user_input.head_orientation
            desired = head_orientation or {"roll": 0.0, "pitch": 0.0}
            roll, pitch = _body_roll_pitch(obs.robot_state.body_quat)
            if not head_orientation:
                pitch -= HOME_TRUNK_PITCH_RAD
            neck_roll = self._default_pose.get("neck_roll", 0.0) + self._neck_stabilize_gain * (desired["roll"] - roll)
            neck_pitch = self._default_pose.get("neck_pitch", 0.0) + self._neck_stabilize_gain * (desired["pitch"] - pitch)
            command.target_angles["neck_roll"] = max(NECK_ROLL_RANGE[0], min(NECK_ROLL_RANGE[1], neck_roll))
            command.target_angles["neck_pitch"] = max(NECK_PITCH_RANGE[0], min(NECK_PITCH_RANGE[1], neck_pitch))

        # Log positions and voltages
        if LOGGING:
            for name in MOTOR_TO_ID:
                self.position[name].append(obs.robot_state.motor_positions[name])
                self.voltage[name].append(obs.robot_state.motor_voltages[name])

    def _hold_safe_targets(self, obs: Observation, command: MotorCommand) -> None:
        """Keep the most recent bounded targets while the actor is faulted."""
        for name in OBSERVATION_DOF_ORDER:
            command.target_angles[name] = self._last_safe_targets.get(
                name, obs.robot_state.motor_positions.get(name, NEUTRAL_POSE[name])
            )
        for name in ("neck_roll", "neck_pitch"):
            command.target_angles[name] = obs.robot_state.motor_positions.get(
                name, NEUTRAL_POSE[name]
            )

    def build_observation(self, obs: Observation) -> list[float]:
        """Build policy observation from robot state."""
        input_obs = []
        
        # IMU data: gyroscope and projected gravity in body frame
        input_obs.extend(obs.robot_state.gyro)
        input_obs.extend(obs.robot_state.projected_gravity)
        
        # Motor positions
        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(obs.robot_state.motor_positions[name] - self._default_pose[name])
        
        # Motor velocities
        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(obs.robot_state.motor_velocities[name])
        
        # Last action
        input_obs.extend(self._last_action)

        # Command
        input_obs.append(obs.user_input.velocity["vx"])
        input_obs.append(obs.user_input.velocity["vy"])
        input_obs.append(obs.user_input.velocity["vtheta"])

        # Reference phase
        if self._use_reference_phase:
            reference_phase = (self._phase_step % self._phase_total_steps) / self._phase_total_steps * 2 * np.pi
            input_obs.append(np.cos(reference_phase))
            input_obs.append(np.sin(reference_phase))

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if "getup" in obs.user_input.active_moves:
            # Get-up is exclusive and owns both commands and gains. Do not finish the
            # normal return later and raise KP halfway through fall recovery.
            self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
            self._phase_step = 0
            self._stop_start_time_s = None
            self._stop_start_angles = {}
            self.state = MoveState.INACTIVE
            return

        # Returning an arbitrary gait pose to neutral in a single 20 ms tick is unsafe,
        # especially if KP is raised on that same tick. Keep the walking gain while the
        # 18 policy joints follow a smoothstep trajectory, then restore the normal gain.
        if self._stop_start_time_s is None:
            self._stop_start_time_s = obs.robot_state.time_s
            self._stop_start_angles = {
                name: obs.robot_state.motor_positions.get(name, NEUTRAL_POSE[name])
                for name in OBSERVATION_DOF_ORDER
            }

        elapsed = max(0.0, obs.robot_state.time_s - self._stop_start_time_s)
        duration = max(1e-6, self._neutral_return_duration_s)
        u = min(1.0, elapsed / duration)
        blend = u * u * (3.0 - 2.0 * u)
        for name in OBSERVATION_DOF_ORDER:
            start = self._stop_start_angles[name]
            neutral = NEUTRAL_POSE[name]
            command.target_angles[name] = start + (neutral - start) * blend

        if u >= 1.0:
            if self._controller is not None:
                ids = [MOTOR_TO_ID[name] for name in OBSERVATION_DOF_ORDER]
                self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
            self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
            self._phase_step = 0
            self._stop_start_time_s = None
            self._stop_start_angles = {}
            self.state = MoveState.INACTIVE

        # Save json logs
        if LOGGING:
            import json
            with open("walk_log.json", "w") as f:
                json.dump({
                    "position": self.position,
                    "voltage": self.voltage,
                }, f, indent=4)

    def on_safety_resume(self, obs: Observation) -> None:
        # If a safety hold interrupted a normal STOPPING transition, its wall-clock
        # interpolation is now stale. Restart it from the newly measured pose. ACTIVE
        # walking will be disarmed by NetworkInputSource and enter this fresh stop path.
        _ = obs
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self.policy_faulted = False
        self._stop_start_time_s = None
        self._stop_start_angles = {}
