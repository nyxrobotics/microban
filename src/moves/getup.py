# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import json
import math
from pathlib import Path
from typing import Any

import onnxruntime as ort

from constants import (
    HOME_ROOT_POS_Z_M,
    HOME_ROOT_QUAT_WXYZ,
    KP_DEFAULT,
    KP_HARDWARE_NEUTRAL,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    POLICY_TARGET_CLIP_RAD,
)
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState


# Policy name
AGENT_NAME = "getup.onnx"
# v4: absolute target = default + raw action, clipped at the model's own
# action_clip_lower/upper (a flat +-1.57 rad in training), and the policy
# observes its own previous RAW output. Every get-up policy that stood in
# simulation was trained this way, as was the one this robot ran on
# 2026-09-26 (runtime 9b36403 fed back the raw output too). v3 fed back the
# applied target instead.
GETUP_CONTRACT_VERSION = "v4"
GETUP_PREVIOUS_ACTION_SEMANTICS = "raw_policy_output"
# The model's HOME must be the robot's NEUTRAL_POSE (the one centered HOME
# every policy shares). microban_getup_home_pose carries it at full precision;
# default_joint_pos is serialized with 3 decimals by mjlab's exporter.
_HOME_POSE_TOLERANCE_RAD = 1.0e-6
_HOME_ROOT_TOLERANCE = 1.0e-6
_SERIALIZED_DEFAULT_TOLERANCE_RAD = 0.0005 + 1.0e-9

# "Slowly" for the torque-on-but-unarmed recovery slew (see
# _step_recover_to_neutral): far below hmd_head.py's 2.5 rad/s, since this is
# a deliberately gentle return to neutral after a limp/fallen state, not a
# tracking response. This is UNRELATED to the active get-up policy's own
# commanded target (see step()): once armed, the policy's clipped target is
# written directly, every tick, with no rate limit at all. An earlier version
# of this file mistakenly reused this same 0.5 rad/s figure for that too
# (a removed _POLICY_MAX_TARGET_SPEED_RAD_S constant), which made the actual
# recovery motion look passive/unable to stand rather than merely imperfect.
_RECOVERY_SLEW_RATE_RAD_S = 0.5
_RECOVERY_MIN_DT_S = 0.001
_RECOVERY_MAX_DT_S = 0.1


def _metadata_floats(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        values = [float(part) for part in value.split(",")]
    except (TypeError, ValueError, OverflowError):
        return []
    return values if all(math.isfinite(item) for item in values) else []


def _home_pose_matches_neutral(value: str | None) -> bool:
    """True when the model's full-precision training HOME is NEUTRAL_POSE."""
    if not value:
        return False
    try:
        home = json.loads(value)
        joints = {str(name): float(angle) for name, angle in home["joint_pos_rad"].items()}
        root_pos = [float(item) for item in home["root_pos_m"]]
        root_quat = [float(item) for item in home["root_quat_wxyz"]]
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
        return False
    return (
        set(joints) == set(NEUTRAL_POSE)
        and all(
            math.isfinite(angle)
            and abs(angle - NEUTRAL_POSE[name]) <= _HOME_POSE_TOLERANCE_RAD
            for name, angle in joints.items()
        )
        and len(root_pos) == 3
        and abs(root_pos[2] - HOME_ROOT_POS_Z_M) <= _HOME_ROOT_TOLERANCE
        and len(root_quat) == 4
        and all(
            abs(actual - expected) <= _HOME_ROOT_TOLERANCE
            for actual, expected in zip(root_quat, HOME_ROOT_QUAT_WXYZ)
        )
    )


class GetupMove(Move):
    """Fall recovery / anti-thrashing-when-held, via a RL policy trained in simulation.

    The policy's action space is OBSERVATION_DOF_ORDER (18 joints), matching
    walk.onnx's own convention -- the neck is not in it. Held steady at its
    measured position instead (see step()): a getup maneuver can pass through
    trunk orientations far outside what a trunk-relative stabilizer like
    WalkMove's assumes is upright-ish, so this deliberately does not attempt to
    actively drive the neck to anything, just holds it. No velocity command:
    this move doesn't walk anywhere, it only gets the robot upright (or calm,
    if held in the air) and then hands off -- or, when the requested walk move
    cannot balance (the PICO static hold), keeps standing in place as the
    balancer. See scheduler.py for the fall-detection switch to/from WalkMove.
    """

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        *,
        policy_path: str | Path | None = None,
        session: Any | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller

        if session is None:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 1
            session_options.inter_op_num_threads = 1
            path = policy_path if policy_path is not None else Path("src/agents") / AGENT_NAME
            session = ort.InferenceSession(str(path), sess_options=session_options)
        self._ort_session = session

        meta = self._ort_session.get_modelmeta().custom_metadata_map
        joint_names = meta.get("joint_names", "").split(",")
        default_positions = _metadata_floats(meta.get("default_joint_pos"))
        joints_valid = (
            len(joint_names) == len(MOTOR_TO_ID)
            and set(joint_names) == set(MOTOR_TO_ID)
            and len(default_positions) == len(joint_names)
        )
        # The model must have been trained at the robot's one shared HOME:
        # both its full-precision HOME stamp and its (3-decimal) default_joint_pos
        # must reproduce NEUTRAL_POSE. A model trained at an older HOME is
        # rejected (model_ready False) rather than offset-corrected.
        home_valid = (
            joints_valid
            and _home_pose_matches_neutral(meta.get("microban_getup_home_pose"))
            and all(
                abs(value - NEUTRAL_POSE[name]) <= _SERIALIZED_DEFAULT_TOLERANCE_RAD
                for name, value in zip(joint_names, default_positions)
            )
        )
        self._joint_names = joint_names if joints_valid else list(MOTOR_TO_ID)
        # Validated equal to NEUTRAL_POSE above; inference uses the robot's
        # full-precision copy rather than the rounded metadata.
        self._default_pose = dict(NEUTRAL_POSE)
        action_joint_names = meta.get("action_joint_names", "").split(",")
        observation_names = meta.get("observation_names", "").split(",")
        clip_lower = _metadata_floats(meta.get("action_clip_lower"))
        clip_upper = _metadata_floats(meta.get("action_clip_upper"))
        action_scale = _metadata_floats(meta.get("action_scale"))
        action_count = len(OBSERVATION_DOF_ORDER)
        # Well-formedness only: the per-joint values come from training's own
        # JointPositionActionCfg.clip, whatever it is for this model.
        clip_valid = (
            len(clip_lower) == action_count
            and len(clip_upper) == action_count
            and all(math.isfinite(value) for value in clip_lower)
            and all(math.isfinite(value) for value in clip_upper)
            and all(lo < hi for lo, hi in zip(clip_lower, clip_upper))
            # Every policy shares the +-POLICY_TARGET_CLIP_RAD absolute clip;
            # never let metadata widen it.
            and all(
                lo >= -POLICY_TARGET_CLIP_RAD - 1.0e-6
                and hi <= POLICY_TARGET_CLIP_RAD + 1.0e-6
                for lo, hi in zip(clip_lower, clip_upper)
            )
        )
        scale_valid = len(action_scale) in (1, action_count) and all(
            abs(value - 1.0) <= 0.001 for value in action_scale
        )
        checkpoint_sha256 = meta.get("checkpoint_sha256", "")
        checkpoint_valid = len(checkpoint_sha256) == 64 and all(
            character in "0123456789abcdef" for character in checkpoint_sha256
        )
        inputs = self._ort_session.get_inputs()
        outputs = self._ort_session.get_outputs()
        self.model_ready = (
            meta.get("microban_getup_contract") == GETUP_CONTRACT_VERSION
            and meta.get("microban_getup_angular_velocity_frame") == "imu_sensor_xyz"
            and meta.get("microban_getup_previous_action_semantics")
            == GETUP_PREVIOUS_ACTION_SEMANTICS
            and joints_valid
            and home_valid
            and action_joint_names == OBSERVATION_DOF_ORDER
            and observation_names == [
                "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"
            ]
            and clip_valid
            and scale_valid
            and checkpoint_valid
            and len(inputs) == 1
            and inputs[0].shape == [1, 60]
            and len(outputs) == 1
            and outputs[0].shape == [1, action_count]
        )
        if not self.model_ready:
            print(
                "Get-up actor disabled: deployed model lacks the v4 action, "
                "IMU-frame or centered-HOME contract; falls return toward neutral",
                end="\r\n", flush=True,
            )
        self._neck_joint_names = [
            name for name in self._joint_names if name not in OBSERVATION_DOF_ORDER
        ]

        self.action_scale = 1.0
        # From the model's own action_clip_lower/upper metadata (validated
        # above as clip_valid), i.e. training's JointPositionActionCfg.clip.
        # Falls back to +-1.57 rad if the metadata is malformed (model_ready
        # is already False then, so this never reaches the motors).
        if clip_valid:
            self._action_clip = dict(zip(OBSERVATION_DOF_ORDER, zip(clip_lower, clip_upper)))
        else:
            self._action_clip = {
                name: (-POLICY_TARGET_CLIP_RAD, POLICY_TARGET_CLIP_RAD)
                for name in OBSERVATION_DOF_ORDER
            }
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)

        # Real-hardware test gating (GamepadInputSource B/A/R3; see step()).
        # None so the very first tick always runs the torque-enable branch
        # below rather than assuming a prior state.
        self._last_torque_enabled: bool | None = None
        self._recovery_targets: dict[str, float] | None = None
        self._recovery_last_time_s: float | None = None
        self._policy_targets: dict[str, float] | None = None
        self.policy_faulted = False

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = [MOTOR_TO_ID[name] for name in self._joint_names]
            if not self.model_ready:
                # An operator can select get-up manually while an older actor
                # is installed. Seed the measured goals before raising P gain
                # for the neutral fallback, as the global A gate does.
                self._controller.sync_write_goal_position(
                    ids,
                    [
                        obs.robot_state.motor_positions.get(name, NEUTRAL_POSE[name])
                        for name in self._joint_names
                    ],
                )
            gains = [
                KP_HARDWARE_NEUTRAL if not self.model_ready
                else KP_RL if name in OBSERVATION_DOF_ORDER else KP_DEFAULT
                for name in self._joint_names
            ]
            self._controller.sync_write_kp(ids, gains)
        for name in self._joint_names:
            command.target_angles[name] = obs.robot_state.motor_positions.get(
                name, NEUTRAL_POSE[name]
            )
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        # Torque is already on whenever the scheduler starts this move; a
        # redundant all-joint ON write runs RobotController's blocking verify
        # (up to TORQUE_VERIFY_TIMEOUT_S = 1 s) on the first get-up tick.
        self._last_torque_enabled = (
            True if obs.user_input.torque_enabled is True else None
        )
        self._recovery_targets = None
        self._policy_targets = None
        self.policy_faulted = False
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        torque_enabled = obs.user_input.torque_enabled
        getup_armed = obs.user_input.getup_armed

        if torque_enabled != self._last_torque_enabled:
            if self._controller is not None:
                ids = [MOTOR_TO_ID[name] for name in self._joint_names]
                self._controller.sync_write_torque_enable(
                    ids, [torque_enabled] * len(ids)
                )
            self._last_torque_enabled = torque_enabled
            if torque_enabled:
                # (Re)start the slew-to-neutral from wherever the robot
                # actually is right now, e.g. wherever it settled while limp.
                self._recovery_targets = {
                    name: obs.robot_state.motor_positions.get(
                        name, NEUTRAL_POSE[name]
                    )
                    for name in self._joint_names
                }
                self._recovery_last_time_s = obs.robot_state.time_s

        if not torque_enabled:
            # LIMP: torque is physically off, nothing to command.
            return

        if not getup_armed or not self.model_ready:
            self._step_recover_to_neutral(obs, command)
            self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
            self._policy_targets = None
            return

        if self.policy_faulted:
            self._hold_policy_targets(obs, command)
            return

        input_obs = self.build_observation(obs)
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        action_values = [float(value) for value in action]
        # Only non-finite output is a fault. A v4 policy's raw output is
        # routinely hundreds of radians: it drives the clipped target to a
        # bound to get torque out of the soft P125 servos, and the clip
        # below bounds the physical target no matter how large it is.
        if len(action) != len(OBSERVATION_DOF_ORDER) or any(
            not math.isfinite(value) for value in action_values
        ):
            if not self.policy_faulted:
                finite_action = [abs(value) for value in action_values if math.isfinite(value)]
                finite_obs = [abs(value) for value in input_obs if math.isfinite(value)]
                obs_group_max = [
                    max(
                        (abs(value) for value in input_obs[start:end] if math.isfinite(value)),
                        default=0.0,
                    )
                    for start, end in ((0, 3), (3, 6), (6, 24), (24, 42), (42, 60))
                ]
                print(
                    "Get-up actor output invalid: "
                    f"count={len(action_values)} "
                    f"nonfinite={len(action_values) - len(finite_action)} "
                    f"max_abs_raw={max(finite_action, default=0.0):.3g} "
                    f"obs_nonfinite={len(input_obs) - len(finite_obs)} "
                    f"obs_max_abs={max(finite_obs, default=0.0):.3g} "
                    "obs_group_max=(gyro,gravity,pos,vel,last_action)="
                    f"{tuple(round(value, 3) for value in obs_group_max)}",
                    end="\r\n",
                    flush=True,
                )
            self.policy_faulted = True
            self._hold_policy_targets(obs, command)
            return
        self._policy_targets = {}
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            lo, hi = self._action_clip[name]
            target = max(lo, min(hi, self._default_pose[name] + action_values[i] * self.action_scale))
            self._policy_targets[name] = target
            command.target_angles[name] = target
        # The v4 training observation is the policy's own raw output, not
        # the clipped target -- and no per-tick rate limit on either.
        self._last_action = action_values

        # Not in the policy's action space (see class docstring): hold steady
        # at the continuously-measured position rather than a stale snapshot,
        # so a soft/compliant gain here does not accumulate drift.
        for name in self._neck_joint_names:
            command.target_angles[name] = obs.robot_state.motor_positions.get(
                name, NEUTRAL_POSE[name]
            )

    def _hold_policy_targets(self, obs: Observation, command: MotorCommand) -> None:
        """Keep the most recent bounded target when the actor output is unusable."""
        for name in OBSERVATION_DOF_ORDER:
            command.target_angles[name] = (
                self._policy_targets[name]
                if self._policy_targets is not None
                else obs.robot_state.motor_positions[name]
            )
        for name in self._neck_joint_names:
            command.target_angles[name] = obs.robot_state.motor_positions.get(
                name, NEUTRAL_POSE[name]
            )

    def _step_recover_to_neutral(
        self, obs: Observation, command: MotorCommand
    ) -> None:
        """Torque on, policy withheld: slew every joint gently toward
        NEUTRAL_POSE (see _RECOVERY_SLEW_RATE_RAD_S), the safe stop between
        limp and letting the policy actually drive (see step())."""
        if self._recovery_targets is None:
            self._recovery_targets = {
                name: obs.robot_state.motor_positions.get(name, NEUTRAL_POSE[name])
                for name in self._joint_names
            }
        now = obs.robot_state.time_s
        previous = (
            self._recovery_last_time_s
            if self._recovery_last_time_s is not None
            else now
        )
        dt = max(_RECOVERY_MIN_DT_S, min(_RECOVERY_MAX_DT_S, now - previous))
        self._recovery_last_time_s = now
        max_step = _RECOVERY_SLEW_RATE_RAD_S * dt

        for name in self._joint_names:
            current = self._recovery_targets[name]
            target = NEUTRAL_POSE[name]
            delta = max(-max_step, min(max_step, target - current))
            updated = current + delta
            self._recovery_targets[name] = updated
            command.target_angles[name] = updated

    def build_observation(self, obs: Observation) -> list[float]:
        """Build policy observation from robot state. Order fixed by the ONNX's own
        observation_names metadata: base_ang_vel, projected_gravity, joint_pos,
        joint_vel, actions -- no foot_contact, this hardware has no such sensor."""
        input_obs = []

        # Mjlab's base_ang_vel actor term reads robot/imu_ang_vel directly;
        # MuJoCo's gyro reports the IMU site's local axes. The BMI088 driver
        # likewise reports sensor-frame axes, so no mount rotation belongs here.
        input_obs.extend(obs.robot_state.gyro)
        input_obs.extend(obs.robot_state.projected_gravity)

        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(obs.robot_state.motor_positions[name] - self._default_pose[name])

        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(obs.robot_state.motor_velocities[name])

        input_obs.extend(self._last_action)

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        # Hold the measured pose on the hand-off tick. When walking is requested on
        # that same tick, leave its 18 policy axes at the learned-policy gain and put
        # the three camera joints back at the normal gain.
        for name in self._joint_names:
            command.target_angles[name] = obs.robot_state.motor_positions.get(
                name, NEUTRAL_POSE[name]
            )
        if self._controller is not None:
            ids = [MOTOR_TO_ID[name] for name in self._joint_names]
            walking = "walk" in obs.user_input.active_moves
            # No locomotion owner yet (Xbox X off, PICO LT still held): the
            # measured stance is then a static hold; P400 does not hold it.
            gains = [
                (KP_RL if walking else KP_HARDWARE_NEUTRAL)
                if name in OBSERVATION_DOF_ORDER else KP_DEFAULT
                for name in self._joint_names
            ]
            self._controller.sync_write_kp(ids, gains)
        self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        # A feed-forward get-up policy observes its previous action. Do not reuse an
        # action from before an IMU outage; restart through on_start's measured hold.
        _ = obs
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self._last_torque_enabled = None
        self._recovery_targets = None
        self._policy_targets = None
        self.policy_faulted = False
        if self.state != MoveState.INACTIVE:
            self.state = MoveState.STARTING
