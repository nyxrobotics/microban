# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math

import onnxruntime as ort

from constants import KP_DEFAULT, KP_HARDWARE_NEUTRAL, KP_RL, MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState


# Policy name
AGENT_NAME = "getup.onnx"
GETUP_CONTRACT_VERSION = "v3"

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
# Rough sanity ceiling on the actor's raw (pre-clip) output, independent of
# any joint's actual range of motion -- that's already 100% enforced by
# _action_clip below no matter how large the raw value is. This just catches
# a badly-behaving/out-of-distribution model and holds the last good target
# instead of trusting it; the specific number isn't load-bearing for
# range-of-motion safety, just a loose "does this look sane" gate.
_POLICY_MAX_RAW_ACTION = 120.0


def _metadata_floats(value: str | None) -> list[float]:
    if not value:
        return []
    try:
        values = [float(part) for part in value.split(",")]
    except (TypeError, ValueError, OverflowError):
        return []
    return values if all(math.isfinite(item) for item in values) else []


class GetupMove(Move):
    """Fall recovery / anti-thrashing-when-held, via a RL policy trained in simulation.

    The policy's action space is OBSERVATION_DOF_ORDER (18 joints), matching
    walk.onnx's own convention -- the neck is not in it. Held steady at its
    measured position instead (see step()): a getup maneuver can pass through
    trunk orientations far outside what a trunk-relative stabilizer like
    WalkMove's assumes is upright-ish, so this deliberately does not attempt to
    actively drive the neck to anything, just holds it. No velocity command:
    this move doesn't walk anywhere, it only gets the robot upright (or calm,
    if held in the air) and then hands off. See scheduler.py for the
    fall-detection switch to/from WalkMove.
    """

    def __init__(self, controller: ControllerProtocol | None = None) -> None:
        super().__init__()
        self._controller = controller

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        self._ort_session = ort.InferenceSession(
            f"src/agents/{AGENT_NAME}", sess_options=session_options
        )

        meta = self._ort_session.get_modelmeta().custom_metadata_map
        joint_names = meta.get("joint_names", "").split(",")
        default_positions = _metadata_floats(meta.get("default_joint_pos"))
        joints_valid = (
            len(joint_names) == len(MOTOR_TO_ID)
            and set(joint_names) == set(MOTOR_TO_ID)
            and len(default_positions) == len(joint_names)
        )
        if joints_valid:
            self._joint_names = joint_names
            self._default_pose = dict(zip(joint_names, default_positions))
        else:
            # Even a malformed ONNX must leave the neutral fallback available.
            self._joint_names = list(MOTOR_TO_ID)
            self._default_pose = dict(NEUTRAL_POSE)
        action_joint_names = meta.get("action_joint_names", "").split(",")
        observation_names = meta.get("observation_names", "").split(",")
        clip_lower = _metadata_floats(meta.get("action_clip_lower"))
        clip_upper = _metadata_floats(meta.get("action_clip_upper"))
        action_scale = _metadata_floats(meta.get("action_scale"))
        previous_lower = _metadata_floats(
            meta.get("microban_getup_previous_action_lower")
        )
        previous_upper = _metadata_floats(
            meta.get("microban_getup_previous_action_upper")
        )
        action_count = len(OBSERVATION_DOF_ORDER)
        # Per-joint, not a blanket range: training clips each joint's target at
        # its own soft limit (MICROBAN_BODY_JOINT_SOFT_LIMITS), which are far
        # from symmetric or uniform across joints (e.g. right_hip_yaw is
        # -3.927/+0.785 rad). An earlier version of this file instead checked
        # for a stale blanket +-1.57 rad, left over from before the training
        # side moved to a per-joint clip -- this just checks the metadata is
        # well-formed and trusts its actual per-joint values.
        clip_valid = (
            len(clip_lower) == action_count
            and len(clip_upper) == action_count
            and all(math.isfinite(value) for value in clip_lower)
            and all(math.isfinite(value) for value in clip_upper)
            and all(lo < hi for lo, hi in zip(clip_lower, clip_upper))
        )
        scale_valid = len(action_scale) in (1, action_count) and all(
            abs(value - 1.0) <= 0.001 for value in action_scale
        )
        previous_valid = (
            len(previous_lower) == action_count
            and len(previous_upper) == action_count
            and all(
                lo <= clip_lo - self._default_pose[name] + 0.001
                and hi >= clip_hi - self._default_pose[name] - 0.001
                for name, lo, hi, clip_lo, clip_hi in zip(
                    OBSERVATION_DOF_ORDER,
                    previous_lower,
                    previous_upper,
                    clip_lower,
                    clip_upper,
                )
            )
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
            == "applied_target_delta_from_default"
            and joints_valid
            and action_joint_names == OBSERVATION_DOF_ORDER
            and observation_names == [
                "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"
            ]
            and clip_valid
            and scale_valid
            and previous_valid
            and checkpoint_valid
            and len(inputs) == 1
            and inputs[0].shape == [1, 60]
            and len(outputs) == 1
            and outputs[0].shape == [1, action_count]
        )
        if not self.model_ready:
            print(
                "Get-up actor disabled: deployed model lacks the v3 action or "
                "IMU-frame contract; falls return toward neutral",
                end="\r\n", flush=True,
            )
        self._neck_joint_names = [
            name for name in self._joint_names if name not in OBSERVATION_DOF_ORDER
        ]

        self.action_scale = 1.0
        # Per-joint, from the model's own action_clip_lower/upper metadata
        # (validated above as clip_valid): matches the training-time
        # JointPositionActionCfg.clip (microban_getup_env_cfg.py), which is
        # each joint's own soft limit, not a blanket range. The policy's raw
        # output relied on the env clamping it to this same per-joint range
        # before becoming a target, so deploying with anything else -- a
        # blanket range, or no clip -- lets occasional out-of-range outputs
        # reach the motors as much larger, wrong targets. Falls back to a
        # loose, safe blanket range if the metadata is malformed (model_ready
        # is already False in that case, so this fallback should never
        # actually reach the motors -- see the neutral-fallback comment above).
        if clip_valid:
            self._action_clip = dict(zip(OBSERVATION_DOF_ORDER, zip(clip_lower, clip_upper)))
        else:
            self._action_clip = {name: (-1.57, 1.57) for name in OBSERVATION_DOF_ORDER}
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
        # Force step()'s torque-enable branch to run fresh on this
        # (re)activation instead of trusting a previous activation's state.
        self._last_torque_enabled = None
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
        if len(action) != len(OBSERVATION_DOF_ORDER) or any(
            not math.isfinite(value) or abs(value) > _POLICY_MAX_RAW_ACTION
            for value in action_values
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
        self._last_action = []
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            lo, hi = self._action_clip[name]
            target = max(lo, min(hi, self._default_pose[name] + action_values[i] * self.action_scale))
            self._policy_targets[name] = target
            # The v3 training observation is the actual applied (clipped)
            # target -- no per-tick rate limit on the active policy.
            self._last_action.append((target - self._default_pose[name]) / self.action_scale)
            command.target_angles[name] = target

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
            gains = [
                KP_RL if walking and name in OBSERVATION_DOF_ORDER else KP_DEFAULT
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
