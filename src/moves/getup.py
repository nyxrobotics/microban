# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import onnxruntime as ort

from constants import KP_DEFAULT, KP_RL, MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from observer import Observation
from moves.move import MotorCommand, Move, MoveState


# Policy name
AGENT_NAME = "getup.onnx"

# "Slowly" for the torque-on-but-unarmed recovery slew (see step()): far
# below hmd_head.py's 2.5 rad/s, since this is a deliberately gentle return
# to neutral after a limp/fallen state, not a tracking response.
_RECOVERY_SLEW_RATE_RAD_S = 0.5
_RECOVERY_MIN_DT_S = 0.001
_RECOVERY_MAX_DT_S = 0.1


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

        self._ort_session = ort.InferenceSession(f"src/agents/{AGENT_NAME}")

        meta = self._ort_session.get_modelmeta().custom_metadata_map
        self._joint_names: list[str] = meta["joint_names"].split(",")
        positions = [float(v) for v in meta["default_joint_pos"].split(",")]
        self._default_pose: dict[str, float] = dict(zip(self._joint_names, positions))
        self._neck_joint_names = [
            name for name in self._joint_names if name not in OBSERVATION_DOF_ORDER
        ]

        self.action_scale = 1.0
        # Matches the training-time JointPositionActionCfg.clip (microban_getup_env_cfg.py):
        # the policy's raw output relied on the env clamping it to this range before
        # becoming a target, so deploying without the same clip lets occasional
        # out-of-range outputs reach the motors as much larger, wrong targets.
        self._action_clip = (-1.57, 1.57)
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)

        # Real-hardware test gating (GamepadInputSource B/A/R3; see step()).
        # None so the very first tick always runs the torque-enable branch
        # below rather than assuming a prior state.
        self._last_torque_enabled: bool | None = None
        self._recovery_targets: dict[str, float] | None = None
        self._recovery_last_time_s: float | None = None

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = [MOTOR_TO_ID[name] for name in self._joint_names]
            gains = [
                KP_RL if name in OBSERVATION_DOF_ORDER else KP_DEFAULT
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

        if not getup_armed:
            self._step_recover_to_neutral(obs, command)
            self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
            return

        input_obs = self.build_observation(obs)
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        self._last_action = action.tolist()

        lo, hi = self._action_clip
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            target = self._default_pose[name] + action[i] * self.action_scale
            command.target_angles[name] = max(lo, min(hi, target))

        # Not in the policy's action space (see class docstring): hold steady
        # at the continuously-measured position rather than a stale snapshot,
        # so a soft/compliant gain here does not accumulate drift.
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
        if self.state != MoveState.INACTIVE:
            self.state = MoveState.STARTING
