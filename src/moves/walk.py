# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math
from pathlib import Path
from typing import Any

from constants import (
    HOME_TRUNK_PITCH_RAD,
    KP_HARDWARE_NEUTRAL,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    POLICY_ACTION_SCALE,
    SERVO_TARGET_RANGE_RAD,
)
from controller import ControllerProtocol
from imu_reader import trunk_roll_pitch
from moves.move import MotorCommand, Move, MoveState
from observer import Observation
from policy_contract import load_installed_policy, parse_policy

# Set to True to log motor positions and voltages during the walk move
# Note: requires to set observe_voltage = True in the Observer to log voltages
LOGGING = False

# Policy name of the walking model (the PICO policy is trained on it as its
# frozen walker).  Its contract (src/policy_contract.py):
# target = clip(NEUTRAL_POSE + raw * 1.0, -pi, +pi) on the 18
# OBSERVATION_DOF_ORDER joints, the raw output fed back as the previous action,
# trained at this robot's HOME (config/home_pose.yaml).
AGENT_NAME = "walk.onnx"
ALL_MOTOR_IDS = list(MOTOR_TO_ID.values())

# Neck roll/pitch joint ranges (rad), from src/model/mjcf/robot.xml. The stabilization
# below clips to these so a large trunk tilt can't request an out-of-range neck target.
NECK_ROLL_RANGE = (-0.436332, 0.436332)
NECK_PITCH_RANGE = (-1.570796, 0.436332)


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

        # Fail closed: an actor trained at another HOME, with another action
        # rule, or failing its startup self-test never reaches the motors. An
        # injected session (tests) is checked but not self-tested.
        if session is None:
            path = policy_path if policy_path is not None else Path("src/agents") / AGENT_NAME
            loaded = load_installed_policy("walk", path)
            session, self.contract = loaded.session, loaded.contract
        else:
            self.contract = parse_policy("walk", session)
        self._ort_session = session
        # Validated equal to NEUTRAL_POSE / the servo range; use the robot's
        # full-precision copies.
        self._default_pose = {name: float(NEUTRAL_POSE[name]) for name in NEUTRAL_POSE}
        self._action_clip = {
            name: (-SERVO_TARGET_RANGE_RAD, SERVO_TARGET_RANGE_RAD)
            for name in OBSERVATION_DOF_ORDER
        }
        self.action_scale = POLICY_ACTION_SCALE

        # Head stabilization: counter-rotate neck_roll/neck_pitch against trunk tilt so the
        # head stays level while walking. Not part of the RL policy (neck is excluded from
        # its action/observation space) — this is a separate proportional control law layered
        # on top; gain=1.0 is full cancellation of the extracted roll/pitch.
        self._neck_stabilize_gain = 1.0
        self._neutral_return_duration_s = neutral_return_duration_s
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}

        # Safety parameters. A fall is a physical attitude, so this stays measured from
        # vertical (trunk tilt > 60 deg) at every HOME, like the scheduler's fall
        # debounce; the forward-lean HOME's 10 deg lean still leaves 50 deg before it trips.
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
        # one control cycle. Never carry action recurrence from an older walk
        # activation into that emergency handoff.
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self.policy_faulted = False
        if self._controller is not None:
            # A learned policy runs every joint, head and neck included, at the
            # gain it was trained with.
            self._controller.sync_write_kp(ALL_MOTOR_IDS, [KP_RL] * len(ALL_MOTOR_IDS))
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
        # (trunk pitched HOME_TRUNK_PITCH_RAD forward; 0 for a vertical-trunk HOME):
        # standing at HOME the neck stays at its trained HOME angle (walk training holds
        # it there) and only the gait's sway around HOME is cancelled. With VR teleop
        # active the commanded head_orientation is a world (gravity-levelled) attitude,
        # as in hmd_head.py, so the full trunk tilt is cancelled.
        trunk_angles = trunk_roll_pitch(obs.robot_state.body_quat)
        if trunk_angles is not None:
            head_orientation = obs.user_input.head_orientation
            desired = head_orientation or {"roll": 0.0, "pitch": 0.0}
            roll, pitch = trunk_angles
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

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if "getup" in obs.user_input.active_moves:
            # Get-up is exclusive and owns both commands and gains. Do not finish the
            # normal return later and raise KP halfway through fall recovery.
            self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
            self._stop_start_time_s = None
            self._stop_start_angles = {}
            self.state = MoveState.INACTIVE
            return

        # Returning an arbitrary gait pose to neutral in a single 20 ms tick is unsafe,
        # especially if KP is raised on that same tick. Keep the walking gain while the
        # 18 policy joints follow a smoothstep trajectory, then raise the holding gain.
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
                # Back at HOME with no learned policy: the static holding gain.
                self._controller.sync_write_kp(
                    ALL_MOTOR_IDS, [KP_HARDWARE_NEUTRAL] * len(ALL_MOTOR_IDS)
                )
            self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
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
