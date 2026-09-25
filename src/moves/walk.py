# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import math

import numpy as np
import onnxruntime as ort

from constants import (
    KP_DEFAULT,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
)
from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from observer import Observation

# Set to True to log motor positions and voltages during the walk move
# Note: requires to set observe_voltage = True in the Observer to log voltages
LOGGING = False

# Policy name
AGENT_NAME = "walk.onnx"

# Neck roll/pitch joint ranges (rad), from src/model/mjcf/robot.xml. The stabilization
# below clips to these so a large trunk tilt can't request an out-of-range neck target.
NECK_ROLL_RANGE = (-0.436332, 0.436332)
NECK_PITCH_RANGE = (-1.570796, 0.436332)


def _body_roll_pitch(body_quat: list[float]) -> tuple[float, float]:
    """Roll (about +X) and pitch (about +Y) of the trunk frame, same convention as
    scheduler.py's IMU display and as the neck_roll/neck_pitch joint axes (robot.xml)."""
    w, x, y, z = body_quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


class WalkMove(Move):
    """Walk using a RL policy trained in simulation."""

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        neutral_return_duration_s: float = 0.8,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)

        # Load ONNX policy
        self._ort_session = ort.InferenceSession(f"src/agents/{AGENT_NAME}")

        self.action_scale = 1.0

        # Head stabilization: counter-rotate neck_roll/neck_pitch against trunk tilt so the
        # head stays level while walking. Not part of the RL policy (neck is excluded from
        # its action/observation space) — this is a separate proportional control law layered
        # on top; gain=1.0 is full cancellation of the extracted roll/pitch.
        self._neck_stabilize_gain = 1.0
        self._neutral_return_duration_s = neutral_return_duration_s
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}

        # Reference pose: read from ONNX metadata
        meta = self._ort_session.get_modelmeta().custom_metadata_map
        names = meta["joint_names"].split(",")
        positions = [float(v) for v in meta["default_joint_pos"].split(",")]
        self._default_pose: dict[str, float] = dict(zip(names, positions))

        # Detect reference phase from model input size:
        # base_obs = gyro(3) + proj_grav(3) + pos(N) + vel(N) + action(N) + cmd(3)
        # phase_obs = base_obs + phase(2)
        base_obs_size = 3 + 3 + 3 * len(OBSERVATION_DOF_ORDER) + 3
        self._use_reference_phase: bool = self._ort_session.get_inputs()[0].shape[1] > base_obs_size
        self._phase_step = 0
        self._phase_total_steps = 20

        # Safety parameters
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
        if self._controller is not None:
            ids = [MOTOR_TO_ID[name] for name in OBSERVATION_DOF_ORDER]
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        # The scheduler's base command is neutral. Hold the measured gait joints on
        # the transition tick so enabling the policy cannot create a one-frame jump.
        for name in OBSERVATION_DOF_ORDER:
            command.target_angles[name] = obs.robot_state.motor_positions.get(
                name, NEUTRAL_POSE[name]
            )
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self.state = MoveState.ACTIVE

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
            return
        
        # Run policy
        input_obs = self.build_observation(obs)
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        self._last_action = action.tolist()

        # Update command
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            command.target_angles[name] = self._default_pose[name] + action[i] * self.action_scale

        # Head stabilization: hold the neck level (or, with VR teleop active, at the
        # commanded head_orientation) against trunk roll/pitch. Independent of the RL policy
        # above, so it still runs even though neck_roll/neck_pitch aren't in its action space.
        if obs.robot_state.body_quat:
            desired = obs.user_input.head_orientation or {"roll": 0.0, "pitch": 0.0}
            roll, pitch = _body_roll_pitch(obs.robot_state.body_quat)
            neck_roll = self._default_pose.get("neck_roll", 0.0) + self._neck_stabilize_gain * (desired["roll"] - roll)
            neck_pitch = self._default_pose.get("neck_pitch", 0.0) + self._neck_stabilize_gain * (desired["pitch"] - pitch)
            command.target_angles["neck_roll"] = max(NECK_ROLL_RANGE[0], min(NECK_ROLL_RANGE[1], neck_roll))
            command.target_angles["neck_pitch"] = max(NECK_PITCH_RANGE[0], min(NECK_PITCH_RANGE[1], neck_pitch))

        # Log positions and voltages
        if LOGGING:
            for name in MOTOR_TO_ID:
                self.position[name].append(obs.robot_state.motor_positions[name])
                self.voltage[name].append(obs.robot_state.motor_voltages[name])

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
            command.target_angles[name] = start + (NEUTRAL_POSE[name] - start) * blend

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
        self._stop_start_time_s = None
        self._stop_start_angles = {}
