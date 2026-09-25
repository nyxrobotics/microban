import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

from constants import (
    IMU_MOUNT_QUAT,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
)
from imu_reader import imu_quat_to_body
from input.input_source import UserInput
from moves.move import MotorCommand, MoveState
from moves.pico_hybrid import (
    EXPECTED_ACCEPTANCE_EVALUATOR_REVISION,
    EXPECTED_ACCEPTANCE_REVISION,
    EXPECTED_ACTION_DEFAULT_JOINT_POS,
    EXPECTED_ACTION_DISTRIBUTION_SEMANTICS,
    EXPECTED_ACTION_SCALE,
    EXPECTED_ACTOR_DEFAULT_INTERIOR_EPSILON_RAD,
    EXPECTED_ACTOR_INITIALIZATION,
    EXPECTED_ACTOR_LATENT_MAX_STD,
    EXPECTED_ACTOR_LATENT_MEAN_FRACTION,
    EXPECTED_ACTOR_LATENT_MEAN_LOWER,
    EXPECTED_ACTOR_LATENT_MEAN_UPPER,
    EXPECTED_ACTOR_LATENT_MIN_STD,
    EXPECTED_ACTOR_LATENT_OPERATIONAL_ABS_MAX,
    EXPECTED_ACTOR_LATENT_OPERATIONAL_LOWER,
    EXPECTED_ACTOR_LATENT_OPERATIONAL_SCALE_MULTIPLIER,
    EXPECTED_ACTOR_LATENT_OPERATIONAL_UPPER,
    EXPECTED_ACTOR_LATENT_STD_ABS_MAX,
    EXPECTED_ACTOR_LATENT_STD_ENVELOPE_DIVISOR,
    EXPECTED_ACTOR_LATENT_STD_MIN_ABS_MAX,
    EXPECTED_ACTOR_LATENT_STD_MIN_ENVELOPE_DIVISOR,
    EXPECTED_ACTOR_RAW_ACTION_LOWER,
    EXPECTED_ACTOR_RAW_ACTION_UPPER,
    EXPECTED_ACTOR_TARGET_GUARD_MARGIN_RATIO,
    EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER,
    EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER,
    EXPECTED_MIGRATION_SOURCE_CHECKPOINT_ITERATION,
    EXPECTED_MIGRATION_SOURCE_CHECKPOINT_SHA256,
    EXPECTED_MIGRATION_SOURCE_GATE_SHA256,
    EXPECTED_MIGRATION_SOURCE_OPTIMIZER_LEARNING_RATE,
    EXPECTED_MIGRATION_SOURCE_TRAINING_PROVENANCE_SHA256,
    EXPECTED_MIGRATION_SOURCE_TREE_SHA256,
    EXPECTED_MIGRATION_STATE_TRANSFER,
    EXPECTED_OBSERVATION_TERMS,
    EXPECTED_PREVIOUS_ACTION_SEMANTICS,
    EXPECTED_RAW_ACTION_SOFT_LOWER,
    EXPECTED_RAW_ACTION_SOFT_UPPER,
    EXPECTED_RECIPE_REVISION,
    EXPECTED_SAFE_VELOCITY_ACCEPTANCE_GATE,
    EXPECTED_SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SHA256,
    EXPECTED_SAFE_VELOCITY_BOOTSTRAP_MAPPING_VERSION,
    EXPECTED_SAFE_VELOCITY_RECEIPT_SCHEMA_VERSION,
    EXPECTED_SAFE_VELOCITY_RECIPE_REVISION,
    EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_ITERATION,
    EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_SHA256,
    EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_LOWER,
    EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_SEMANTICS,
    EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_UPPER,
    EXPECTED_SOFT_JOINT_POS_LOWER,
    EXPECTED_SOFT_JOINT_POS_UPPER,
    EXPECTED_TRAINING_CONTRACT_VERSION,
    EXPECTED_TRAINING_FIXED_LEARNING_RATE,
    EXPECTED_TRAINING_PROVENANCE_MODE,
    EXPECTED_TRAINING_PROVENANCE_SCHEMA_VERSION,
    EXPECTED_V12_ACTION_DISTRIBUTION_SEMANTICS,
    EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    EXPECTED_V12_BOOTSTRAP_MAPPING_VERSION,
    EXPECTED_V12_COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
    EXPECTED_V12_DEADLINE_FINAL_TRACKING_PROFILE,
    EXPECTED_V12_EXTRA_OBSERVATION_COLUMNS,
    EXPECTED_V12_HAND_TARGET_FK,
    EXPECTED_V12_LEGACY_PROBE_SHA256,
    EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_ITERATION,
    EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256,
    EXPECTED_V12_NORMALIZER_SEMANTICS,
    EXPECTED_V12_OBSERVATION_JOINT_NAMES,
    EXPECTED_V12_RAW_ACTION_ENVELOPE_SCHEMA_VERSION,
    EXPECTED_V12_RAW_ACTION_GUARD_FORMULA,
    EXPECTED_V12_RAW_ACTION_GUARD_MULTIPLIER,
    EXPECTED_V12_RAW_ACTION_GUARD_SEMANTICS,
    EXPECTED_V12_RECIPE_REVISION,
    EXPECTED_V12_RUNTIME_ACTION_SEMANTICS,
    EXPECTED_V12_SOURCE_TO_TARGET_COLUMNS,
    EXPECTED_V12_TRAINING_CONTRACT_VERSION,
    PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS,
    PICO_TELEOP_HOME_POSE,
    PicoHybridMove,
    PicoHybridPolicyContractError,
    PicoHybridPolicyRuntimeError,
    onnxruntime_compatibility_smoke_inputs,
    runtime_source_identity,
    sensor_gyro_to_body,
    validate_onnxruntime_compatibility,
    validate_v12_onnxruntime_compatibility,
)
from observer import Observation, RobotState

OBSERVATION_JOINTS = (
    "head",
    "neck_roll",
    "neck_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
)
CURRENT_RUNTIME_SOURCE_IDENTITY = runtime_source_identity()


class _Io:
    def __init__(self, name, shape, tensor_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _Metadata:
    def __init__(self, values):
        self.custom_metadata_map = values


def _csv(values):
    return ",".join(str(value) for value in values)


def _metadata_csv(values):
    return ",".join(f"{value:.3f}" for value in values)


def valid_metadata():
    observation_defaults = [PICO_TELEOP_HOME_POSE[name] for name in OBSERVATION_JOINTS]
    return {
        "policy_type": "microban_pico_hybrid_teleop",
        "checkpoint_filename": "model_14999.pt",
        "checkpoint_iteration": "14999",
        "checkpoint_iteration_semantics": (
            "zero_based_completed_update_index_from_model_filename"
        ),
        "checkpoint_completed_updates": "15000",
        "checkpoint_sha256": "0123456789abcdef" * 4,
        "training_provenance_schema_version": str(
            EXPECTED_TRAINING_PROVENANCE_SCHEMA_VERSION
        ),
        "training_provenance_sha256": "1" * 64,
        "training_source_tree_sha256": "2" * 64,
        "training_recipe_revision": EXPECTED_RECIPE_REVISION,
        "training_actor_initialization": EXPECTED_ACTOR_INITIALIZATION,
        "training_provenance_mode": EXPECTED_TRAINING_PROVENANCE_MODE,
        "canonical_training_stage": "true",
        "training_stage_start_boundary": "10000",
        "training_stage_target_boundary": "15000",
        "training_parent_checkpoint_sha256": "3" * 64,
        "training_parent_gate_sha256": "4" * 64,
        "training_resume_source_checkpoint_sha256": "9" * 64,
        "training_resume_source_checkpoint_iteration": "9999",
        "migration_source_checkpoint_sha256": (
            EXPECTED_MIGRATION_SOURCE_CHECKPOINT_SHA256
        ),
        "migration_source_checkpoint_iteration": str(
            EXPECTED_MIGRATION_SOURCE_CHECKPOINT_ITERATION
        ),
        "migration_source_training_provenance_sha256": (
            EXPECTED_MIGRATION_SOURCE_TRAINING_PROVENANCE_SHA256
        ),
        "migration_source_tree_sha256": EXPECTED_MIGRATION_SOURCE_TREE_SHA256,
        "migration_source_gate_sha256": EXPECTED_MIGRATION_SOURCE_GATE_SHA256,
        "migration_state_transfer": EXPECTED_MIGRATION_STATE_TRANSFER,
        "migration_source_optimizer_learning_rate": str(
            EXPECTED_MIGRATION_SOURCE_OPTIMIZER_LEARNING_RATE
        ),
        "training_fixed_learning_rate": str(EXPECTED_TRAINING_FIXED_LEARNING_RATE),
        "safe_velocity_source_checkpoint_sha256": (
            EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_SHA256
        ),
        "safe_velocity_source_checkpoint_iteration": str(
            EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_ITERATION
        ),
        "safe_velocity_source_recipe_revision": (
            EXPECTED_SAFE_VELOCITY_RECIPE_REVISION
        ),
        "safe_velocity_acceptance_receipt_sha256": (
            EXPECTED_SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SHA256
        ),
        "safe_velocity_acceptance_receipt_schema_version": str(
            EXPECTED_SAFE_VELOCITY_RECEIPT_SCHEMA_VERSION
        ),
        "safe_velocity_acceptance_gate": EXPECTED_SAFE_VELOCITY_ACCEPTANCE_GATE,
        "safe_velocity_bootstrap_mapping_version": (
            EXPECTED_SAFE_VELOCITY_BOOTSTRAP_MAPPING_VERSION
        ),
        "deployment_accepted": "true",
        "acceptance_receipt_schema_version": "3",
        "acceptance_receipt_sha256": "5" * 64,
        "acceptance_status": "pass",
        "acceptance_boundary": "15000",
        "acceptance_evaluator_revision": EXPECTED_ACCEPTANCE_EVALUATOR_REVISION,
        "acceptance_revision": EXPECTED_ACCEPTANCE_REVISION,
        "acceptance_evaluator_source_sha256": "6" * 64,
        "acceptance_checkpoint_sha256": "0123456789abcdef" * 4,
        "acceptance_training_provenance_sha256": "1" * 64,
        "acceptance_recipe_revision": EXPECTED_RECIPE_REVISION,
        "acceptance_nominal_report_count": "3",
        "acceptance_moving_hmd_report_count": "3",
        "onnx_parity_gate_version": "1",
        "onnx_parity_verified": "true",
        "onnx_parity_runtime": "onnx.reference.ReferenceEvaluator",
        "onnx_parity_seed": "20260924",
        "onnx_parity_sample_count": "16",
        "onnx_parity_atol": "1e-05",
        "onnx_parity_rtol": "0.0001",
        "microban_teleop_training_contract_version": (
            EXPECTED_TRAINING_CONTRACT_VERSION
        ),
        "microban_teleop_actor_initialization": EXPECTED_ACTOR_INITIALIZATION,
        "microban_teleop_recipe_revision": EXPECTED_RECIPE_REVISION,
        "observation_schema_version": "2",
        "base_ang_vel_frame": "robot_body_xyz",
        "base_ang_vel_units": "rad_s",
        "observation_width": "83",
        "control_hz": "50.0",
        "locomotion_command_order": "linear_velocity_x,linear_velocity_y,angular_velocity_z",
        "locomotion_command_units": "m_s,m_s,rad_s",
        "locomotion_command_frame": "robot_body_forward_left_yaw_up",
        "observation_names": _csv(EXPECTED_OBSERVATION_TERMS),
        "observation_joint_names": _csv(OBSERVATION_JOINTS),
        "observation_default_joint_pos": _metadata_csv(observation_defaults),
        "action_joint_names": _csv(OBSERVATION_DOF_ORDER),
        "default_joint_pos": _metadata_csv(EXPECTED_ACTION_DEFAULT_JOINT_POS),
        "action_scale": _metadata_csv(EXPECTED_ACTION_SCALE),
        "soft_joint_pos_lower": _metadata_csv(EXPECTED_SOFT_JOINT_POS_LOWER),
        "soft_joint_pos_upper": _metadata_csv(EXPECTED_SOFT_JOINT_POS_UPPER),
        "previous_action_semantics": EXPECTED_PREVIOUS_ACTION_SEMANTICS,
        "action_target_semantics": "default_joint_pos_plus_raw_action_times_scale",
        "action_clip_semantics": "absolute_joint_position_radians",
        "action_distribution_semantics": EXPECTED_ACTION_DISTRIBUTION_SEMANTICS,
        "actor_target_guard_margin_ratio": str(
            EXPECTED_ACTOR_TARGET_GUARD_MARGIN_RATIO
        ),
        "actor_default_interior_epsilon_rad": str(
            EXPECTED_ACTOR_DEFAULT_INTERIOR_EPSILON_RAD
        ),
        "actor_latent_operational_scale_multiplier": str(
            EXPECTED_ACTOR_LATENT_OPERATIONAL_SCALE_MULTIPLIER
        ),
        "actor_latent_operational_abs_max": str(
            EXPECTED_ACTOR_LATENT_OPERATIONAL_ABS_MAX
        ),
        "actor_latent_mean_fraction": str(EXPECTED_ACTOR_LATENT_MEAN_FRACTION),
        "actor_latent_std_min_abs_max": str(EXPECTED_ACTOR_LATENT_STD_MIN_ABS_MAX),
        "actor_latent_std_min_envelope_divisor": str(
            EXPECTED_ACTOR_LATENT_STD_MIN_ENVELOPE_DIVISOR
        ),
        "actor_latent_std_abs_max": str(EXPECTED_ACTOR_LATENT_STD_ABS_MAX),
        "actor_latent_std_envelope_divisor": str(
            EXPECTED_ACTOR_LATENT_STD_ENVELOPE_DIVISOR
        ),
        "raw_action_soft_lower_json": json.dumps(
            EXPECTED_RAW_ACTION_SOFT_LOWER, separators=(",", ":")
        ),
        "raw_action_soft_upper_json": json.dumps(
            EXPECTED_RAW_ACTION_SOFT_UPPER, separators=(",", ":")
        ),
        "actor_raw_action_lower_json": json.dumps(
            EXPECTED_ACTOR_RAW_ACTION_LOWER, separators=(",", ":")
        ),
        "actor_raw_action_upper_json": json.dumps(
            EXPECTED_ACTOR_RAW_ACTION_UPPER, separators=(",", ":")
        ),
        "foot_target_lower": _csv((-0.03, -0.03, 0.0) * 2),
        "foot_target_upper": _csv((0.03, 0.03, 0.05) * 2),
        "simultaneous_both_feet_target_lower": _csv(
            EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_LOWER
        ),
        "simultaneous_both_feet_target_upper": _csv(
            EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_UPPER
        ),
        "simultaneous_both_feet_target_semantics": (
            EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_SEMANTICS
        ),
        "simultaneous_both_feet_requires_zero_twist": "true",
        "foot_target_frame": "robot_trunk_xyz_forward_left_up",
        "foot_target_units": "metres",
        "foot_target_semantics": (
            "left_xyz_then_right_xyz_trunk_frame_offset_from_episode_reset_"
            "reference_metres_periodic_command_resampling_does_not_move_reference"
        ),
        "hand_target_lower": _csv((-0.08, -0.08, -0.08) * 2),
        "hand_target_upper": _csv((0.08, 0.08, 0.08) * 2),
        "hand_target_frame": "robot_trunk_xyz_forward_left_up",
        "hand_target_units": "metres",
        "hand_target_semantics": (
            "left_xyz_then_right_xyz_then_left_right_active_flags_"
            "trunk_frame_offset_from_episode_reset_reference_metres_"
            "periodic_command_resampling_does_not_move_reference"
        ),
    }


def valid_v12_metadata():
    """Metadata fixture shared by every contract-v12 runtime test."""

    metadata = valid_metadata()
    checkpoint_sha256 = metadata["checkpoint_sha256"]
    v12_minimum = [-3.0] * 18
    v12_maximum = [4.0] * 18
    v12_absolute_maximum = [4.0] * 18
    source_minimum = [-2.0] * 18
    source_maximum = [2.0] * 18
    source_absolute_maximum = [2.0] * 18
    delta_minimum = [-1.0] * 18
    delta_maximum = [1.0] * 18
    delta_absolute_maximum = [1.0] * 18
    guard_absolute_maximum = [24.0] * 18
    source_minimum[0] = -3.0
    source_maximum[0] = 3.0
    source_absolute_maximum[0] = 3.0
    delta_minimum[0] = -2.0
    delta_maximum[0] = 2.0
    delta_absolute_maximum[0] = 2.0
    guard_absolute_maximum[0] = 30.0
    metadata.update(
        {
            "microban_teleop_training_contract_version": (
                EXPECTED_V12_TRAINING_CONTRACT_VERSION
            ),
            "microban_teleop_recipe_revision": EXPECTED_V12_RECIPE_REVISION,
            "action_width": "18",
            "observation_schema_json": json.dumps(
                [
                    ["base_ang_vel", 3],
                    ["projected_gravity", 3],
                    ["joint_pos", 21],
                    ["joint_vel", 21],
                    ["actions", 18],
                    ["command", 3],
                    ["foot_target", 6],
                    ["hand_target", 8],
                ],
                separators=(",", ":"),
            ),
            "observation_joint_names": _csv(EXPECTED_V12_OBSERVATION_JOINT_NAMES),
            "previous_action_semantics": "raw_actor_output",
            "action_clip_semantics": "none",
            "action_distribution_semantics": (
                EXPECTED_V12_ACTION_DISTRIBUTION_SEMANTICS
            ),
            "runtime_action_semantics": EXPECTED_V12_RUNTIME_ACTION_SEMANTICS,
            "deployment_accepted": "true",
            "v12_stage_gate_schema_version": "2",
            "v12_stage_gate_name": "microban_teleop_v12_stage",
            "v12_stage_gate_status": "pass",
            "v12_stage_gate_canonical_boundary": "true",
            "v12_stage_gate_sha256": "a" * 64,
            "v12_stage_gate_checkpoint_sha256": checkpoint_sha256,
            "v12_stage_gate_checkpoint_iteration": "14999",
            "v12_stage_gate_completed_updates": "15000",
            "v12_locomotion_report_sha256": "b" * 64,
            "v12_onnx_report_sha256": "c" * 64,
            "v12_tracking_report_sha256": "d" * 64,
            "v12_tracking_profile": "full_body_reachable_performance_perturbation_v2",
            "v12_raw_action_envelope_schema_version": str(
                EXPECTED_V12_RAW_ACTION_ENVELOPE_SCHEMA_VERSION
            ),
            "v12_raw_action_joint_names_json": json.dumps(
                list(OBSERVATION_DOF_ORDER), separators=(",", ":")
            ),
            "v12_raw_action_min_json": json.dumps(v12_minimum),
            "v12_raw_action_max_json": json.dumps(v12_maximum),
            "v12_raw_action_absmax_json": json.dumps(v12_absolute_maximum),
            "v12_source_raw_action_min_json": json.dumps(source_minimum),
            "v12_source_raw_action_max_json": json.dumps(source_maximum),
            "v12_source_raw_action_absmax_json": json.dumps(source_absolute_maximum),
            "v12_learned_source_delta_min_json": json.dumps(delta_minimum),
            "v12_learned_source_delta_max_json": json.dumps(delta_maximum),
            "v12_learned_source_delta_absmax_json": json.dumps(delta_absolute_maximum),
            "runtime_raw_action_guard_formula": (EXPECTED_V12_RAW_ACTION_GUARD_FORMULA),
            "runtime_raw_action_guard_multiplier": str(
                EXPECTED_V12_RAW_ACTION_GUARD_MULTIPLIER
            ),
            "runtime_raw_action_guard_absmax_json": json.dumps(guard_absolute_maximum),
            "runtime_raw_action_guard_semantics": (
                EXPECTED_V12_RAW_ACTION_GUARD_SEMANTICS
            ),
            "v12_bootstrap_provenance_schema_version": "1",
            "v12_bootstrap_mapping_version": EXPECTED_V12_BOOTSTRAP_MAPPING_VERSION,
            "v12_legacy_source_checkpoint_sha256": (
                EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256
            ),
            "v12_legacy_source_checkpoint_iteration": str(
                EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_ITERATION
            ),
            "v12_legacy_probe_sha256": EXPECTED_V12_LEGACY_PROBE_SHA256,
            "v12_legacy_probe_scenario_count": "9",
            "v12_legacy_probe_steps_per_scenario": "300",
            "v12_legacy_probe_settle_steps": "50",
            "v12_legacy_probe_seed": "42",
            "v12_source_to_target_columns_json": json.dumps(
                [list(pair) for pair in EXPECTED_V12_SOURCE_TO_TARGET_COLUMNS],
                separators=(",", ":"),
            ),
            "v12_extra_observation_columns_json": json.dumps(
                list(EXPECTED_V12_EXTRA_OBSERVATION_COLUMNS), separators=(",", ":")
            ),
            "v12_actor_topology_json": "[83,512,256,128,18]",
            "v12_normalizer_eps": "0.01",
            "v12_normalizer_semantics": EXPECTED_V12_NORMALIZER_SEMANTICS,
            "v12_trainable_actor_parameters": "mlp.0.weight_extra_columns_only",
            "adapter_gradient_schedule_revision": (
                "freeze_extra_to7000_then_hmd_hand_to10000_then_all_v1"
            ),
            "v12_active_actor_columns_at_save_json": json.dumps(
                list(EXPECTED_V12_EXTRA_OBSERVATION_COLUMNS), separators=(",", ":")
            ),
            "v12_frozen_legacy_tensors_verified": "true",
            "v12_locomotion_gate": ("microban_teleop_v12_neutral_locomotion_9x300"),
            "v12_locomotion_status": "pass",
            "v12_locomotion_seed": "42",
            "v12_locomotion_scenario_count": "9",
            "v12_locomotion_steps_per_scenario": "300",
            "v12_locomotion_settle_steps": "50",
            "v12_locomotion_fall_scenario_count": "0",
            "v12_locomotion_nonfinite_scenario_count": "0",
            "v12_actual_dynamic_soft_limit_overshoot_max_deg": str(
                EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG
            ),
            "v12_actual_dynamic_soft_limit_overshoot_max_rad": str(
                EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD
            ),
            "v12_commanded_target_soft_limit_excess_max_rad": str(
                EXPECTED_V12_COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD
            ),
            "v12_locomotion_actual_soft_limit_violation_scenario_count": "0",
            "v12_locomotion_directionally_correct_scenario_count": "8",
            "v12_locomotion_directional_scenario_count": "8",
            "v12_locomotion_raw_action_recurrence_all_steps": "true",
            "v12_onnx_gate": "microban_teleop_v12_checkpoint_onnx",
            "v12_onnx_verified": "true",
            "v12_onnx_parity_teleop_columns": "random_finite_not_zeroed",
            "v12_onnx_parity_seed": "20260925",
            "v12_onnx_parity_sample_count": "64",
            "v12_onnx_parity_atol": "2e-05",
            "v12_onnx_reference_max_abs_error": "3e-06",
            "v12_onnxruntime_cpu_max_abs_error": "4e-06",
            "v12_neutral_legacy_parity_max_abs_error": "9e-06",
            "v12_neutral_legacy_parity_sample_count": "10000",
            "hand_target_fk": json.dumps(
                EXPECTED_V12_HAND_TARGET_FK, separators=(",", ":")
            ),
        }
    )
    metadata.update(CURRENT_RUNTIME_SOURCE_IDENTITY)
    return metadata


class FakeSession:
    def __init__(
        self,
        metadata=None,
        output=None,
        *,
        input_name="obs",
        output_name="actions",
        input_type="tensor(float)",
        output_type="tensor(float)",
    ):
        self.metadata = valid_metadata() if metadata is None else metadata
        self.output = np.zeros((1, 18), dtype=np.float32) if output is None else output
        self.input_name = input_name
        self.output_name = output_name
        self.input_type = input_type
        self.output_type = output_type
        self.last_feed = None
        self.run_count = 0

    def get_inputs(self):
        return [_Io(self.input_name, [1, 83], self.input_type)]

    def get_outputs(self):
        return [_Io(self.output_name, [1, 18], self.output_type)]

    def get_modelmeta(self):
        return _Metadata(self.metadata)

    def run(self, names, feed):
        self.last_feed = feed
        self.run_count += 1
        return [self.output]


class FakeController:
    def __init__(self):
        self.kp_writes = []

    def sync_write_kp(self, ids, values):
        self.kp_writes.append((ids, values))


def observation(time_s=0.0):
    positions = {name: PICO_TELEOP_HOME_POSE[name] for name in MOTOR_TO_ID}
    velocities = {name: 0.0 for name in MOTOR_TO_ID}
    return Observation(
        robot_state=RobotState(
            time_s=time_s,
            gyro=[0.1, 0.2, 0.3],
            projected_gravity=[0.0, 0.0, -1.0],
            motor_positions=positions,
            motor_velocities=velocities,
        ),
        user_input=UserInput(
            active_moves={"walk"},
            velocity={"vx": 0.2, "vy": -0.1, "vtheta": 0.3},
            foot_target={
                "left": (9.0, -9.0, 0.04),
                "right": (0.0, 0.0, 0.0),
            },
            hand_target={"left": (0.02, 0.03, -0.04), "right": None},
        ),
    )


class PicoHybridMoveTest(unittest.TestCase):
    def test_v12_contract_rejects_runtime_source_identity_mismatch_before_inference(self):
        metadata = valid_v12_metadata()
        metadata["microban_scheduler_source_sha256"] = "0" * 64
        session = FakeSession(metadata=metadata)

        with self.assertRaisesRegex(
            PicoHybridPolicyContractError, "runtime source identity"
        ):
            PicoHybridMove(session=session)

        self.assertEqual(session.run_count, 0)

    def test_v12_contract_rechecks_runtime_source_identity_after_smoke(self):
        changed = dict(CURRENT_RUNTIME_SOURCE_IDENTITY)
        changed["microban_scheduler_source_sha256"] = "0" * 64
        session = FakeSession(metadata=valid_v12_metadata())

        with (
            patch(
                "moves.pico_hybrid.runtime_source_identity",
                side_effect=[CURRENT_RUNTIME_SOURCE_IDENTITY, changed],
            ),
            self.assertRaisesRegex(
                PicoHybridPolicyContractError, "runtime source identity"
            ),
        ):
            PicoHybridMove(session=session)

        self.assertEqual(session.run_count, 16)

    def test_v12_contract_accepts_only_final_hash_bound_raw_policy(self):
        session = FakeSession(metadata=valid_v12_metadata())
        move = PicoHybridMove(session=session)

        self.assertEqual(move._contract.training_contract_version, "12")
        self.assertEqual(
            move._contract.runtime_action_semantics,
            EXPECTED_V12_RUNTIME_ACTION_SEMANTICS,
        )
        self.assertEqual(move._contract.actor_raw_action_lower, ())
        self.assertEqual(move._contract.actor_raw_action_upper, ())
        self.assertEqual(
            move._contract.v12_legacy_source_checkpoint_sha256,
            EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            move._contract.v12_legacy_probe_sha256,
            EXPECTED_V12_LEGACY_PROBE_SHA256,
        )
        self.assertEqual(
            move._contract.v12_runtime_raw_action_guard_absolute_maximum,
            (30.0, *((24.0,) * 17)),
        )
        self.assertEqual(move._compatibility_smoke_sample_count, 16)
        self.assertEqual(session.run_count, 16)
        self.assertEqual(
            PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS,
            "compiled_soft_limit_continuous_clamp_preserve_policy_recurrence_v1",
        )

    def test_v12_contract_accepts_deadline_final_tracking_profile(self):
        metadata = valid_v12_metadata()
        metadata["v12_tracking_profile"] = (
            EXPECTED_V12_DEADLINE_FINAL_TRACKING_PROFILE
        )
        metadata["v12_onnx_parity_atol"] = "2.5e-05"
        session = FakeSession(metadata=metadata)

        move = PicoHybridMove(session=session)

        self.assertEqual(move._contract.training_contract_version, "12")
        self.assertEqual(move._compatibility_smoke_sample_count, 16)
        self.assertEqual(session.run_count, 16)

    def test_v12_soft_limits_match_metadata_or_compiled_fallback(self):
        metadata = valid_v12_metadata()
        move = PicoHybridMove(session=FakeSession(metadata=metadata))
        self.assertEqual(move._contract.soft_lower, EXPECTED_SOFT_JOINT_POS_LOWER)
        self.assertEqual(move._contract.soft_upper, EXPECTED_SOFT_JOINT_POS_UPPER)

        without_exported_limits = valid_v12_metadata()
        without_exported_limits.pop("soft_joint_pos_lower")
        without_exported_limits.pop("soft_joint_pos_upper")
        fallback = PicoHybridMove(
            session=FakeSession(metadata=without_exported_limits)
        )
        self.assertEqual(fallback._contract.soft_lower, EXPECTED_SOFT_JOINT_POS_LOWER)
        self.assertEqual(fallback._contract.soft_upper, EXPECTED_SOFT_JOINT_POS_UPPER)

        for case, mutate in (
            (
                "mismatch",
                lambda values: values.__setitem__(
                    "soft_joint_pos_upper",
                    _metadata_csv((*EXPECTED_SOFT_JOINT_POS_UPPER[:-1], 99.0)),
                ),
            ),
            (
                "incomplete",
                lambda values: values.pop("soft_joint_pos_upper"),
            ),
        ):
            rejected = valid_v12_metadata()
            mutate(rejected)
            with (
                self.subTest(case=case),
                self.assertRaises(PicoHybridPolicyContractError),
            ):
                PicoHybridMove(session=FakeSession(metadata=rejected))

    def test_compiled_soft_limit_fallback_fails_before_onnx_smoke_if_malformed(self):
        metadata = valid_v12_metadata()
        metadata.pop("soft_joint_pos_lower")
        metadata.pop("soft_joint_pos_upper")
        malformed = list(EXPECTED_SOFT_JOINT_POS_LOWER)
        malformed[0] = math.nan
        session = FakeSession(metadata=metadata)

        with (
            patch(
                "moves.pico_hybrid.EXPECTED_SOFT_JOINT_POS_LOWER",
                tuple(malformed),
            ),
            self.assertRaisesRegex(
                PicoHybridPolicyContractError,
                "physical motor-target guard is non-finite",
            ),
        ):
            PicoHybridMove(session=session)

        self.assertEqual(session.run_count, 0)

    def test_v12_contract_rejects_intermediate_or_mutated_provenance(self):
        wrong_fk_type = json.loads(
            json.dumps(EXPECTED_V12_HAND_TARGET_FK, separators=(",", ":"))
        )
        wrong_fk_type["home_joint_deg"][0][0] = False
        cases = {
            "intermediate": ("checkpoint_iteration", "2999"),
            "source": ("v12_legacy_source_checkpoint_sha256", "0" * 64),
            "probe": ("v12_legacy_probe_sha256", "0" * 64),
            "retired_v1_recipe": (
                "microban_teleop_recipe_revision",
                "legacy_velocity_model14999_masked_extra20_raw_actions_v1",
            ),
            "gradient_schedule": ("adapter_gradient_schedule_revision", "other"),
            "inactive_final_columns": (
                "v12_active_actor_columns_at_save_json",
                "[]",
            ),
            "mapping": ("v12_source_to_target_columns_json", "[]"),
            "raw_recurrence": (
                "v12_locomotion_raw_action_recurrence_all_steps",
                "false",
            ),
            "teleop_parity": ("v12_onnx_parity_teleop_columns", "zero"),
            "tracking_profile": ("v12_tracking_profile", "locomotion_only"),
            "normalizer_semantics": ("v12_normalizer_semantics", "identity"),
            "hand_fk": ("hand_target_fk", "{}"),
            "hand_fk_json_type": (
                "hand_target_fk",
                json.dumps(wrong_fk_type, separators=(",", ":")),
            ),
            "parity_error": ("v12_onnxruntime_cpu_max_abs_error", "0.001"),
            "action_clip": ("action_clip_semantics", "soft_limits"),
            "raw_envelope_joint_order": (
                "v12_raw_action_joint_names_json",
                json.dumps(list(reversed(OBSERVATION_DOF_ORDER))),
            ),
            "raw_envelope_absmax": (
                "v12_raw_action_absmax_json",
                json.dumps([3.0] * 18),
            ),
            "raw_guard_formula": (
                "runtime_raw_action_guard_formula",
                "max(v12_absmax,source_absmax+delta_absmax)",
            ),
            "raw_guard_multiplier": (
                "runtime_raw_action_guard_multiplier",
                "1.999",
            ),
            "raw_guard_value": (
                "runtime_raw_action_guard_absmax_json",
                json.dumps([7.999] * 18),
            ),
            "raw_guard_semantics": (
                "runtime_raw_action_guard_semantics",
                "clamp",
            ),
            "measured_limit_tolerance": (
                "v12_actual_dynamic_soft_limit_overshoot_max_deg",
                "5.0001",
            ),
            "command_limit_tolerance": (
                "v12_commanded_target_soft_limit_excess_max_rad",
                str(math.radians(5.0)),
            ),
        }
        for case, (name, value) in cases.items():
            metadata = valid_v12_metadata()
            metadata[name] = value
            if case == "intermediate":
                metadata["checkpoint_filename"] = "model_2999.pt"
                metadata["checkpoint_completed_updates"] = "3000"
                metadata["v12_stage_gate_checkpoint_iteration"] = "2999"
                metadata["v12_stage_gate_completed_updates"] = "3000"
            with (
                self.subTest(case=case),
                self.assertRaises(PicoHybridPolicyContractError),
            ):
                PicoHybridMove(session=FakeSession(metadata=metadata))

        for case, session in (
            (
                "input_name",
                FakeSession(metadata=valid_v12_metadata(), input_name="input"),
            ),
            (
                "output_name",
                FakeSession(metadata=valid_v12_metadata(), output_name="output"),
            ),
        ):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(PicoHybridPolicyContractError, "tensor names"),
            ):
                PicoHybridMove(session=session)

        for case, session in (
            (
                "input_type",
                FakeSession(metadata=valid_v12_metadata(), input_type="tensor(double)"),
            ),
            (
                "output_type",
                FakeSession(metadata=valid_v12_metadata(), output_type="tensor(int64)"),
            ),
        ):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(PicoHybridPolicyContractError, "float32"),
            ):
                PicoHybridMove(session=session)

    def test_v12_contract_rejects_missing_or_malformed_envelope_evidence(self):
        cases = {
            "missing": ("v12_source_raw_action_absmax_json", None),
            "wrong_width": ("v12_raw_action_min_json", "[]"),
            "boolean": (
                "v12_learned_source_delta_max_json",
                json.dumps([True] * 18),
            ),
            "nonfinite": (
                "v12_source_raw_action_min_json",
                "[NaN," + ",".join("0" for _ in range(17)) + "]",
            ),
            "float32_overflow": (
                "v12_source_raw_action_max_json",
                json.dumps([1.0e100] * 18),
            ),
            "reversed_range": (
                "v12_learned_source_delta_min_json",
                json.dumps([2.0] * 18),
            ),
        }
        for case, (name, value) in cases.items():
            metadata = valid_v12_metadata()
            if value is None:
                metadata.pop(name)
            else:
                metadata[name] = value
            with (
                self.subTest(case=case),
                self.assertRaises(PicoHybridPolicyContractError),
            ):
                PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_v12_compatibility_smoke_requires_finite_float32_output(self):
        for case, output in (
            ("shape", np.zeros(18, dtype=np.float32)),
            ("nan", np.full((1, 18), np.nan, dtype=np.float32)),
            ("float32_overflow", np.full((1, 18), 1.0e100, dtype=np.float64)),
            ("nonnumeric", np.full((1, 18), "bad", dtype=object)),
            ("finite_amplitude", np.full((1, 18), 8.01, dtype=np.float32)),
        ):
            with (
                self.subTest(case=case),
                self.assertRaises(PicoHybridPolicyRuntimeError),
            ):
                validate_v12_onnxruntime_compatibility(
                    FakeSession(metadata=valid_v12_metadata(), output=output),
                    "obs",
                    (8.0,) * 18,
                )

    def test_v12_step_clamps_motor_target_but_preserves_raw_recurrence(self):
        raw = np.linspace(-3.5, 3.5, 18, dtype=np.float32).reshape(1, 18)
        session = FakeSession(metadata=valid_v12_metadata(), output=raw)
        move = PicoHybridMove(session=session)
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)

        expected_targets = {
            name: max(
                EXPECTED_SOFT_JOINT_POS_LOWER[index],
                min(
                    EXPECTED_SOFT_JOINT_POS_UPPER[index],
                    EXPECTED_ACTION_DEFAULT_JOINT_POS[index]
                    + float(raw[0, index]),
                ),
            )
            for index, name in enumerate(OBSERVATION_DOF_ORDER)
        }
        for name, expected in expected_targets.items():
            self.assertEqual(command.target_angles[name], expected)
        self.assertEqual(
            command.target_angles["left_ankle_roll"],
            EXPECTED_SOFT_JOINT_POS_UPPER[-1],
        )
        next_observation = move.build_observation(obs)
        np.testing.assert_array_equal(
            np.asarray(next_observation[48:66], dtype=np.float32), raw[0]
        )

    def test_v12_step_accepts_guard_boundary_and_clamps_without_exception(self):
        raw = np.full((1, 18), 24.0, dtype=np.float32)
        move = PicoHybridMove(
            session=FakeSession(metadata=valid_v12_metadata(), output=raw)
        )
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)

        np.testing.assert_array_equal(move._last_action, raw[0])
        self.assertEqual(
            command.target_angles[OBSERVATION_DOF_ORDER[0]],
            EXPECTED_SOFT_JOINT_POS_UPPER[0],
        )

    def test_v12_extreme_targets_never_add_five_degree_command_margin(self):
        # Five degrees is reserved for measured-angle acceptance adjudication.
        # It must never widen an actuator command beyond the compiled soft limit.
        raw = np.asarray(
            [[30.0, *(-24.0 if index % 2 else 24.0 for index in range(1, 18))]],
            dtype=np.float32,
        )
        move = PicoHybridMove(
            session=FakeSession(metadata=valid_v12_metadata(), output=raw)
        )
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)

        move.step(obs, command)

        self.assertEqual(move.state, MoveState.ACTIVE)
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            lower = EXPECTED_SOFT_JOINT_POS_LOWER[index]
            upper = EXPECTED_SOFT_JOINT_POS_UPPER[index]
            target = command.target_angles[name]
            self.assertLessEqual(lower, target)
            self.assertLessEqual(target, upper)
            expected = upper if raw[0, index] > 0.0 else lower
            self.assertEqual(target, expected)
            widened = expected + math.copysign(
                math.radians(5.0), float(raw[0, index])
            )
            self.assertNotEqual(target, widened)
        np.testing.assert_array_equal(move._last_action, raw[0])

    def test_v12_step_rejects_guard_escape_before_any_target_write(self):
        outside = np.nextafter(np.float32(24.0), np.float32(math.inf))
        for sign in (-1.0, 1.0):
            raw = np.zeros((1, 18), dtype=np.float32)
            raw[0, 7] = sign * outside
            session = FakeSession(metadata=valid_v12_metadata())
            move = PicoHybridMove(session=session)
            obs = observation()
            command = MotorCommand()
            move.on_start(obs, command)
            held = dict(command.target_angles)
            session.output = raw

            with (
                self.subTest(sign=sign),
                self.assertRaisesRegex(
                    PicoHybridPolicyRuntimeError, "finite-amplitude guard"
                ),
            ):
                move.step(obs, command)

            self.assertEqual(command.target_angles, held)
            np.testing.assert_array_equal(
                move._last_action, np.zeros(18, dtype=np.float32)
            )

    def test_v12_step_rejects_late_float32_overflow_before_any_target_write(self):
        session = FakeSession(metadata=valid_v12_metadata())
        move = PicoHybridMove(session=session)
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        held = dict(command.target_angles)
        session.output = np.full((1, 18), 1.0e100, dtype=np.float64)

        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "float32"):
            move.step(obs, command)

        self.assertEqual(command.target_angles, held)
        np.testing.assert_array_equal(move._last_action, np.zeros(18, dtype=np.float32))

    def test_v12_step_rejects_observation_float32_overflow_before_inference(self):
        session = FakeSession(metadata=valid_v12_metadata())
        move = PicoHybridMove(session=session)
        obs = observation()
        command = MotorCommand()
        move.on_start(obs, command)
        held = dict(command.target_angles)
        calls_before_step = session.run_count
        obs.robot_state.motor_positions[OBSERVATION_DOF_ORDER[0]] = 1.0e100

        with self.assertRaisesRegex(
            PicoHybridPolicyRuntimeError, "observation.*float32"
        ):
            move.step(obs, command)

        self.assertEqual(session.run_count, calls_before_step)
        self.assertEqual(command.target_angles, held)

    def test_onnxruntime_compatibility_smoke_uses_fixed_finite_corpus(self):
        corpus = onnxruntime_compatibility_smoke_inputs()
        self.assertEqual(corpus.shape, (16, 1, 83))
        self.assertTrue(np.isfinite(corpus).all())
        self.assertEqual(corpus[0, 0, 5], -1.0)
        self.assertEqual(np.count_nonzero(corpus[0]), 1)
        self.assertTrue(np.all(corpus[1] <= corpus[2]))

        session = FakeSession()
        sample_count = validate_onnxruntime_compatibility(session, "obs")
        self.assertEqual(sample_count, 16)
        self.assertEqual(session.run_count, 16)
        self.assertEqual(session.last_feed["obs"].shape, (1, 83))

    def test_onnxruntime_compatibility_smoke_rejects_unsafe_output(self):
        for case, output in (
            ("wrong_shape", np.zeros((18,), dtype=np.float32)),
            ("nonfinite", np.full((1, 18), np.nan, dtype=np.float32)),
            ("nonnumeric", np.full((1, 18), "bad", dtype=object)),
        ):
            with (
                self.subTest(case=case),
                self.assertRaises(PicoHybridPolicyRuntimeError),
            ):
                validate_onnxruntime_compatibility(
                    FakeSession(output=output),
                    "obs",
                )

    def test_onnxruntime_compatibility_requires_open_actor_bounds(self):
        for case, output in (
            (
                "lower_endpoint",
                np.asarray([EXPECTED_ACTOR_RAW_ACTION_LOWER], dtype=np.float64),
            ),
            (
                "upper_endpoint",
                np.asarray([EXPECTED_ACTOR_RAW_ACTION_UPPER], dtype=np.float64),
            ),
            (
                "outside",
                np.asarray([EXPECTED_ACTOR_RAW_ACTION_UPPER], dtype=np.float64) + 1.0,
            ),
        ):
            with (
                self.subTest(case=case),
                self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "actor bound"),
            ):
                validate_onnxruntime_compatibility(
                    FakeSession(output=output),
                    "obs",
                )

    def test_onnxruntime_compatibility_requires_current_transform_envelope(self):
        output = np.zeros((1, 18), dtype=np.float64)
        output[0, 0] = (
            EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER[0]
            + EXPECTED_ACTOR_RAW_ACTION_UPPER[0]
        ) / 2.0
        with self.assertRaisesRegex(
            PicoHybridPolicyRuntimeError, "deterministic transform envelope"
        ):
            validate_onnxruntime_compatibility(FakeSession(output=output), "obs")

    def test_v10_latent_envelope_is_finite_nested_and_has_ten_sigma_margin(self):
        vectors = (
            EXPECTED_ACTOR_LATENT_OPERATIONAL_LOWER,
            EXPECTED_ACTOR_LATENT_OPERATIONAL_UPPER,
            EXPECTED_ACTOR_LATENT_MEAN_LOWER,
            EXPECTED_ACTOR_LATENT_MEAN_UPPER,
            EXPECTED_ACTOR_LATENT_MIN_STD,
            EXPECTED_ACTOR_LATENT_MAX_STD,
            EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER,
            EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER,
        )
        self.assertTrue(all(len(vector) == 18 for vector in vectors))
        self.assertTrue(
            all(math.isfinite(value) for vector in vectors for value in vector)
        )
        for index in range(18):
            operational_lower = EXPECTED_ACTOR_LATENT_OPERATIONAL_LOWER[index]
            operational_upper = EXPECTED_ACTOR_LATENT_OPERATIONAL_UPPER[index]
            mean_lower = EXPECTED_ACTOR_LATENT_MEAN_LOWER[index]
            mean_upper = EXPECTED_ACTOR_LATENT_MEAN_UPPER[index]
            max_std = EXPECTED_ACTOR_LATENT_MAX_STD[index]
            self.assertLess(operational_lower, mean_lower)
            self.assertLess(mean_lower, 0.0)
            self.assertLess(0.0, mean_upper)
            self.assertLess(mean_upper, operational_upper)
            self.assertGreaterEqual(mean_lower - 10.0 * max_std, operational_lower)
            self.assertLessEqual(mean_upper + 10.0 * max_std, operational_upper)
            self.assertLess(
                EXPECTED_ACTOR_RAW_ACTION_LOWER[index],
                EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER[index],
            )
            self.assertLess(
                EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER[index],
                EXPECTED_ACTOR_RAW_ACTION_UPPER[index],
            )

    def test_model_load_runs_fixed_corpus_and_rejects_actor_endpoint(self):
        session = FakeSession()
        move = PicoHybridMove(session=session)
        self.assertEqual(move._compatibility_smoke_sample_count, 16)
        self.assertEqual(session.run_count, 16)

        endpoint = np.asarray([EXPECTED_ACTOR_RAW_ACTION_UPPER], dtype=np.float64)
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "actor bound"):
            PicoHybridMove(session=FakeSession(output=endpoint))

    def test_contract_accepts_and_exposes_gated_export_provenance(self):
        move = PicoHybridMove(session=FakeSession())
        self.assertEqual(move._contract.checkpoint_filename, "model_14999.pt")
        self.assertEqual(move._contract.checkpoint_iteration, 14999)
        self.assertEqual(move._contract.checkpoint_completed_updates, 15000)
        self.assertEqual(
            move._contract.checkpoint_sha256,
            "0123456789abcdef" * 4,
        )
        self.assertEqual(
            valid_metadata()["microban_teleop_actor_initialization"],
            EXPECTED_ACTOR_INITIALIZATION,
        )
        self.assertEqual(
            valid_metadata()["microban_teleop_recipe_revision"],
            EXPECTED_RECIPE_REVISION,
        )
        self.assertEqual(move._contract.training_provenance_sha256, "1" * 64)
        self.assertEqual(move._contract.training_source_tree_sha256, "2" * 64)
        self.assertEqual(
            move._contract.training_resume_source_checkpoint_sha256, "9" * 64
        )
        self.assertEqual(
            move._contract.training_resume_source_checkpoint_iteration, 9999
        )
        self.assertEqual(
            move._contract.migration_source_checkpoint_sha256,
            EXPECTED_MIGRATION_SOURCE_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            move._contract.migration_source_checkpoint_iteration,
            EXPECTED_MIGRATION_SOURCE_CHECKPOINT_ITERATION,
        )
        self.assertEqual(
            move._contract.migration_source_training_provenance_sha256,
            EXPECTED_MIGRATION_SOURCE_TRAINING_PROVENANCE_SHA256,
        )
        self.assertEqual(
            move._contract.migration_source_tree_sha256,
            EXPECTED_MIGRATION_SOURCE_TREE_SHA256,
        )
        self.assertEqual(
            move._contract.migration_source_gate_sha256,
            EXPECTED_MIGRATION_SOURCE_GATE_SHA256,
        )
        self.assertEqual(
            move._contract.migration_state_transfer,
            EXPECTED_MIGRATION_STATE_TRANSFER,
        )
        self.assertEqual(
            move._contract.migration_source_optimizer_learning_rate,
            EXPECTED_MIGRATION_SOURCE_OPTIMIZER_LEARNING_RATE,
        )
        self.assertEqual(
            move._contract.training_fixed_learning_rate,
            EXPECTED_TRAINING_FIXED_LEARNING_RATE,
        )
        self.assertEqual(move._contract.acceptance_receipt_sha256, "5" * 64)
        self.assertEqual(
            move._contract.safe_velocity_source_checkpoint_sha256,
            EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            move._contract.safe_velocity_source_checkpoint_iteration,
            EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_ITERATION,
        )
        self.assertEqual(
            move._contract.safe_velocity_acceptance_receipt_sha256,
            EXPECTED_SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SHA256,
        )
        self.assertEqual(
            move._contract.acceptance_evaluator_source_sha256,
            "6" * 64,
        )

    def test_contract_rejects_missing_or_inconsistent_export_gate_metadata(self):
        mutations = {
            "missing_verified": ("onnx_parity_verified", None),
            "false_verified": ("onnx_parity_verified", "false"),
            "wrong_gate_version": ("onnx_parity_gate_version", "2"),
            "wrong_runtime": ("onnx_parity_runtime", "onnxruntime"),
            "wrong_seed": ("onnx_parity_seed", "1"),
            "wrong_sample_count": ("onnx_parity_sample_count", "15"),
            "wrong_atol": ("onnx_parity_atol", "0.001"),
            "wrong_rtol": ("onnx_parity_rtol", "0.001"),
            "bad_filename": ("checkpoint_filename", "checkpoint.pt"),
            "noncanonical_filename": ("checkpoint_filename", "model_014999.pt"),
            "nonnumeric_iteration": ("checkpoint_iteration", "latest"),
            "noncanonical_iteration": ("checkpoint_iteration", "014999"),
            "mismatched_iteration": ("checkpoint_iteration", "14998"),
            "wrong_iteration_semantics": ("checkpoint_iteration_semantics", "unknown"),
            "wrong_completed_updates": ("checkpoint_completed_updates", "14999"),
            "uppercase_sha256": ("checkpoint_sha256", "A" * 64),
            "short_sha256": ("checkpoint_sha256", "a" * 63),
        }
        for case, (field, value) in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                if value is None:
                    del metadata[field]
                else:
                    metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_missing_body_frame(self):
        metadata = valid_metadata()
        del metadata["base_ang_vel_frame"]
        with self.assertRaises(PicoHybridPolicyContractError):
            PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_v1_through_v9_or_missing_effective_action_contract(self):
        mutations = {
            "missing_training_contract": (
                "microban_teleop_training_contract_version",
                None,
            ),
            "v1_training_contract": (
                "microban_teleop_training_contract_version",
                "1",
            ),
            "v2_training_contract": (
                "microban_teleop_training_contract_version",
                "2",
            ),
            "v3_training_contract": (
                "microban_teleop_training_contract_version",
                "3",
            ),
            "v4_training_contract": (
                "microban_teleop_training_contract_version",
                "4",
            ),
            "v5_training_contract": (
                "microban_teleop_training_contract_version",
                "5",
            ),
            "v6_training_contract": (
                "microban_teleop_training_contract_version",
                "6",
            ),
            "v7_training_contract": (
                "microban_teleop_training_contract_version",
                "7",
            ),
            "v8_training_contract": (
                "microban_teleop_training_contract_version",
                "8",
            ),
            "v9_training_contract": (
                "microban_teleop_training_contract_version",
                "9",
            ),
            "v1_schema": ("observation_schema_version", "1"),
            "v1_raw_previous_action": (
                "previous_action_semantics",
                "raw_policy_output_before_target_clip",
            ),
        }
        for case, (field, value) in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                if value is None:
                    del metadata[field]
                else:
                    metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_requires_exact_v10_actor_and_recipe_provenance(self):
        mutations = {
            "missing_actor_initialization": (
                "microban_teleop_actor_initialization",
                None,
            ),
            "legacy_actor_initialization": (
                "microban_teleop_actor_initialization",
                "bounded_raw_safe_velocity_actor_only_63_to_83_zero_new_columns_v1",
            ),
            "missing_recipe_revision": ("microban_teleop_recipe_revision", None),
            "different_recipe_revision": (
                "microban_teleop_recipe_revision",
                (
                    "v9_accepted_safe_velocity_bootstrap_no_walk004_prior_"
                    "full_pico_curriculum_v3"
                ),
            ),
        }
        for case, (field, value) in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                if value is None:
                    del metadata[field]
                else:
                    metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_requires_final_canonical_training_and_acceptance(self):
        required_fields = (
            "training_provenance_schema_version",
            "training_provenance_sha256",
            "training_source_tree_sha256",
            "training_recipe_revision",
            "training_actor_initialization",
            "training_provenance_mode",
            "canonical_training_stage",
            "training_stage_start_boundary",
            "training_stage_target_boundary",
            "training_parent_checkpoint_sha256",
            "training_parent_gate_sha256",
            "training_resume_source_checkpoint_sha256",
            "training_resume_source_checkpoint_iteration",
            "migration_source_checkpoint_sha256",
            "migration_source_checkpoint_iteration",
            "migration_source_training_provenance_sha256",
            "migration_source_tree_sha256",
            "migration_source_gate_sha256",
            "migration_state_transfer",
            "migration_source_optimizer_learning_rate",
            "training_fixed_learning_rate",
            "deployment_accepted",
            "acceptance_receipt_schema_version",
            "acceptance_receipt_sha256",
            "acceptance_status",
            "acceptance_boundary",
            "acceptance_evaluator_revision",
            "acceptance_revision",
            "acceptance_evaluator_source_sha256",
            "acceptance_checkpoint_sha256",
            "acceptance_training_provenance_sha256",
            "acceptance_recipe_revision",
            "acceptance_nominal_report_count",
            "acceptance_moving_hmd_report_count",
        )
        for field in required_fields:
            with self.subTest(field=field, mutation="missing"):
                metadata = valid_metadata()
                del metadata[field]
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

        mutations = {
            "missing_training_schema": ("training_provenance_schema_version", None),
            "wrong_training_schema": ("training_provenance_schema_version", "1"),
            "noncanonical_training_schema": (
                "training_provenance_schema_version",
                "01",
            ),
            "missing_training_digest": ("training_provenance_sha256", None),
            "uppercase_training_digest": ("training_provenance_sha256", "A" * 64),
            "missing_source_digest": ("training_source_tree_sha256", None),
            "short_source_digest": ("training_source_tree_sha256", "a" * 63),
            "wrong_training_recipe": (
                "training_recipe_revision",
                (
                    "v9_accepted_safe_velocity_bootstrap_no_walk004_prior_"
                    "full_pico_curriculum_v3"
                ),
            ),
            "wrong_training_initialization": (
                "training_actor_initialization",
                "bounded_raw_safe_velocity_actor_only_63_to_83_zero_new_columns_v1",
            ),
            "generic_training": ("training_provenance_mode", "generic"),
            "noncanonical_training": ("canonical_training_stage", "false"),
            "wrong_stage_start": ("training_stage_start_boundary", "7000"),
            "wrong_stage_target": ("training_stage_target_boundary", "10000"),
            "missing_parent_checkpoint": (
                "training_parent_checkpoint_sha256",
                None,
            ),
            "missing_parent_gate": ("training_parent_gate_sha256", None),
            "uppercase_resume_source_digest": (
                "training_resume_source_checkpoint_sha256",
                "A" * 64,
            ),
            "none_resume_source_digest": (
                "training_resume_source_checkpoint_sha256",
                "none",
            ),
            "none_resume_source_iteration": (
                "training_resume_source_checkpoint_iteration",
                "none",
            ),
            "noncanonical_resume_source_iteration": (
                "training_resume_source_checkpoint_iteration",
                "09999",
            ),
            "resume_source_before_final_stage": (
                "training_resume_source_checkpoint_iteration",
                "9998",
            ),
            "resume_source_is_final_checkpoint": (
                "training_resume_source_checkpoint_iteration",
                "14999",
            ),
            "not_accepted": ("deployment_accepted", "false"),
            "missing_receipt_schema": ("acceptance_receipt_schema_version", None),
            "old_receipt_schema": ("acceptance_receipt_schema_version", "2"),
            "missing_receipt_digest": ("acceptance_receipt_sha256", None),
            "uppercase_receipt_digest": ("acceptance_receipt_sha256", "F" * 64),
            "diagnostic_status": ("acceptance_status", "diagnostic"),
            "wrong_acceptance_boundary": ("acceptance_boundary", "10000"),
            "old_evaluator": (
                "acceptance_evaluator_revision",
                "microban_teleop_deterministic_evaluator_v9_2",
            ),
            "old_acceptance": (
                "acceptance_revision",
                "microban_teleop_acceptance_v9_2",
            ),
            "missing_evaluator_digest": (
                "acceptance_evaluator_source_sha256",
                None,
            ),
            "wrong_accepted_checkpoint": (
                "acceptance_checkpoint_sha256",
                "7" * 64,
            ),
            "wrong_accepted_training": (
                "acceptance_training_provenance_sha256",
                "8" * 64,
            ),
            "wrong_accepted_recipe": (
                "acceptance_recipe_revision",
                (
                    "v9_accepted_safe_velocity_bootstrap_no_walk004_prior_"
                    "full_pico_curriculum_v3"
                ),
            ),
            "wrong_nominal_count": ("acceptance_nominal_report_count", "2"),
            "wrong_moving_hmd_count": (
                "acceptance_moving_hmd_report_count",
                "2",
            ),
        }
        for case, (field, value) in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                if value is None:
                    del metadata[field]
                else:
                    metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

        metadata = valid_metadata()
        metadata.update(
            {
                "checkpoint_filename": "model_19999.pt",
                "checkpoint_iteration": "19999",
                "checkpoint_completed_updates": "20000",
            }
        )
        with self.assertRaisesRegex(
            PicoHybridPolicyContractError,
            "final stage boundary",
        ):
            PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_accepts_final_stage_resume_source_iteration_boundaries(self):
        for iteration in ("9999", "14998"):
            with self.subTest(iteration=iteration):
                metadata = valid_metadata()
                metadata["training_resume_source_checkpoint_iteration"] = iteration
                PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_requires_exact_v10_migration_provenance(self):
        mutations = {
            "wrong_source_checkpoint": (
                "migration_source_checkpoint_sha256",
                "a" * 64,
            ),
            "noncanonical_source_iteration": (
                "migration_source_checkpoint_iteration",
                "01499",
            ),
            "wrong_source_iteration": (
                "migration_source_checkpoint_iteration",
                "1500",
            ),
            "wrong_source_training": (
                "migration_source_training_provenance_sha256",
                "b" * 64,
            ),
            "wrong_source_tree": ("migration_source_tree_sha256", "c" * 64),
            "wrong_source_gate": ("migration_source_gate_sha256", "d" * 64),
            "wrong_state_transfer": (
                "migration_state_transfer",
                "actor_only_v1",
            ),
            "wrong_source_learning_rate": (
                "migration_source_optimizer_learning_rate",
                "1e-5",
            ),
            "nonfinite_source_learning_rate": (
                "migration_source_optimizer_learning_rate",
                "nan",
            ),
            "wrong_fixed_learning_rate": ("training_fixed_learning_rate", "1e-4"),
            "nonfinite_fixed_learning_rate": (
                "training_fixed_learning_rate",
                "inf",
            ),
        }
        for case, (field, value) in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_requires_exact_safe_velocity_bootstrap_identity(self):
        mutations = {
            "missing_source_sha": ("safe_velocity_source_checkpoint_sha256", None),
            "uppercase_source_sha": (
                "safe_velocity_source_checkpoint_sha256",
                "A" * 64,
            ),
            "wrong_source_sha": (
                "safe_velocity_source_checkpoint_sha256",
                "a" * 64,
            ),
            "missing_source_iteration": (
                "safe_velocity_source_checkpoint_iteration",
                None,
            ),
            "noncanonical_source_iteration": (
                "safe_velocity_source_checkpoint_iteration",
                "0500",
            ),
            "wrong_source_iteration": (
                "safe_velocity_source_checkpoint_iteration",
                "501",
            ),
            "wrong_source_recipe": (
                "safe_velocity_source_recipe_revision",
                "scratch_bounded_inward_shoulder_sagittal_exploration_v6",
            ),
            "missing_receipt_sha": (
                "safe_velocity_acceptance_receipt_sha256",
                None,
            ),
            "wrong_receipt_sha": (
                "safe_velocity_acceptance_receipt_sha256",
                "b" * 64,
            ),
            "wrong_receipt_schema": (
                "safe_velocity_acceptance_receipt_schema_version",
                "1",
            ),
            "wrong_acceptance_gate": (
                "safe_velocity_acceptance_gate",
                "microban_safe_velocity_fixed_forward_v1",
            ),
            "wrong_mapping": (
                "safe_velocity_bootstrap_mapping_version",
                "legacy_velocity_63_to_teleop_83",
            ),
        }
        for case, (field, value) in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                if value is None:
                    del metadata[field]
                else:
                    metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_missing_or_widened_both_feet_support(self):
        mutations = {
            "missing_lower": ("simultaneous_both_feet_target_lower", None),
            "play_environment_12mm_upper": (
                "simultaneous_both_feet_target_upper",
                _csv((0.01, 0.01, 0.012) * 2),
            ),
            "widened_upper": (
                "simultaneous_both_feet_target_upper",
                _csv((0.03, 0.03, 0.05) * 2),
            ),
            "wrong_semantics": (
                "simultaneous_both_feet_target_semantics",
                "ordinary_foot_bounds",
            ),
            "moving_allowed": (
                "simultaneous_both_feet_requires_zero_twist",
                "false",
            ),
        }
        for case, (field, value) in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                if value is None:
                    del metadata[field]
                else:
                    metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_mutated_action_safety_vectors(self):
        mutations = {
            "default_joint_pos": (0, 0.5),
            "action_scale": (0, 2.0),
            "soft_joint_pos_lower": (0, -99.0),
            "soft_joint_pos_upper": (0, 99.0),
        }
        for field, (index, value) in mutations.items():
            with self.subTest(field=field):
                metadata = valid_metadata()
                values = metadata[field].split(",")
                values[index] = str(value)
                metadata[field] = ",".join(values)
                with self.assertRaisesRegex(
                    PicoHybridPolicyContractError,
                    f"{field} does not match the fixed Microban contract",
                ):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_requires_exact_v10_bounded_actor_metadata(self):
        scalar_mutations = {
            "missing_distribution": ("action_distribution_semantics", None),
            "wrong_distribution": ("action_distribution_semantics", "tanh"),
            "missing_guard_ratio": ("actor_target_guard_margin_ratio", None),
            "wrong_guard_ratio": ("actor_target_guard_margin_ratio", "0.051"),
            "nonfinite_guard_ratio": ("actor_target_guard_margin_ratio", "nan"),
            "missing_epsilon": ("actor_default_interior_epsilon_rad", None),
            "wrong_epsilon": ("actor_default_interior_epsilon_rad", "0.001"),
            "missing_latent_scale": (
                "actor_latent_operational_scale_multiplier",
                None,
            ),
            "wrong_latent_scale": (
                "actor_latent_operational_scale_multiplier",
                "1025",
            ),
            "missing_latent_abs_max": ("actor_latent_operational_abs_max", None),
            "nonfinite_latent_abs_max": (
                "actor_latent_operational_abs_max",
                "inf",
            ),
            "wrong_mean_fraction": ("actor_latent_mean_fraction", "0.5"),
            "missing_std_min_abs_max": ("actor_latent_std_min_abs_max", None),
            "wrong_std_min_abs_max": ("actor_latent_std_min_abs_max", "0.01"),
            "wrong_std_min_divisor": (
                "actor_latent_std_min_envelope_divisor",
                "63",
            ),
            "missing_std_abs_max": ("actor_latent_std_abs_max", None),
            "wrong_std_abs_max": ("actor_latent_std_abs_max", "0.15"),
            "wrong_std_divisor": ("actor_latent_std_envelope_divisor", "15"),
        }
        for case, (field, value) in scalar_mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                if value is None:
                    del metadata[field]
                else:
                    metadata[field] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

        for field in (
            "raw_action_soft_lower_json",
            "raw_action_soft_upper_json",
            "actor_raw_action_lower_json",
            "actor_raw_action_upper_json",
        ):
            with self.subTest(field=field, mutation="missing"):
                metadata = valid_metadata()
                del metadata[field]
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

            with self.subTest(field=field, mutation="tampered"):
                metadata = valid_metadata()
                values = json.loads(metadata[field])
                values[0] += 1.0e-5
                metadata[field] = json.dumps(values)
                with self.assertRaisesRegex(
                    PicoHybridPolicyContractError,
                    "independently derived Microban contract",
                ):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_non_strict_or_malformed_json_bound_vectors(self):
        mutations = {
            "malformed": "[",
            "wrong_width": json.dumps([0.0] * 17),
            "nonstandard_nan": "[NaN," + ",".join(["0"] * 17) + "]",
            "boolean": json.dumps([True] + [0.0] * 17),
            "string": json.dumps(["0"] + [0.0] * 17),
        }
        for case, value in mutations.items():
            with self.subTest(case=case):
                metadata = valid_metadata()
                metadata["raw_action_soft_lower_json"] = value
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_actor_bounds_without_strict_zero_and_soft_interior(self):
        for case, field, replacement in (
            (
                "zero_on_lower_endpoint",
                "actor_raw_action_lower_json",
                0.0,
            ),
            (
                "actor_equals_soft_lower",
                "actor_raw_action_lower_json",
                EXPECTED_RAW_ACTION_SOFT_LOWER[0],
            ),
            (
                "actor_equals_soft_upper",
                "actor_raw_action_upper_json",
                EXPECTED_RAW_ACTION_SOFT_UPPER[0],
            ),
        ):
            with self.subTest(case=case):
                metadata = valid_metadata()
                values = json.loads(metadata[field])
                values[0] = replacement
                metadata[field] = json.dumps(values)
                with self.assertRaisesRegex(
                    PicoHybridPolicyContractError,
                    "strictly inside raw soft bounds",
                ):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_pristine_or_negative_checkpoint_identity(self):
        for filename, iteration in (
            ("model_pristine.pt", "-1"),
            ("model_-1.pt", "-1"),
        ):
            with self.subTest(filename=filename):
                metadata = valid_metadata()
                metadata["checkpoint_filename"] = filename
                metadata["checkpoint_iteration"] = iteration
                metadata["checkpoint_completed_updates"] = "0"
                metadata["checkpoint_iteration_semantics"] = (
                    "pristine_velocity_bootstrap_before_first_ppo_update"
                )
                with self.assertRaises(PicoHybridPolicyContractError):
                    PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_contract_rejects_mutated_observation_default(self):
        metadata = valid_metadata()
        values = metadata["observation_default_joint_pos"].split(",")
        values[0] = "0.5"
        metadata["observation_default_joint_pos"] = ",".join(values)
        with self.assertRaisesRegex(
            PicoHybridPolicyContractError,
            "observation_default_joint_pos does not match",
        ):
            PicoHybridMove(session=FakeSession(metadata=metadata))

    def test_observation_is_exact_83_ordered_values_and_clips_targets(self):
        session = FakeSession()
        move = PicoHybridMove(
            session=session,
            gyro_transform=lambda value: value,
        )
        values = move.build_observation(observation())
        self.assertEqual(len(values), 83)
        self.assertEqual(values[:6], [0.1, 0.2, 0.3, 0.0, 0.0, -1.0])
        self.assertEqual(values[6:27], [0.0] * 21)
        self.assertEqual(values[27:48], [0.0] * 21)
        self.assertEqual(values[48:66], [0.0] * 18)
        self.assertEqual(values[66:69], [0.2, -0.1, 0.3])
        self.assertEqual(values[69:75], [0.03, -0.03, 0.04, 0.0, 0.0, 0.0])
        self.assertEqual(values[75:83], [0.02, 0.03, -0.04, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_step_rejects_out_of_contract_output_before_writing_any_target(self):
        raw = np.full((1, 18), 100.0, dtype=np.float32)
        session = FakeSession()
        move = PicoHybridMove(session=session, gyro_transform=lambda value: value)
        # Mutate the fake after the load-time corpus gate. The current contract
        # treats a graph that escapes its physical transform as a fault, not an
        # action to silently saturate.
        session.output = raw
        obs = observation()
        move.on_start(obs, MotorCommand())
        command = MotorCommand()
        initial_targets = dict(command.target_angles)
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "actor bound"):
            move.step(obs, command)
        self.assertEqual(command.target_angles, initial_targets)
        np.testing.assert_array_equal(move._last_action, np.zeros(18))

    def test_step_consumes_physical_onnx_action_without_second_transform(self):
        raw = np.zeros((1, 18), dtype=np.float32)
        raw[0, 0] = 0.25 * EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER[0]
        session = FakeSession()
        move = PicoHybridMove(session=session, gyro_transform=lambda value: value)
        session.output = raw
        obs = observation()
        move.on_start(obs, MotorCommand())
        command = MotorCommand()
        move.step(obs, command)
        expected_target = EXPECTED_ACTION_DEFAULT_JOINT_POS[0] + float(raw[0, 0])
        self.assertEqual(
            command.target_angles[OBSERVATION_DOF_ORDER[0]], expected_target
        )
        self.assertEqual(move._last_action[0], raw[0, 0])

    def test_simultaneous_both_feet_require_live_bounds_and_zero_twist(self):
        move = PicoHybridMove(session=FakeSession(), gyro_transform=lambda value: value)
        obs = observation()
        obs.user_input.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
        obs.user_input.foot_target = {
            "left": (0.008, -0.008, 0.016),
            "right": (-0.008, 0.008, 0.016),
        }
        feet, _hands = move._body_targets(obs)
        np.testing.assert_allclose(
            feet,
            [*obs.user_input.foot_target["left"], *obs.user_input.foot_target["right"]],
        )

        obs.user_input.foot_target["left"] = (0.0080001, 0.0, 0.01)
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "live bound"):
            move._body_targets(obs)

        obs.user_input.foot_target["left"] = (0.005, 0.0, 0.01)
        obs.user_input.velocity["vx"] = 0.01
        with self.assertRaisesRegex(
            PicoHybridPolicyRuntimeError, "zero locomotion command"
        ):
            move._body_targets(obs)

    def test_support_floor_band_must_arrive_as_exact_zero(self):
        move = PicoHybridMove(session=FakeSession(), gyro_transform=lambda value: value)
        obs = observation()
        obs.user_input.foot_target = {
            "left": (0.001, 0.0, 0.0),
            "right": (0.0, 0.0, 0.0),
        }
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "floor-band"):
            move._body_targets(obs)

        obs.user_input.foot_target["left"] = (0.0, 0.0, 0.0025)
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "floor-band"):
            move._body_targets(obs)

        obs.user_input.foot_target["left"] = (0.0, 0.0, 0.0)
        feet, _hands = move._body_targets(obs)
        self.assertEqual(feet, [0.0] * 6)

    def test_nonfinite_policy_output_fails_closed(self):
        output = np.zeros((1, 18), dtype=np.float32)
        output[0, 0] = math.nan
        session = FakeSession()
        move = PicoHybridMove(session=session, gyro_transform=lambda value: value)
        session.output = output
        obs = observation()
        move.on_start(obs, MotorCommand())
        with self.assertRaises(PicoHybridPolicyRuntimeError):
            move.step(obs, MotorCommand())

    def test_release_ends_at_next_inactive_motor_command_without_discontinuity(self):
        move = PicoHybridMove(session=FakeSession(), gyro_transform=lambda value: value)
        obs = observation(10.0)
        move.on_start(obs, MotorCommand())
        move.state = MoveState.STOPPING
        obs.robot_state.motor_positions = {
            name: PICO_TELEOP_HOME_POSE[name] + 0.2 for name in MOTOR_TO_ID
        }
        first = MotorCommand()
        move.on_stop(obs, first)
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            measured = PICO_TELEOP_HOME_POSE[name] + 0.2
            expected = max(
                EXPECTED_SOFT_JOINT_POS_LOWER[index],
                min(EXPECTED_SOFT_JOINT_POS_UPPER[index], measured),
            )
            self.assertEqual(first.target_angles[name], expected)

        obs.robot_state.time_s = 10.8
        final = MotorCommand()
        move.on_stop(obs, final)
        self.assertEqual(move.state, MoveState.INACTIVE)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(final.target_angles[name], NEUTRAL_POSE[name])
        next_inactive_tick = MotorCommand()
        self.assertEqual(final.target_angles, next_inactive_tick.target_angles)
        self.assertEqual(
            final.target_angles["left_shoulder_pitch"], math.radians(10.0)
        )
        self.assertEqual(
            final.target_angles["right_shoulder_pitch"], math.radians(10.0)
        )

    def test_getup_cancels_release_interpolation(self):
        move = PicoHybridMove(session=FakeSession(), gyro_transform=lambda value: value)
        obs = observation(10.0)
        move.on_start(obs, MotorCommand())

        move.state = MoveState.STOPPING
        obs.user_input.active_moves.add("getup")
        move.on_stop(obs, MotorCommand())
        self.assertEqual(move.state, MoveState.INACTIVE)

    def test_start_rejects_nonfinite_motor_position_before_gain_write(self):
        controller = FakeController()
        move = PicoHybridMove(
            controller=controller,
            session=FakeSession(),
            gyro_transform=lambda value: value,
        )
        obs = observation()
        obs.robot_state.motor_positions[OBSERVATION_DOF_ORDER[0]] = math.nan
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "non-finite"):
            move.on_start(obs, MotorCommand())
        self.assertEqual(controller.kp_writes, [])
        self.assertEqual(move.state, MoveState.INACTIVE)

    def test_stop_rejects_nonfinite_motor_position_before_interpolation(self):
        move = PicoHybridMove(session=FakeSession(), gyro_transform=lambda value: value)
        obs = observation()
        move.on_start(obs, MotorCommand())
        move.state = MoveState.STOPPING
        obs.robot_state.motor_positions[OBSERVATION_DOF_ORDER[-1]] = math.inf
        with self.assertRaisesRegex(PicoHybridPolicyRuntimeError, "non-finite"):
            move.on_stop(obs, MotorCommand())
        self.assertIsNone(move._stop_start_time_s)
        self.assertEqual(move._stop_start_angles, {})

    def test_mount_transform_is_finite_and_norm_preserving(self):
        transformed = sensor_gyro_to_body((1.0, 2.0, 3.0))
        self.assertTrue(all(math.isfinite(value) for value in transformed))
        self.assertAlmostEqual(
            sum(value * value for value in transformed),
            1.0 + 4.0 + 9.0,
        )

    def test_mount_transform_sensor_axes_map_to_expected_body_axes(self):
        expected = (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
        )
        for sensor_axis, body_axis in zip(np.eye(3), expected, strict=True):
            with self.subTest(sensor_axis=sensor_axis):
                np.testing.assert_allclose(
                    sensor_gyro_to_body(sensor_axis), body_axis, atol=1e-12
                )

    def test_mount_contract_matches_mjcf_and_orientation_conversion(self):
        model_path = Path(__file__).resolve().parents[1] / "src/model/mjcf/robot.xml"
        root = ET.parse(model_path).getroot()
        imu_site = root.find(".//site[@name='imu']")
        self.assertIsNotNone(imu_site)
        assert imu_site is not None
        site_quat = tuple(float(value) for value in imu_site.attrib["quat"].split())
        np.testing.assert_allclose(site_quat, IMU_MOUNT_QUAT, atol=1e-12)

        # At identity trunk orientation the world-from-sensor quaternion equals
        # q_body_sensor from the MJCF site. Removing that fixed mount must recover
        # identity world-from-body, matching the gyro's sensor-to-body direction.
        np.testing.assert_allclose(
            imu_quat_to_body(site_quat),
            (1.0, 0.0, 0.0, 0.0),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
