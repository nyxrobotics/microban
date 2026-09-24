import json
import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

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
    EXPECTED_ACTION_DEFAULT_JOINT_POS,
    EXPECTED_ACTION_DISTRIBUTION_SEMANTICS,
    EXPECTED_ACTION_SCALE,
    EXPECTED_ACTOR_DEFAULT_INTERIOR_EPSILON_RAD,
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
    EXPECTED_OBSERVATION_TERMS,
    EXPECTED_PREVIOUS_ACTION_SEMANTICS,
    EXPECTED_RAW_ACTION_SOFT_LOWER,
    EXPECTED_RAW_ACTION_SOFT_UPPER,
    EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_LOWER,
    EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_SEMANTICS,
    EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_UPPER,
    EXPECTED_SOFT_JOINT_POS_LOWER,
    EXPECTED_SOFT_JOINT_POS_UPPER,
    PicoHybridMove,
    PicoHybridPolicyContractError,
    PicoHybridPolicyRuntimeError,
    onnxruntime_compatibility_smoke_inputs,
    sensor_gyro_to_body,
    validate_onnxruntime_compatibility,
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


class _Io:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _Metadata:
    def __init__(self, values):
        self.custom_metadata_map = values


def _csv(values):
    return ",".join(str(value) for value in values)


def _metadata_csv(values):
    return ",".join(f"{value:.3f}" for value in values)


def valid_metadata():
    observation_defaults = [NEUTRAL_POSE[name] for name in OBSERVATION_JOINTS]
    return {
        "policy_type": "microban_pico_hybrid_teleop",
        "checkpoint_filename": "model_14999.pt",
        "checkpoint_iteration": "14999",
        "checkpoint_iteration_semantics": (
            "zero_based_completed_update_index_from_model_filename"
        ),
        "checkpoint_completed_updates": "15000",
        "checkpoint_sha256": "0123456789abcdef" * 4,
        "onnx_parity_gate_version": "1",
        "onnx_parity_verified": "true",
        "onnx_parity_runtime": "onnx.reference.ReferenceEvaluator",
        "onnx_parity_seed": "20260924",
        "onnx_parity_sample_count": "16",
        "onnx_parity_atol": "1e-05",
        "onnx_parity_rtol": "0.0001",
        "microban_teleop_training_contract_version": "7",
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


class FakeSession:
    def __init__(self, metadata=None, output=None):
        self.metadata = valid_metadata() if metadata is None else metadata
        self.output = np.zeros((1, 18), dtype=np.float32) if output is None else output
        self.last_feed = None
        self.run_count = 0

    def get_inputs(self):
        return [_Io("obs", [1, 83])]

    def get_outputs(self):
        return [_Io("actions", [1, 18])]

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
    positions = {name: NEUTRAL_POSE[name] for name in MOTOR_TO_ID}
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

    def test_onnxruntime_compatibility_requires_v7_transform_envelope(self):
        output = np.zeros((1, 18), dtype=np.float64)
        output[0, 0] = (
            EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER[0]
            + EXPECTED_ACTOR_RAW_ACTION_UPPER[0]
        ) / 2.0
        with self.assertRaisesRegex(
            PicoHybridPolicyRuntimeError, "deterministic transform envelope"
        ):
            validate_onnxruntime_compatibility(FakeSession(output=output), "obs")

    def test_v7_latent_envelope_is_finite_nested_and_has_ten_sigma_margin(self):
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

    def test_contract_rejects_v1_through_v6_or_missing_effective_action_contract(self):
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

    def test_contract_rejects_missing_or_widened_both_feet_support(self):
        mutations = {
            "missing_lower": ("simultaneous_both_feet_target_lower", None),
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

    def test_contract_requires_exact_v7_bounded_actor_metadata(self):
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
            "wrong_std_min_abs_max": ("actor_latent_std_min_abs_max", "0.02"),
            "wrong_std_min_divisor": (
                "actor_latent_std_min_envelope_divisor",
                "63",
            ),
            "wrong_std_abs_max": ("actor_latent_std_abs_max", "0.16"),
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
                    "independently derived Microban v7 contract",
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
        # Mutate the fake after the load-time corpus gate. Contract v7 treats a
        # graph that escapes its physical transform as a fault, not an action
        # to silently saturate.
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

    def test_release_returns_to_neutral_and_getup_cancels(self):
        move = PicoHybridMove(session=FakeSession(), gyro_transform=lambda value: value)
        obs = observation(10.0)
        move.on_start(obs, MotorCommand())
        move.state = MoveState.STOPPING
        obs.robot_state.motor_positions = {
            name: NEUTRAL_POSE[name] + 0.2 for name in MOTOR_TO_ID
        }
        first = MotorCommand()
        move.on_stop(obs, first)
        for name in OBSERVATION_DOF_ORDER:
            self.assertAlmostEqual(first.target_angles[name], NEUTRAL_POSE[name] + 0.2)

        obs.robot_state.time_s = 10.8
        final = MotorCommand()
        move.on_stop(obs, final)
        self.assertEqual(move.state, MoveState.INACTIVE)
        for name in OBSERVATION_DOF_ORDER:
            self.assertAlmostEqual(final.target_angles[name], NEUTRAL_POSE[name])

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
