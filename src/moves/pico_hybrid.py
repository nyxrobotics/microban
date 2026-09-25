# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Opt-in runtime for the Microban PICO hybrid teleoperation policy.

This move deliberately has a stricter contract than :mod:`moves.walk`.  A
PICO policy observes all 21 encoders and body-tracking targets but controls
only the 18 non-head joints.  Loading a legacy walking policy, or an export
whose observation order changed, therefore fails before motor torque is used.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from constants import (
    IMU_MOUNT_QUAT,
    KP_DEFAULT,
    KP_RL,
    MOTOR_TO_ID,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
)
from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from observer import Observation

AGENT_NAME = "pico_teleop.onnx"
EXPECTED_POLICY_TYPE = "microban_pico_hybrid_teleop"
EXPECTED_TRAINING_CONTRACT_VERSION = "10"
EXPECTED_V12_TRAINING_CONTRACT_VERSION = "12"
# Keep the recipe in one deployment-side constant: a deliberately promoted
# training recipe then requires one reviewed line change here.  It must match
# the exporter exactly; accepting a different marker would attach current
# runtime semantics to weights trained under another reward/config recipe.
EXPECTED_ACTOR_INITIALIZATION = "full_state_v9_model1499_to_v10_fixed_lr_v1"
EXPECTED_RECIPE_REVISION = (
    "v10_v9_model1499_full_state_migration_fixed_lr_pico_curriculum_v1"
)
EXPECTED_SAFE_VELOCITY_RECIPE_REVISION = (
    "scratch_bounded_inward_shoulder_sagittal_bodyprogress_v9"
)
EXPECTED_SAFE_VELOCITY_RECEIPT_SCHEMA_VERSION = 2
EXPECTED_SAFE_VELOCITY_ACCEPTANCE_GATE = "microban_safe_velocity_fixed_forward_v3"
EXPECTED_SAFE_VELOCITY_BOOTSTRAP_MAPPING_VERSION = (
    "bounded_raw_safe_velocity_63_to_teleop_83_v1"
)
EXPECTED_SCHEMA_VERSION = "2"
EXPECTED_PREVIOUS_ACTION_SEMANTICS = (
    "effective_action_after_absolute_target_soft_clip_in_raw_delta_coordinates"
)
EXPECTED_ACTION_DISTRIBUTION_SEMANTICS = (
    "diagonal_normal_ppo_latent_stored_exactly_then_per_joint_"
    "asymmetric_zero_anchored_arctan_environment_transform_with_"
    "operational_envelope_v1"
)
EXPECTED_ACTOR_TARGET_GUARD_MARGIN_RATIO = 0.05
EXPECTED_ACTOR_DEFAULT_INTERIOR_EPSILON_RAD = 1.0e-4
EXPECTED_ACTOR_LATENT_OPERATIONAL_SCALE_MULTIPLIER = 1024.0
EXPECTED_ACTOR_LATENT_OPERATIONAL_ABS_MAX = 32.0
EXPECTED_ACTOR_LATENT_MEAN_FRACTION = 3.0 / 8.0
EXPECTED_ACTOR_LATENT_STD_MIN_ABS_MAX = 0.025
EXPECTED_ACTOR_LATENT_STD_MIN_ENVELOPE_DIVISOR = 64.0
EXPECTED_ACTOR_LATENT_STD_ABS_MAX = 1.0
EXPECTED_ACTOR_LATENT_STD_ENVELOPE_DIVISOR = 16.0
# MjLab applies the soft-limit factor and action offset in float32 after MuJoCo
# has resolved the XML.  Reassociating those operations from Microban's already
# compiled soft limits can differ by a few float32 epsilons through cancellation.
# This remains far tighter than three-decimal legacy metadata (and below
# 0.000028 degrees for scale 1.0) while accepting both legitimate operation
# orders.
EXPECTED_ACTION_BOUND_FLOAT32_ATOL = 4.0 * float(np.finfo(np.float32).eps)
EXPECTED_OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "foot_target",
    "hand_target",
)
EXPECTED_OBSERVATION_WIDTH = 83
EXPECTED_ACTION_WIDTH = 18

# Contract v12 deliberately retains the proven, normalized legacy velocity
# actor instead of the bounded v10 action transform.  These identities are
# pinned on the robot so arbitrary 83-input ONNX files cannot opt themselves
# into the raw-action execution path by adding a version string.
EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256 = (
    "b0bcdadac39716be784207dd6b2b93157162a3e80650e23c05f490c400b9e141"
)
EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_ITERATION = 14_999
EXPECTED_V12_LEGACY_PROBE_SHA256 = (
    "f51378d59ff4d68fb1185a91eb2a863749e5c7be6ec4cd0ab4a0b08f1565e69d"
)
EXPECTED_V12_BOOTSTRAP_PROVENANCE_SCHEMA_VERSION = 1
EXPECTED_V12_BOOTSTRAP_MAPPING_VERSION = (
    "normalized_legacy_velocity_63_to_teleop83_reachable_fk_elbow_minus10_v4"
)
EXPECTED_V12_RECIPE_REVISION = (
    "legacy_velocity_model14999_staged_mask_reachable_fk_elbow_minus10_raw_actions_v5"
)
EXPECTED_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION = (
    "freeze_extra_to7000_then_hmd_hand_to10000_then_all_v1"
)
EXPECTED_V12_ACTOR_TOPOLOGY = (83, 512, 256, 128, 18)
EXPECTED_V12_NORMALIZER_EPS = 1.0e-2
EXPECTED_V12_NORMALIZER_SEMANTICS = (
    "frozen_source63_identity_hmd_flags_reachable_fk_target_scaling_v3"
)
EXPECTED_V12_PREVIOUS_ACTION_SEMANTICS = "raw_actor_output"
EXPECTED_V12_ACTION_CLIP_SEMANTICS = "none"
EXPECTED_V12_ACTION_DISTRIBUTION_SEMANTICS = "unbounded_gaussian_deterministic_mean_raw"
EXPECTED_V12_RUNTIME_ACTION_SEMANTICS = (
    "raw_unbounded_default_plus_scale_no_target_clip_v1"
)
# This guard is deliberately outside the learned-policy/recurrence contract
# above.  V12 still observes its exact raw actor output on the next tick, while
# the actuator-facing MotorCommand is the continuous saturation of the derived
# absolute target to Microban's compiled physical soft limits.
PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS = (
    "compiled_soft_limit_continuous_clamp_preserve_policy_recurrence_v1"
)
EXPECTED_V12_FINAL_CHECKPOINT_ITERATION = 14_999
EXPECTED_V12_FINAL_COMPLETED_UPDATES = 15_000
EXPECTED_V12_STAGE_GATE = "microban_teleop_v12_stage"
EXPECTED_V12_LOCOMOTION_GATE = "microban_teleop_v12_neutral_locomotion_9x300"
EXPECTED_V12_ONNX_GATE = "microban_teleop_v12_checkpoint_onnx"
EXPECTED_V12_TRACKING_PROFILE = "full_body_reachable_performance_perturbation_v2"
EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG = 5.0
EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD = math.radians(
    EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG
)
EXPECTED_V12_COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD = 1.0e-7
EXPECTED_V12_RAW_ACTION_ENVELOPE_SCHEMA_VERSION = 1
EXPECTED_V12_RAW_ACTION_GUARD_FORMULA = (
    "max(v12_absmax,source_absmax+delta_absmax)*multiplier"
)
EXPECTED_V12_RAW_ACTION_GUARD_MULTIPLIER = 2.0
EXPECTED_V12_RAW_ACTION_GUARD_SEMANTICS = (
    "finite_float32_then_per_joint_absmax_else_same_cycle_legacy_fallback_v1"
)
EXPECTED_V12_PARITY_SEED = 20260925
EXPECTED_V12_PARITY_SAMPLE_COUNT = 64
EXPECTED_V12_PARITY_ATOL = 2.0e-5
EXPECTED_V12_OBSERVATION_JOINT_NAMES = (
    "head",
    "neck_roll",
    "neck_pitch",
    *OBSERVATION_DOF_ORDER,
)
EXPECTED_V12_SOURCE_TO_TARGET_COLUMNS = (
    *((index, index) for index in range(6)),
    *((index, index + 3) for index in range(6, 24)),
    *((index, index + 6) for index in range(24, 42)),
    *((index, index + 6) for index in range(42, 60)),
    *((index, index + 6) for index in range(60, 63)),
)
EXPECTED_V12_EXTRA_OBSERVATION_COLUMNS = (
    6,
    7,
    8,
    27,
    28,
    29,
    *range(69, 83),
)
# This is deliberately a complete, robot-local copy of the exporter contract.
# Merely trusting the model's revision label would let a differently scaled or
# differently framed hand policy opt itself into the raw-action v12 runtime.
EXPECTED_V12_HAND_TARGET_FK = {
    "revision": "microban_robot_xml_arm_fk_reachable_box_elbow_upper_minus10_v2",
    "side_order": ["left", "right"],
    "joint_order": ["shoulder_pitch", "shoulder_roll", "elbow"],
    "joint_lower_deg": [[-25.0, 10.0, -50.0], [-25.0, -30.0, -50.0]],
    "joint_upper_deg": [[25.0, 30.0, -10.0], [25.0, -10.0, -10.0]],
    "home_joint_deg": [[0.0, 10.0, -20.0], [0.0, -10.0, -20.0]],
    "bound_grid_points_per_axis": 401,
    "offset_aabb_min_m": [
        [-0.06120170602356862, -0.0034550417440758485, -0.0035619649312883805],
        [-0.06120170602356864, -0.0387512193701912, -0.0035619649312883944],
    ],
    "offset_aabb_max_m": [
        [0.06289464331528255, 0.0387512193701912, 0.060477220857479266],
        [0.06289464331528258, 0.0034550417440758485, 0.060477220857479225],
    ],
    "normalizer_abs_bound_m": [0.063, 0.0388, 0.0605],
    "wire_abs_bound_m": [0.08, 0.08, 0.08],
    "runtime_validated_abs_limit_m": [0.064, 0.064, 0.064],
    "evaluation_joint_degrees": [
        ["F", [-25.0, 25.0, -50.0]],
        ["B", [25.0, 20.0, -10.0]],
        ["f", [-12.0, 18.0, -32.0]],
        ["b", [12.0, 18.0, -32.0]],
    ],
    "evaluation_offsets_m": [
        [
            "F",
            [
                [0.05965182377803401, 0.02001299541823158, 0.05695429454214103],
                [0.059651823778034005, -0.02001299541823158, 0.056954294542141],
            ],
        ],
        [
            "B",
            [
                [-0.05889031574552025, 0.020262900077370374, 0.00788636819595702],
                [-0.058890315745520276, -0.020262900077370388, 0.007886368195956998],
            ],
        ],
        [
            "f",
            [
                [0.03317253316144149, 0.013564446512561182, 0.01946788065760361],
                [0.0331725331614415, -0.013564446512561182, 0.019467880657603583],
            ],
        ],
        [
            "b",
            [
                [-0.009540835008328632, 0.013564446512561182, 0.004701156748151122],
                [-0.009540835008328644, -0.013564446512561182, 0.0047011567481511154],
            ],
        ],
    ],
    "source": "src/mjlab_microban/robot/microban/robot.xml",
    "sampling": "uniform_independent_joint_box_then_exact_fk_offset_from_home",
}
EXPECTED_ONNX_PARITY_GATE_VERSION = "1"
EXPECTED_ONNX_PARITY_RUNTIME = "onnx.reference.ReferenceEvaluator"
EXPECTED_ONNX_PARITY_SEED = 20260924
EXPECTED_ONNX_PARITY_SAMPLE_COUNT = 16
EXPECTED_ONNX_PARITY_ATOL = 1e-5
EXPECTED_ONNX_PARITY_RTOL = 1e-4
EXPECTED_TRAINING_PROVENANCE_SCHEMA_VERSION = 2
EXPECTED_TRAINING_PROVENANCE_MODE = "canonical_v10_stage"
EXPECTED_FINAL_TRAINING_STAGE_START_BOUNDARY = 10_000
EXPECTED_FINAL_TRAINING_STAGE_TARGET_BOUNDARY = 15_000
EXPECTED_MIGRATION_SOURCE_CHECKPOINT_SHA256 = (
    "de8b6139872179679a16d72f3007f6d96cf65c97fa88565841eaa5f89511a65f"
)
EXPECTED_MIGRATION_SOURCE_CHECKPOINT_ITERATION = 1_499
EXPECTED_MIGRATION_SOURCE_TRAINING_PROVENANCE_SHA256 = (
    "f09f5580f03d3e38deef4916db7aea3bd8b1f683dc02a75079d24dd17923fce9"
)
EXPECTED_MIGRATION_SOURCE_TREE_SHA256 = (
    "61a9fc7b1fe10436c0f033f89710e33e9e5470716d94794f110731c18e7d792a"
)
EXPECTED_MIGRATION_SOURCE_GATE_SHA256 = (
    "acb2e39411155d70aed2b18561a243bd8ad20eef56e0942d2dafbb4f96f39b7c"
)
EXPECTED_MIGRATION_STATE_TRANSFER = (
    "actor_critic_optimizer_moments_iteration_common_step_v1"
)
EXPECTED_MIGRATION_SOURCE_OPTIMIZER_LEARNING_RATE = 7.593750000000002e-05
EXPECTED_TRAINING_FIXED_LEARNING_RATE = 1.0e-5
EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_SHA256 = (
    "416a8b16f7f7980822e4e1df81ffaf9515bc18a246e6fc257405a2c46ceece93"
)
EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_ITERATION = 500
EXPECTED_SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SHA256 = (
    "e68701b11774dd30c8e45a2fd89614a2e4423a9486d01a0d936f0fa6fb760492"
)
EXPECTED_ACCEPTANCE_RECEIPT_SCHEMA_VERSION = 3
EXPECTED_ACCEPTANCE_EVALUATOR_REVISION = "microban_teleop_deterministic_evaluator_v10_1"
EXPECTED_ACCEPTANCE_REVISION = "microban_teleop_acceptance_v10_1"
EXPECTED_ACCEPTANCE_NOMINAL_REPORT_COUNT = 3
EXPECTED_ACCEPTANCE_MOVING_HMD_REPORT_COUNT = 3

_CHECKPOINT_FILENAME_RE = re.compile(r"model_(0|[1-9][0-9]*)\.pt\Z")
_CHECKPOINT_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# The exporter serializes numeric metadata to three decimal places.  Keep the
# full-precision, robot-side action contract here and only use the serialized
# form when checking the ONNX metadata.  Inference always uses these values,
# never model-provided limits, so altered metadata cannot widen motor targets.
# The PICO policy was trained with shoulder pitch at 0 degrees.  Keep this
# deployment contract local: NEUTRAL_POSE is shared by unrelated legacy moves.
# Its +10-degree shoulder pitch originated in main-repository commit f27a9e29;
# the robot MJCF supplies only joint ranges, not that HOME value.  See the
# runtime guide for the separate training-history and ONNX-metadata evidence.
PICO_TELEOP_HOME_POSE = {
    **NEUTRAL_POSE,
    "left_shoulder_pitch": 0.0,
    "right_shoulder_pitch": 0.0,
}
EXPECTED_ACTION_DEFAULT_JOINT_POS = tuple(
    float(PICO_TELEOP_HOME_POSE[name]) for name in OBSERVATION_DOF_ORDER
)
EXPECTED_ACTION_SCALE = (1.0,) * EXPECTED_ACTION_WIDTH
# Midpoint-centered 0.9 soft limits derived from Microban's deployed MJCF joint
# ranges, in OBSERVATION_DOF_ORDER.  These are intentionally independent of the
# ONNX file and must change in review with the robot model/training contract.
EXPECTED_SOFT_JOINT_POS_LOWER = (
    -2.82743338823,
    -2.98451302091,
    -1.96349540849,
    -3.92699081699,
    -0.3926988,
    -1.41371669412,
    -0.628318530718,
    -1.46171324855,
    -0.549778714378,
    -2.82743338823,
    0.157079632679,
    -1.96349540849,
    -0.785398163397,
    -0.3926988,
    -1.41371669412,
    -0.628318530718,
    -1.46171324855,
    -0.549778714378,
)
EXPECTED_SOFT_JOINT_POS_UPPER = (
    2.82743338823,
    -0.157079632679,
    1.96349540849,
    0.785398163397,
    0.3926988,
    1.41371669412,
    2.19911485751,
    0.501782159948,
    0.549778714378,
    2.82743338823,
    2.98451302091,
    1.96349540849,
    3.92699081699,
    0.3926988,
    1.41371669412,
    2.19911485751,
    0.501782159948,
    0.549778714378,
)


def _derive_expected_action_bound_contract() -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Independently derive the current raw and guarded actor bounds.

    MjLab resolves the robot defaults, soft limits and action parameters as
    float32 tensors before exporting the four full-precision JSON vectors.  Do
    the same IEEE-754 operations here from Microban's compiled-in constants;
    no model-provided value participates in this derivation.
    """

    defaults = np.asarray(EXPECTED_ACTION_DEFAULT_JOINT_POS, dtype=np.float32)
    scales = np.asarray(EXPECTED_ACTION_SCALE, dtype=np.float32)
    soft_lower = np.asarray(EXPECTED_SOFT_JOINT_POS_LOWER, dtype=np.float32)
    soft_upper = np.asarray(EXPECTED_SOFT_JOINT_POS_UPPER, dtype=np.float32)
    raw_soft_lower = ((soft_lower - defaults) / scales).tolist()
    raw_soft_upper = ((soft_upper - defaults) / scales).tolist()

    actor_lower: list[float] = []
    actor_upper: list[float] = []
    for default, lower, upper, scale in zip(
        defaults.tolist(),
        soft_lower.tolist(),
        soft_upper.tolist(),
        scales.tolist(),
        strict=True,
    ):
        span = upper - lower
        epsilon = min(
            EXPECTED_ACTOR_DEFAULT_INTERIOR_EPSILON_RAD,
            0.5 * (default - lower),
            0.5 * (upper - default),
        )
        target_lower = min(
            default - epsilon,
            lower + EXPECTED_ACTOR_TARGET_GUARD_MARGIN_RATIO * span,
        )
        target_upper = max(
            default + epsilon,
            upper - EXPECTED_ACTOR_TARGET_GUARD_MARGIN_RATIO * span,
        )
        actor_lower.append((target_lower - default) / scale)
        actor_upper.append((target_upper - default) / scale)

    contract = (
        tuple(float(value) for value in raw_soft_lower),
        tuple(float(value) for value in raw_soft_upper),
        tuple(actor_lower),
        tuple(actor_upper),
    )
    if not all(
        soft_lo < actor_lo < 0.0 < actor_hi < soft_hi
        for soft_lo, soft_hi, actor_lo, actor_hi in zip(
            *contract,
            strict=True,
        )
    ):
        raise RuntimeError("compiled-in actor bounds are not strictly guarded")
    return contract


(
    EXPECTED_RAW_ACTION_SOFT_LOWER,
    EXPECTED_RAW_ACTION_SOFT_UPPER,
    EXPECTED_ACTOR_RAW_ACTION_LOWER,
    EXPECTED_ACTOR_RAW_ACTION_UPPER,
) = _derive_expected_action_bound_contract()


def _derive_expected_latent_envelope_contract() -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Independently derive the current finite PPO-latent operating envelope.

    Training performs these operations on float32 actor-bound tensors.  Keep
    the same precision here so the robot validates the declared scalar
    transform contract against the same per-joint geometry.  The ONNX graph
    emits the final physical raw action; this envelope is validation data and
    is never applied as a second runtime transform.
    """

    actor_lower = np.asarray(EXPECTED_ACTOR_RAW_ACTION_LOWER, dtype=np.float32)
    actor_upper = np.asarray(EXPECTED_ACTOR_RAW_ACTION_UPPER, dtype=np.float32)
    absolute_cap = np.full_like(actor_lower, EXPECTED_ACTOR_LATENT_OPERATIONAL_ABS_MAX)
    operational_lower = -np.minimum(
        -actor_lower * np.float32(EXPECTED_ACTOR_LATENT_OPERATIONAL_SCALE_MULTIPLIER),
        absolute_cap,
    )
    operational_upper = np.minimum(
        actor_upper * np.float32(EXPECTED_ACTOR_LATENT_OPERATIONAL_SCALE_MULTIPLIER),
        absolute_cap,
    )
    mean_lower = operational_lower * np.float32(EXPECTED_ACTOR_LATENT_MEAN_FRACTION)
    mean_upper = operational_upper * np.float32(EXPECTED_ACTOR_LATENT_MEAN_FRACTION)
    closest_side = np.minimum(-operational_lower, operational_upper)
    min_std = np.minimum(
        np.full_like(actor_lower, EXPECTED_ACTOR_LATENT_STD_MIN_ABS_MAX),
        closest_side / np.float32(EXPECTED_ACTOR_LATENT_STD_MIN_ENVELOPE_DIVISOR),
    )
    max_std = np.minimum(
        np.full_like(actor_lower, EXPECTED_ACTOR_LATENT_STD_ABS_MAX),
        closest_side / np.float32(EXPECTED_ACTOR_LATENT_STD_ENVELOPE_DIVISOR),
    )

    contract = tuple(
        tuple(float(value) for value in values)
        for values in (
            operational_lower,
            operational_upper,
            mean_lower,
            mean_upper,
            min_std,
            max_std,
        )
    )
    if not all(
        op_lo < mean_lo < 0.0 < mean_hi < op_hi
        and 0.0 < std_min < std_max
        and mean_lo - 10.0 * std_max >= op_lo
        and mean_hi + 10.0 * std_max <= op_hi
        for op_lo, op_hi, mean_lo, mean_hi, std_min, std_max in zip(
            *contract,
            strict=True,
        )
    ):
        raise RuntimeError("compiled-in latent envelope is inconsistent")
    return contract


(
    EXPECTED_ACTOR_LATENT_OPERATIONAL_LOWER,
    EXPECTED_ACTOR_LATENT_OPERATIONAL_UPPER,
    EXPECTED_ACTOR_LATENT_MEAN_LOWER,
    EXPECTED_ACTOR_LATENT_MEAN_UPPER,
    EXPECTED_ACTOR_LATENT_MIN_STD,
    EXPECTED_ACTOR_LATENT_MAX_STD,
) = _derive_expected_latent_envelope_contract()


def _asymmetric_arctan_transform_float32(
    latent: Sequence[float],
) -> tuple[float, ...]:
    """Mirror the export graph's one physical-action transform in float32."""

    latent_values = np.asarray(latent, dtype=np.float32)
    actor_lower = np.asarray(EXPECTED_ACTOR_RAW_ACTION_LOWER, dtype=np.float32)
    actor_upper = np.asarray(EXPECTED_ACTOR_RAW_ACTION_UPPER, dtype=np.float32)
    scale = np.where(latent_values >= 0.0, actor_upper, -actor_lower)
    transformed = (
        np.float32(2.0)
        * scale
        / np.float32(math.pi)
        * np.arctan(np.float32(math.pi) * latent_values / (np.float32(2.0) * scale))
    )
    return tuple(float(value) for value in transformed)


_deterministic_lower = np.asarray(
    _asymmetric_arctan_transform_float32(EXPECTED_ACTOR_LATENT_MEAN_LOWER),
    dtype=np.float32,
)
_deterministic_upper = np.asarray(
    _asymmetric_arctan_transform_float32(EXPECTED_ACTOR_LATENT_MEAN_UPPER),
    dtype=np.float32,
)
# NumPy, PyTorch and ONNX Runtime can differ by one last-place float32 bit in
# the transcendental kernel.  Move the inclusive validation envelope one ULP
# outward; this is many orders of magnitude inside the guarded actor bounds.
EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER = tuple(
    float(value)
    for value in np.nextafter(
        _deterministic_lower,
        np.full_like(_deterministic_lower, -np.inf),
    )
)
EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER = tuple(
    float(value)
    for value in np.nextafter(
        _deterministic_upper,
        np.full_like(_deterministic_upper, np.inf),
    )
)
if not all(
    actor_lo < output_lo < 0.0 < output_hi < actor_hi
    for actor_lo, actor_hi, output_lo, output_hi in zip(
        EXPECTED_ACTOR_RAW_ACTION_LOWER,
        EXPECTED_ACTOR_RAW_ACTION_UPPER,
        EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER,
        EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER,
        strict=True,
    )
):
    raise RuntimeError("compiled-in deterministic transform envelope is unsafe")

# These ranges are the command support used by Mjlab-Teleop-Microban.  The
# exporter also records them in the policy metadata; these constants are only
# the expected contract used to reject a mismatched model.
EXPECTED_FOOT_TARGET_LOWER = (-0.03, -0.03, 0.0) * 2
EXPECTED_FOOT_TARGET_UPPER = (0.03, 0.03, 0.05) * 2
EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_LOWER = (-0.01, -0.01, 0.0) * 2
EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_UPPER = (0.01, 0.01, 0.02) * 2
EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_SEMANTICS = (
    "left_and_right_nonzero_offsets_use_conservative_stationary_training_support"
)
LIVE_BODY_TARGET_SAFETY_MARGIN = 0.8
SUPPORT_FOOT_FLOOR_BAND_M = 0.0025
EXPECTED_HAND_TARGET_LOWER = (-0.08, -0.08, -0.08) * 2
EXPECTED_HAND_TARGET_UPPER = (0.08, 0.08, 0.08) * 2


class PicoHybridPolicyContractError(ValueError):
    """The ONNX file does not describe the exact safe runtime contract."""


class PicoHybridPolicyRuntimeError(RuntimeError):
    """Inference produced a result that is unsafe to send to actuators."""


def _split_csv(value: str | None, name: str) -> tuple[str, ...]:
    if not value:
        raise PicoHybridPolicyContractError(f"missing ONNX metadata: {name}")
    result = tuple(item.strip() for item in value.split(","))
    if not result or any(not item for item in result):
        raise PicoHybridPolicyContractError(f"invalid ONNX metadata: {name}")
    return result


def _float_csv(value: str | None, name: str, count: int) -> tuple[float, ...]:
    items = _split_csv(value, name)
    if len(items) != count:
        raise PicoHybridPolicyContractError(
            f"{name} has {len(items)} values; expected {count}"
        )
    try:
        result = tuple(float(item) for item in items)
    except ValueError as exc:
        raise PicoHybridPolicyContractError(f"{name} must contain numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise PicoHybridPolicyContractError(f"{name} contains a non-finite value")
    return result


def _float_json_vector(value: str | None, name: str, count: int) -> tuple[float, ...]:
    """Parse an exact-width finite-number vector from strict JSON metadata."""

    if value is None:
        raise PicoHybridPolicyContractError(f"missing ONNX metadata: {name}")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant}")

    try:
        decoded = json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PicoHybridPolicyContractError(
            f"{name} must be a strict JSON array"
        ) from exc
    if not isinstance(decoded, list) or len(decoded) != count:
        actual_count = len(decoded) if isinstance(decoded, list) else "not an array"
        raise PicoHybridPolicyContractError(
            f"{name} has {actual_count} values; expected {count}"
        )

    result: list[float] = []
    for item in decoded:
        # JSON booleans are Python integers, so reject them explicitly.
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PicoHybridPolicyContractError(
                f"{name} must contain only JSON numbers"
            )
        numeric = float(item)
        if not math.isfinite(numeric):
            raise PicoHybridPolicyContractError(f"{name} contains a non-finite value")
        result.append(numeric)
    return tuple(result)


def _require_tight_json_vector(
    name: str,
    actual: Sequence[float],
    expected: Sequence[float],
) -> None:
    """Require unrounded JSON values to tightly match the robot derivation."""

    if all(
        math.isclose(
            received,
            required,
            rel_tol=0.0,
            abs_tol=EXPECTED_ACTION_BOUND_FLOAT32_ATOL,
        )
        for received, required in zip(actual, expected, strict=True)
    ):
        return
    mismatch = next(
        (
            index
            for index, (received, required) in enumerate(
                zip(actual, expected, strict=True)
            )
            if not math.isclose(
                received,
                required,
                rel_tol=0.0,
                abs_tol=EXPECTED_ACTION_BOUND_FLOAT32_ATOL,
            )
        ),
        None,
    )
    raise PicoHybridPolicyContractError(
        f"{name} does not match the independently derived Microban contract"
        + (
            ""
            if mismatch is None
            else (
                f" at action index {mismatch}: got {actual[mismatch]!r}, "
                f"expected {expected[mismatch]!r}"
            )
        )
    )


def _require_exact_finite_scalar(
    metadata: Mapping[str, str], name: str, expected: float
) -> float:
    value = metadata.get(name)
    if value is None:
        raise PicoHybridPolicyContractError(f"missing ONNX metadata: {name}")
    try:
        actual = float(value)
    except (TypeError, ValueError) as exc:
        raise PicoHybridPolicyContractError(f"{name} must be numeric") from exc
    if not math.isfinite(actual) or actual != expected:
        raise PicoHybridPolicyContractError(
            f"{name} does not match the fixed Microban contract"
        )
    return actual


def _canonical_nonnegative_int(value: str | None, name: str) -> int:
    if value is None or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise PicoHybridPolicyContractError(
            f"{name} must be a canonical non-negative integer"
        )
    return int(value)


def _require_lowercase_sha256(metadata: Mapping[str, str], name: str) -> str:
    value = metadata.get(name, "")
    if _CHECKPOINT_SHA256_RE.fullmatch(value) is None:
        raise PicoHybridPolicyContractError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_exact_sha256(metadata: Mapping[str, str], name: str, expected: str) -> str:
    value = _require_lowercase_sha256(metadata, name)
    if value != expected:
        raise PicoHybridPolicyContractError(
            f"{name} does not match the fixed Microban contract"
        )
    return value


def _require_v10_migration_provenance(
    metadata: Mapping[str, str],
) -> tuple[str, int, str, str, str, str, float, float]:
    """Require the one authenticated full-state v9-to-v10 migration ledger."""

    checkpoint_sha256 = _require_exact_sha256(
        metadata,
        "migration_source_checkpoint_sha256",
        EXPECTED_MIGRATION_SOURCE_CHECKPOINT_SHA256,
    )
    checkpoint_iteration = _canonical_nonnegative_int(
        metadata.get("migration_source_checkpoint_iteration"),
        "migration_source_checkpoint_iteration",
    )
    if checkpoint_iteration != EXPECTED_MIGRATION_SOURCE_CHECKPOINT_ITERATION:
        raise PicoHybridPolicyContractError(
            "migration source checkpoint iteration does not match contract v10"
        )
    training_sha256 = _require_exact_sha256(
        metadata,
        "migration_source_training_provenance_sha256",
        EXPECTED_MIGRATION_SOURCE_TRAINING_PROVENANCE_SHA256,
    )
    source_tree_sha256 = _require_exact_sha256(
        metadata,
        "migration_source_tree_sha256",
        EXPECTED_MIGRATION_SOURCE_TREE_SHA256,
    )
    gate_sha256 = _require_exact_sha256(
        metadata,
        "migration_source_gate_sha256",
        EXPECTED_MIGRATION_SOURCE_GATE_SHA256,
    )
    state_transfer = metadata.get("migration_state_transfer", "")
    if state_transfer != EXPECTED_MIGRATION_STATE_TRANSFER:
        raise PicoHybridPolicyContractError(
            "migration state transfer does not match contract v10"
        )
    source_learning_rate = _require_exact_finite_scalar(
        metadata,
        "migration_source_optimizer_learning_rate",
        EXPECTED_MIGRATION_SOURCE_OPTIMIZER_LEARNING_RATE,
    )
    fixed_learning_rate = _require_exact_finite_scalar(
        metadata,
        "training_fixed_learning_rate",
        EXPECTED_TRAINING_FIXED_LEARNING_RATE,
    )
    return (
        checkpoint_sha256,
        checkpoint_iteration,
        training_sha256,
        source_tree_sha256,
        gate_sha256,
        state_transfer,
        source_learning_rate,
        fixed_learning_rate,
    )


def _require_gated_export_provenance(
    metadata: Mapping[str, str],
) -> tuple[str, int, int, str]:
    """Reject artifacts that did not pass the checked-in v1 parity gate.

    This validates the gate assertion and internally consistent checkpoint
    identity recorded by the exporter.  It provides fail-closed traceability;
    it is not a signature and does not authenticate an untrusted ONNX file.
    """

    if metadata.get("onnx_parity_verified") != "true":
        raise PicoHybridPolicyContractError(
            "onnx_parity_verified metadata must be true"
        )
    if metadata.get("onnx_parity_gate_version") != EXPECTED_ONNX_PARITY_GATE_VERSION:
        raise PicoHybridPolicyContractError(
            "unsupported or missing ONNX parity gate version"
        )
    if metadata.get("onnx_parity_runtime") != EXPECTED_ONNX_PARITY_RUNTIME:
        raise PicoHybridPolicyContractError("unsupported ONNX parity runtime")

    parity_seed = _canonical_nonnegative_int(
        metadata.get("onnx_parity_seed"), "onnx_parity_seed"
    )
    parity_samples = _canonical_nonnegative_int(
        metadata.get("onnx_parity_sample_count"), "onnx_parity_sample_count"
    )
    if parity_seed != EXPECTED_ONNX_PARITY_SEED:
        raise PicoHybridPolicyContractError("unsupported ONNX parity seed")
    if parity_samples != EXPECTED_ONNX_PARITY_SAMPLE_COUNT:
        raise PicoHybridPolicyContractError("unsupported ONNX parity sample count")

    for name, expected in (
        ("onnx_parity_atol", EXPECTED_ONNX_PARITY_ATOL),
        ("onnx_parity_rtol", EXPECTED_ONNX_PARITY_RTOL),
    ):
        try:
            actual = float(metadata.get(name, "nan"))
        except (TypeError, ValueError) as exc:
            raise PicoHybridPolicyContractError(f"{name} must be numeric") from exc
        if not math.isfinite(actual) or actual != expected:
            raise PicoHybridPolicyContractError(f"{name} does not match parity gate v1")

    filename = metadata.get("checkpoint_filename", "")
    filename_match = _CHECKPOINT_FILENAME_RE.fullmatch(filename)
    if filename_match is None:
        raise PicoHybridPolicyContractError(
            "checkpoint_filename must be canonical model_N.pt"
        )
    iteration = _canonical_nonnegative_int(
        metadata.get("checkpoint_iteration"), "checkpoint_iteration"
    )
    if int(filename_match.group(1)) != iteration:
        raise PicoHybridPolicyContractError(
            "checkpoint_iteration does not match checkpoint_filename"
        )
    if metadata.get("checkpoint_iteration_semantics") != (
        "zero_based_completed_update_index_from_model_filename"
    ):
        raise PicoHybridPolicyContractError(
            "unsupported checkpoint_iteration semantics"
        )
    completed_updates = _canonical_nonnegative_int(
        metadata.get("checkpoint_completed_updates"),
        "checkpoint_completed_updates",
    )
    if completed_updates != iteration + 1:
        raise PicoHybridPolicyContractError(
            "checkpoint_completed_updates must equal checkpoint_iteration + 1"
        )

    checkpoint_sha256 = _require_lowercase_sha256(metadata, "checkpoint_sha256")
    return filename, iteration, completed_updates, checkpoint_sha256


def _require_final_deployment_provenance(
    metadata: Mapping[str, str],
    *,
    checkpoint_iteration: int,
    checkpoint_completed_updates: int,
    checkpoint_sha256: str,
) -> tuple[str, str, str, str, str, int]:
    """Require a canonical final-stage checkpoint and its schema-3 pass receipt.

    Diagnostic and automatic exports deliberately carry
    ``deployment_accepted=false``.  The robot must never infer deployability
    merely from a v10 recipe label or a final-looking checkpoint filename.
    """

    schema_version = _canonical_nonnegative_int(
        metadata.get("training_provenance_schema_version"),
        "training_provenance_schema_version",
    )
    if schema_version != EXPECTED_TRAINING_PROVENANCE_SCHEMA_VERSION:
        raise PicoHybridPolicyContractError(
            "unsupported training provenance schema version"
        )
    training_sha256 = _require_lowercase_sha256(metadata, "training_provenance_sha256")
    source_tree_sha256 = _require_lowercase_sha256(
        metadata, "training_source_tree_sha256"
    )
    if metadata.get("training_recipe_revision") != EXPECTED_RECIPE_REVISION:
        raise PicoHybridPolicyContractError(
            "training provenance recipe revision does not match deployment"
        )
    if metadata.get("training_actor_initialization") != EXPECTED_ACTOR_INITIALIZATION:
        raise PicoHybridPolicyContractError(
            "training provenance actor initialization does not match deployment"
        )
    if metadata.get("training_provenance_mode") != EXPECTED_TRAINING_PROVENANCE_MODE:
        raise PicoHybridPolicyContractError(
            "training provenance is not from the canonical v10 stage driver"
        )
    if metadata.get("canonical_training_stage") != "true":
        raise PicoHybridPolicyContractError(
            "canonical_training_stage metadata must be true"
        )

    stage_start = _canonical_nonnegative_int(
        metadata.get("training_stage_start_boundary"),
        "training_stage_start_boundary",
    )
    stage_target = _canonical_nonnegative_int(
        metadata.get("training_stage_target_boundary"),
        "training_stage_target_boundary",
    )
    if (
        stage_start != EXPECTED_FINAL_TRAINING_STAGE_START_BOUNDARY
        or stage_target != EXPECTED_FINAL_TRAINING_STAGE_TARGET_BOUNDARY
    ):
        raise PicoHybridPolicyContractError(
            "deployment requires canonical training stage 10000->15000"
        )
    if (
        checkpoint_iteration != stage_target - 1
        or checkpoint_completed_updates != stage_target
    ):
        raise PicoHybridPolicyContractError(
            "deployment checkpoint identity does not match the final stage boundary"
        )
    _require_lowercase_sha256(metadata, "training_parent_checkpoint_sha256")
    _require_lowercase_sha256(metadata, "training_parent_gate_sha256")
    resume_source_checkpoint_sha256 = _require_lowercase_sha256(
        metadata, "training_resume_source_checkpoint_sha256"
    )
    resume_source_checkpoint_iteration = _canonical_nonnegative_int(
        metadata.get("training_resume_source_checkpoint_iteration"),
        "training_resume_source_checkpoint_iteration",
    )
    if not (stage_start - 1 <= resume_source_checkpoint_iteration < stage_target - 1):
        raise PicoHybridPolicyContractError(
            "training resume source iteration must be inside the final stage "
            "from model_9999.pt through model_14998.pt"
        )

    if metadata.get("deployment_accepted") != "true":
        raise PicoHybridPolicyContractError("deployment_accepted metadata must be true")
    receipt_schema = _canonical_nonnegative_int(
        metadata.get("acceptance_receipt_schema_version"),
        "acceptance_receipt_schema_version",
    )
    if receipt_schema != EXPECTED_ACCEPTANCE_RECEIPT_SCHEMA_VERSION:
        raise PicoHybridPolicyContractError(
            "unsupported acceptance receipt schema version"
        )
    receipt_sha256 = _require_lowercase_sha256(metadata, "acceptance_receipt_sha256")
    if metadata.get("acceptance_status") != "pass":
        raise PicoHybridPolicyContractError("acceptance_status metadata must be pass")
    acceptance_boundary = _canonical_nonnegative_int(
        metadata.get("acceptance_boundary"), "acceptance_boundary"
    )
    if acceptance_boundary != stage_target:
        raise PicoHybridPolicyContractError(
            "acceptance boundary does not match the final training stage"
        )
    if (
        metadata.get("acceptance_evaluator_revision")
        != EXPECTED_ACCEPTANCE_EVALUATOR_REVISION
    ):
        raise PicoHybridPolicyContractError("unsupported acceptance evaluator revision")
    if metadata.get("acceptance_revision") != EXPECTED_ACCEPTANCE_REVISION:
        raise PicoHybridPolicyContractError("unsupported acceptance revision")
    evaluator_source_sha256 = _require_lowercase_sha256(
        metadata, "acceptance_evaluator_source_sha256"
    )
    if metadata.get("acceptance_checkpoint_sha256") != checkpoint_sha256:
        raise PicoHybridPolicyContractError(
            "acceptance receipt checkpoint SHA-256 does not match the ONNX checkpoint"
        )
    if metadata.get("acceptance_training_provenance_sha256") != training_sha256:
        raise PicoHybridPolicyContractError(
            "acceptance receipt training provenance SHA-256 does not match"
        )
    if metadata.get("acceptance_recipe_revision") != EXPECTED_RECIPE_REVISION:
        raise PicoHybridPolicyContractError(
            "acceptance receipt recipe revision does not match deployment"
        )
    nominal_reports = _canonical_nonnegative_int(
        metadata.get("acceptance_nominal_report_count"),
        "acceptance_nominal_report_count",
    )
    moving_hmd_reports = _canonical_nonnegative_int(
        metadata.get("acceptance_moving_hmd_report_count"),
        "acceptance_moving_hmd_report_count",
    )
    if (
        nominal_reports != EXPECTED_ACCEPTANCE_NOMINAL_REPORT_COUNT
        or moving_hmd_reports != EXPECTED_ACCEPTANCE_MOVING_HMD_REPORT_COUNT
    ):
        raise PicoHybridPolicyContractError(
            "final acceptance must contain three nominal and three moving-HMD reports"
        )
    return (
        training_sha256,
        source_tree_sha256,
        receipt_sha256,
        evaluator_source_sha256,
        resume_source_checkpoint_sha256,
        resume_source_checkpoint_iteration,
    )


def _require_safe_velocity_bootstrap_provenance(
    metadata: Mapping[str, str],
) -> tuple[str, int, str]:
    """Require the safe source inherited through the pinned v10 migration."""

    checkpoint_sha256 = _require_exact_sha256(
        metadata,
        "safe_velocity_source_checkpoint_sha256",
        EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_SHA256,
    )
    checkpoint_iteration = _canonical_nonnegative_int(
        metadata.get("safe_velocity_source_checkpoint_iteration"),
        "safe_velocity_source_checkpoint_iteration",
    )
    if checkpoint_iteration != EXPECTED_SAFE_VELOCITY_SOURCE_CHECKPOINT_ITERATION:
        raise PicoHybridPolicyContractError(
            "safe-velocity source checkpoint iteration does not match migration"
        )
    if (
        metadata.get("safe_velocity_source_recipe_revision")
        != EXPECTED_SAFE_VELOCITY_RECIPE_REVISION
    ):
        raise PicoHybridPolicyContractError(
            "safe-velocity source recipe does not match deployment"
        )
    receipt_sha256 = _require_exact_sha256(
        metadata,
        "safe_velocity_acceptance_receipt_sha256",
        EXPECTED_SAFE_VELOCITY_ACCEPTANCE_RECEIPT_SHA256,
    )
    receipt_schema = _canonical_nonnegative_int(
        metadata.get("safe_velocity_acceptance_receipt_schema_version"),
        "safe_velocity_acceptance_receipt_schema_version",
    )
    if receipt_schema != EXPECTED_SAFE_VELOCITY_RECEIPT_SCHEMA_VERSION:
        raise PicoHybridPolicyContractError(
            "unsupported safe-velocity acceptance receipt schema"
        )
    if (
        metadata.get("safe_velocity_acceptance_gate")
        != EXPECTED_SAFE_VELOCITY_ACCEPTANCE_GATE
    ):
        raise PicoHybridPolicyContractError("unsupported safe-velocity acceptance gate")
    if (
        metadata.get("safe_velocity_bootstrap_mapping_version")
        != EXPECTED_SAFE_VELOCITY_BOOTSTRAP_MAPPING_VERSION
    ):
        raise PicoHybridPolicyContractError(
            "unsupported safe-velocity bootstrap mapping"
        )
    return checkpoint_sha256, checkpoint_iteration, receipt_sha256


def onnxruntime_compatibility_smoke_inputs() -> np.ndarray:
    """Build the fixed v1 corpus used to exercise the deployed ORT runtime.

    The corpus mirrors the export gate's finite 83-value observations, but the
    robot does not have PyTorch reference outputs.  It therefore supports only
    a runtime compatibility smoke (load/run/shape/finiteness), not a second
    numerical parity claim.
    """

    lower = np.asarray(
        [-4.0] * 3
        + [-1.0] * 3
        + [-math.pi] * 21
        + [-12.0] * 21
        + [-1.0] * 18
        + [-1.0, -1.0, -2.0]
        + [-0.03, -0.03, 0.0] * 2
        + [-0.08] * 6
        + [0.0, 0.0],
        dtype=np.float32,
    )
    upper = np.asarray(
        [4.0] * 3
        + [1.0] * 3
        + [math.pi] * 21
        + [12.0] * 21
        + [1.0] * 18
        + [1.0, 1.0, 2.0]
        + [0.03, 0.03, 0.05] * 2
        + [0.08] * 6
        + [1.0, 1.0],
        dtype=np.float32,
    )
    if lower.shape != (EXPECTED_OBSERVATION_WIDTH,) or upper.shape != (
        EXPECTED_OBSERVATION_WIDTH,
    ):
        raise RuntimeError("ORT smoke bounds do not match the 83-value schema")

    neutral = np.zeros(EXPECTED_OBSERVATION_WIDTH, dtype=np.float32)
    neutral[5] = -1.0
    rows = [neutral, lower, upper, (lower + upper) * np.float32(0.5)]
    rng = np.random.default_rng(EXPECTED_ONNX_PARITY_SEED)
    random_rows = rng.uniform(
        lower,
        upper,
        size=(
            EXPECTED_ONNX_PARITY_SAMPLE_COUNT - len(rows),
            EXPECTED_OBSERVATION_WIDTH,
        ),
    ).astype(np.float32)
    random_rows[:, -2:] = rng.integers(0, 2, size=(random_rows.shape[0], 2)).astype(
        np.float32
    )
    rows.extend(random_rows)
    observations = np.stack(rows, axis=0).reshape(
        EXPECTED_ONNX_PARITY_SAMPLE_COUNT,
        1,
        EXPECTED_OBSERVATION_WIDTH,
    )
    if not np.isfinite(observations).all():
        raise RuntimeError("ORT compatibility smoke corpus is non-finite")
    return observations


def validate_onnxruntime_compatibility(
    session: Any,
    input_name: str,
    actor_lower: Sequence[float] = EXPECTED_ACTOR_RAW_ACTION_LOWER,
    actor_upper: Sequence[float] = EXPECTED_ACTOR_RAW_ACTION_UPPER,
    deterministic_lower: Sequence[float] = EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER,
    deterministic_upper: Sequence[float] = EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER,
) -> int:
    """Run the fixed corpus through ONNX Runtime and validate its outputs.

    Returns the number of observations executed.  This deliberately does not
    compare against PyTorch: the export gate owns that numerical parity check.
    Contract v10's exported deterministic graph must also keep every result
    strictly inside its open actor interval; reaching either endpoint fails the
    load.
    """

    lower = np.asarray(actor_lower, dtype=np.float64)
    upper = np.asarray(actor_upper, dtype=np.float64)
    output_lower = np.asarray(deterministic_lower, dtype=np.float64)
    output_upper = np.asarray(deterministic_upper, dtype=np.float64)
    if (
        lower.shape != (EXPECTED_ACTION_WIDTH,)
        or upper.shape != (EXPECTED_ACTION_WIDTH,)
        or output_lower.shape != (EXPECTED_ACTION_WIDTH,)
        or output_upper.shape != (EXPECTED_ACTION_WIDTH,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.isfinite(output_lower).all()
        or not np.isfinite(output_upper).all()
        or not np.all(lower < 0.0)
        or not np.all(upper > 0.0)
        or not np.all(lower < output_lower)
        or not np.all(output_lower < 0.0)
        or not np.all(output_upper > 0.0)
        or not np.all(output_upper < upper)
    ):
        raise PicoHybridPolicyRuntimeError(
            "invalid ORT actor-bound or deterministic-envelope contract"
        )

    observations = onnxruntime_compatibility_smoke_inputs()
    for sample_index, observation in enumerate(observations):
        try:
            outputs = session.run(None, {input_name: observation})
        except Exception as exc:
            raise PicoHybridPolicyRuntimeError(
                f"ONNX Runtime compatibility smoke failed at sample {sample_index}"
            ) from exc
        if len(outputs) != 1:
            raise PicoHybridPolicyRuntimeError(
                "ONNX Runtime compatibility smoke expected exactly one output "
                f"at sample {sample_index}"
            )
        output = np.asarray(outputs[0])
        try:
            finite = bool(np.isfinite(output).all())
        except TypeError as exc:
            raise PicoHybridPolicyRuntimeError(
                "ONNX Runtime compatibility smoke returned a non-numeric output "
                f"at sample {sample_index}"
            ) from exc
        if output.shape != (1, EXPECTED_ACTION_WIDTH) or not finite:
            raise PicoHybridPolicyRuntimeError(
                "ONNX Runtime compatibility smoke returned an unsafe output "
                f"at sample {sample_index}: shape={output.shape}, finite={finite}"
            )
        if not bool(np.all(output[0] > lower) and np.all(output[0] < upper)):
            raise PicoHybridPolicyRuntimeError(
                "ONNX Runtime compatibility smoke output reached or exceeded "
                f"an open actor bound at sample {sample_index}"
            )
        if not bool(
            np.all(output[0] >= output_lower) and np.all(output[0] <= output_upper)
        ):
            raise PicoHybridPolicyRuntimeError(
                "ONNX Runtime compatibility smoke output escaped the current "
                f"deterministic transform envelope at sample {sample_index}"
            )
    return len(observations)


def validate_v12_onnxruntime_compatibility(
    session: Any,
    input_name: str,
    raw_action_absolute_maximum: Sequence[float],
) -> int:
    """Exercise a raw/unbounded v12 graph under its evidence-derived guard.

    The legacy actor was trained with an unbounded Gaussian mean and its proven
    closed-loop contract sends that raw value directly to the joint-position
    action term.  This gate does not reuse v10's bounded transform or modify an
    output.  It checks numeric type, fixed shape, float32 finiteness and the
    final tracking evidence's gross finite-amplitude envelope.
    """

    guard = np.asarray(raw_action_absolute_maximum, dtype=np.float64)
    if (
        guard.shape != (EXPECTED_ACTION_WIDTH,)
        or not np.isfinite(guard).all()
        or bool(np.any(guard < 0.0))
    ):
        raise PicoHybridPolicyContractError(
            "contract-v12 runtime raw-action guard is malformed"
        )
    observations = onnxruntime_compatibility_smoke_inputs()
    for sample_index, observation in enumerate(observations):
        try:
            outputs = session.run(None, {input_name: observation})
        except Exception as exc:
            raise PicoHybridPolicyRuntimeError(
                f"contract-v12 ONNX Runtime smoke failed at sample {sample_index}"
            ) from exc
        if len(outputs) != 1:
            raise PicoHybridPolicyRuntimeError(
                "contract-v12 ONNX Runtime smoke expected exactly one output "
                f"at sample {sample_index}"
            )
        output = np.asarray(outputs[0])
        try:
            finite = bool(np.isfinite(output).all())
            with np.errstate(over="ignore", invalid="ignore"):
                float32_output = output.astype(np.float32, casting="unsafe", copy=False)
            finite_float32 = bool(np.isfinite(float32_output).all())
        except (TypeError, ValueError, OverflowError) as exc:
            raise PicoHybridPolicyRuntimeError(
                "contract-v12 ONNX Runtime smoke returned a non-numeric output "
                f"at sample {sample_index}"
            ) from exc
        if (
            output.shape != (1, EXPECTED_ACTION_WIDTH)
            or not finite
            or not finite_float32
        ):
            raise PicoHybridPolicyRuntimeError(
                "contract-v12 ONNX Runtime smoke returned an unsafe output "
                f"at sample {sample_index}: shape={output.shape}, "
                f"finite={finite and finite_float32}"
            )
        if bool(np.any(np.abs(float32_output[0]) > guard)):
            raise PicoHybridPolicyRuntimeError(
                "contract-v12 ONNX Runtime smoke output escaped the authenticated "
                f"finite-amplitude guard at sample {sample_index}"
            )
    return len(observations)


def _serialized_metadata_values(values: Sequence[float]) -> tuple[float, ...]:
    """Mirror MjLab's three-decimal ONNX metadata serialization."""

    return tuple(float(f"{value:.3f}") for value in values)


def _require_fixed_metadata_vector(
    name: str,
    actual: Sequence[float],
    expected_runtime: Sequence[float],
) -> None:
    """Require metadata to describe the compiled-in robot-side contract."""

    expected_metadata = _serialized_metadata_values(expected_runtime)
    if tuple(actual) == expected_metadata:
        return
    mismatch = next(
        (
            index
            for index, (received, expected) in enumerate(
                zip(actual, expected_metadata, strict=True)
            )
            if received != expected
        ),
        None,
    )
    raise PicoHybridPolicyContractError(
        f"{name} does not match the fixed Microban contract"
        + (
            ""
            if mismatch is None
            else (
                f" at action index {mismatch}: got {actual[mismatch]}, "
                f"expected {expected_metadata[mismatch]}"
            )
        )
    )


def _fixed_width(node: Any, name: str) -> int:
    shape = getattr(node, "shape", None)
    if not isinstance(shape, Sequence) or len(shape) != 2:
        raise PicoHybridPolicyContractError(
            f"{name} must be a fixed rank-2 tensor; got {shape!r}"
        )
    batch, width = shape
    if batch not in (1, "1") or not isinstance(width, int):
        raise PicoHybridPolicyContractError(
            f"{name} must have fixed shape [1, N]; got {shape!r}"
        )
    return width


def _quat_rotate_vector(
    quat_wxyz: Sequence[float], vector_xyz: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate ``vector_xyz`` by the normalized WXYZ quaternion."""

    if len(quat_wxyz) != 4 or len(vector_xyz) != 3:
        raise PicoHybridPolicyRuntimeError("invalid gyro mount transform dimensions")
    q = np.asarray(quat_wxyz, dtype=np.float64)
    vector = np.asarray(vector_xyz, dtype=np.float64)
    if not np.isfinite(q).all() or not np.isfinite(vector).all():
        raise PicoHybridPolicyRuntimeError("gyro or mount transform is non-finite")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise PicoHybridPolicyRuntimeError("gyro mount quaternion has zero length")
    w, x, y, z = q / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    rotated = rotation @ vector
    return tuple(float(value) for value in rotated)


def sensor_gyro_to_body(
    gyro_sensor_xyz: Sequence[float],
    mount_quat_wxyz: Sequence[float] = IMU_MOUNT_QUAT,
) -> tuple[float, float, float]:
    """Transform BMI088 sensor-frame angular velocity into the robot body frame.

    The established runtime convention applies ``IMU_MOUNT_QUAT`` directly as
    the sensor-to-body vector rotation.  The basis mapping is locked by a unit
    test, while its physical axes/signs still require the documented supported-
    robot acceptance test before enabling this policy on hardware.
    """

    return _quat_rotate_vector(mount_quat_wxyz, gyro_sensor_xyz)


@dataclass(frozen=True)
class _PolicyContract:
    input_name: str
    observation_joint_names: tuple[str, ...]
    observation_default_joint_pos: tuple[float, ...]
    action_joint_names: tuple[str, ...]
    action_default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    soft_lower: tuple[float, ...]
    soft_upper: tuple[float, ...]
    raw_action_soft_lower: tuple[float, ...]
    raw_action_soft_upper: tuple[float, ...]
    actor_raw_action_lower: tuple[float, ...]
    actor_raw_action_upper: tuple[float, ...]
    foot_lower: tuple[float, ...]
    foot_upper: tuple[float, ...]
    simultaneous_both_feet_lower: tuple[float, ...]
    simultaneous_both_feet_upper: tuple[float, ...]
    hand_lower: tuple[float, ...]
    hand_upper: tuple[float, ...]
    checkpoint_filename: str
    checkpoint_iteration: int
    checkpoint_completed_updates: int
    checkpoint_sha256: str
    training_provenance_sha256: str
    training_source_tree_sha256: str
    training_resume_source_checkpoint_sha256: str
    training_resume_source_checkpoint_iteration: int
    migration_source_checkpoint_sha256: str
    migration_source_checkpoint_iteration: int
    migration_source_training_provenance_sha256: str
    migration_source_tree_sha256: str
    migration_source_gate_sha256: str
    migration_state_transfer: str
    migration_source_optimizer_learning_rate: float
    training_fixed_learning_rate: float
    acceptance_receipt_sha256: str
    acceptance_evaluator_source_sha256: str
    safe_velocity_source_checkpoint_sha256: str
    safe_velocity_source_checkpoint_iteration: int
    safe_velocity_acceptance_receipt_sha256: str
    training_contract_version: str = EXPECTED_TRAINING_CONTRACT_VERSION
    runtime_action_semantics: str = "bounded_v10_effective_action"
    v12_legacy_source_checkpoint_sha256: str | None = None
    v12_legacy_probe_sha256: str | None = None
    v12_stage_gate_sha256: str | None = None
    v12_locomotion_report_sha256: str | None = None
    v12_onnx_report_sha256: str | None = None
    v12_tracking_report_sha256: str | None = None
    v12_raw_action_minimum: tuple[float, ...] = ()
    v12_raw_action_maximum: tuple[float, ...] = ()
    v12_raw_action_absolute_maximum: tuple[float, ...] = ()
    v12_source_raw_action_minimum: tuple[float, ...] = ()
    v12_source_raw_action_maximum: tuple[float, ...] = ()
    v12_source_raw_action_absolute_maximum: tuple[float, ...] = ()
    v12_learned_source_delta_minimum: tuple[float, ...] = ()
    v12_learned_source_delta_maximum: tuple[float, ...] = ()
    v12_learned_source_delta_absolute_maximum: tuple[float, ...] = ()
    v12_runtime_raw_action_guard_absolute_maximum: tuple[float, ...] = ()


def _parse_v10_contract(session: Any) -> _PolicyContract:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise PicoHybridPolicyContractError(
            "policy must have exactly one input and output"
        )
    if _fixed_width(inputs[0], "policy input") != EXPECTED_OBSERVATION_WIDTH:
        raise PicoHybridPolicyContractError("policy input width must be exactly 83")
    if _fixed_width(outputs[0], "policy output") != EXPECTED_ACTION_WIDTH:
        raise PicoHybridPolicyContractError("policy output width must be exactly 18")
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("policy_type") != EXPECTED_POLICY_TYPE:
        raise PicoHybridPolicyContractError("ONNX is not a Microban PICO hybrid policy")
    (
        checkpoint_filename,
        checkpoint_iteration,
        checkpoint_completed_updates,
        checkpoint_sha256,
    ) = _require_gated_export_provenance(metadata)
    if (
        metadata.get("microban_teleop_training_contract_version")
        != EXPECTED_TRAINING_CONTRACT_VERSION
    ):
        raise PicoHybridPolicyContractError(
            "unsupported or missing Microban teleop training contract version"
        )
    if (
        metadata.get("microban_teleop_actor_initialization")
        != EXPECTED_ACTOR_INITIALIZATION
    ):
        raise PicoHybridPolicyContractError(
            "unsupported or missing Microban teleop actor initialization"
        )
    if metadata.get("microban_teleop_recipe_revision") != EXPECTED_RECIPE_REVISION:
        raise PicoHybridPolicyContractError(
            "unsupported or missing Microban teleop recipe revision"
        )
    (
        safe_velocity_source_checkpoint_sha256,
        safe_velocity_source_checkpoint_iteration,
        safe_velocity_acceptance_receipt_sha256,
    ) = _require_safe_velocity_bootstrap_provenance(metadata)
    (
        training_provenance_sha256,
        training_source_tree_sha256,
        acceptance_receipt_sha256,
        acceptance_evaluator_source_sha256,
        training_resume_source_checkpoint_sha256,
        training_resume_source_checkpoint_iteration,
    ) = _require_final_deployment_provenance(
        metadata,
        checkpoint_iteration=checkpoint_iteration,
        checkpoint_completed_updates=checkpoint_completed_updates,
        checkpoint_sha256=checkpoint_sha256,
    )
    (
        migration_source_checkpoint_sha256,
        migration_source_checkpoint_iteration,
        migration_source_training_provenance_sha256,
        migration_source_tree_sha256,
        migration_source_gate_sha256,
        migration_state_transfer,
        migration_source_optimizer_learning_rate,
        training_fixed_learning_rate,
    ) = _require_v10_migration_provenance(metadata)
    if metadata.get("observation_schema_version") != EXPECTED_SCHEMA_VERSION:
        raise PicoHybridPolicyContractError("unsupported observation schema version")
    if metadata.get("base_ang_vel_frame") != "robot_body_xyz":
        raise PicoHybridPolicyContractError("base_ang_vel_frame must be robot_body_xyz")
    if metadata.get("base_ang_vel_units") != "rad_s":
        raise PicoHybridPolicyContractError("base_ang_vel_units must be rad_s")
    if metadata.get("observation_width") != str(EXPECTED_OBSERVATION_WIDTH):
        raise PicoHybridPolicyContractError("observation_width metadata must be 83")
    if _split_csv(
        metadata.get("locomotion_command_order"), "locomotion_command_order"
    ) != (
        "linear_velocity_x",
        "linear_velocity_y",
        "angular_velocity_z",
    ):
        raise PicoHybridPolicyContractError("unsupported locomotion command order")
    if _split_csv(
        metadata.get("locomotion_command_units"), "locomotion_command_units"
    ) != (
        "m_s",
        "m_s",
        "rad_s",
    ):
        raise PicoHybridPolicyContractError("unsupported locomotion command units")
    if metadata.get("locomotion_command_frame") != "robot_body_forward_left_yaw_up":
        raise PicoHybridPolicyContractError("unsupported locomotion command frame")
    if metadata.get("previous_action_semantics") != EXPECTED_PREVIOUS_ACTION_SEMANTICS:
        raise PicoHybridPolicyContractError("unsupported previous-action semantics")
    if (
        metadata.get("action_target_semantics")
        != "default_joint_pos_plus_raw_action_times_scale"
    ):
        raise PicoHybridPolicyContractError("unsupported action target semantics")
    if metadata.get("action_clip_semantics") != "absolute_joint_position_radians":
        raise PicoHybridPolicyContractError("unsupported action clipping semantics")
    if (
        metadata.get("action_distribution_semantics")
        != EXPECTED_ACTION_DISTRIBUTION_SEMANTICS
    ):
        raise PicoHybridPolicyContractError(
            "unsupported or missing action distribution semantics"
        )
    _require_exact_finite_scalar(
        metadata,
        "actor_target_guard_margin_ratio",
        EXPECTED_ACTOR_TARGET_GUARD_MARGIN_RATIO,
    )
    _require_exact_finite_scalar(
        metadata,
        "actor_default_interior_epsilon_rad",
        EXPECTED_ACTOR_DEFAULT_INTERIOR_EPSILON_RAD,
    )
    for name, expected in (
        (
            "actor_latent_operational_scale_multiplier",
            EXPECTED_ACTOR_LATENT_OPERATIONAL_SCALE_MULTIPLIER,
        ),
        (
            "actor_latent_operational_abs_max",
            EXPECTED_ACTOR_LATENT_OPERATIONAL_ABS_MAX,
        ),
        ("actor_latent_mean_fraction", EXPECTED_ACTOR_LATENT_MEAN_FRACTION),
        (
            "actor_latent_std_min_abs_max",
            EXPECTED_ACTOR_LATENT_STD_MIN_ABS_MAX,
        ),
        (
            "actor_latent_std_min_envelope_divisor",
            EXPECTED_ACTOR_LATENT_STD_MIN_ENVELOPE_DIVISOR,
        ),
        ("actor_latent_std_abs_max", EXPECTED_ACTOR_LATENT_STD_ABS_MAX),
        (
            "actor_latent_std_envelope_divisor",
            EXPECTED_ACTOR_LATENT_STD_ENVELOPE_DIVISOR,
        ),
    ):
        _require_exact_finite_scalar(metadata, name, expected)

    try:
        control_hz = float(metadata.get("control_hz", "nan"))
    except ValueError as exc:
        raise PicoHybridPolicyContractError("control_hz must be numeric") from exc
    if not math.isclose(control_hz, 50.0, abs_tol=1e-6):
        raise PicoHybridPolicyContractError(
            f"policy control_hz must be 50, got {control_hz}"
        )

    observation_names = _split_csv(
        metadata.get("observation_names"), "observation_names"
    )
    if observation_names != EXPECTED_OBSERVATION_TERMS:
        raise PicoHybridPolicyContractError(
            f"unsafe observation order: {observation_names!r}"
        )
    observation_joints = _split_csv(
        metadata.get("observation_joint_names"), "observation_joint_names"
    )
    if len(observation_joints) != len(MOTOR_TO_ID) or set(observation_joints) != set(
        MOTOR_TO_ID
    ):
        raise PicoHybridPolicyContractError(
            "observation_joint_names must contain each of Microban's 21 joints exactly once"
        )
    action_joints = _split_csv(metadata.get("action_joint_names"), "action_joint_names")
    if action_joints != tuple(OBSERVATION_DOF_ORDER):
        raise PicoHybridPolicyContractError(f"unsafe action order: {action_joints!r}")

    observation_defaults = _float_csv(
        metadata.get("observation_default_joint_pos"),
        "observation_default_joint_pos",
        len(observation_joints),
    )
    action_defaults = _float_csv(
        metadata.get("default_joint_pos"), "default_joint_pos", len(action_joints)
    )
    scales = _float_csv(
        metadata.get("action_scale"), "action_scale", len(action_joints)
    )
    soft_lower = _float_csv(
        metadata.get("soft_joint_pos_lower"), "soft_joint_pos_lower", len(action_joints)
    )
    soft_upper = _float_csv(
        metadata.get("soft_joint_pos_upper"), "soft_joint_pos_upper", len(action_joints)
    )
    if any(lower >= upper for lower, upper in zip(soft_lower, soft_upper)):
        raise PicoHybridPolicyContractError("invalid policy soft joint limits")
    if any(scale <= 0.0 for scale in scales):
        raise PicoHybridPolicyContractError("action_scale values must be positive")

    expected_observation_defaults = tuple(
        float(PICO_TELEOP_HOME_POSE[name]) for name in observation_joints
    )
    _require_fixed_metadata_vector(
        "observation_default_joint_pos",
        observation_defaults,
        expected_observation_defaults,
    )
    _require_fixed_metadata_vector(
        "default_joint_pos", action_defaults, EXPECTED_ACTION_DEFAULT_JOINT_POS
    )
    _require_fixed_metadata_vector("action_scale", scales, EXPECTED_ACTION_SCALE)
    _require_fixed_metadata_vector(
        "soft_joint_pos_lower", soft_lower, EXPECTED_SOFT_JOINT_POS_LOWER
    )
    _require_fixed_metadata_vector(
        "soft_joint_pos_upper", soft_upper, EXPECTED_SOFT_JOINT_POS_UPPER
    )

    exact_action_bounds = {
        name: _float_json_vector(metadata.get(name), name, EXPECTED_ACTION_WIDTH)
        for name in (
            "raw_action_soft_lower_json",
            "raw_action_soft_upper_json",
            "actor_raw_action_lower_json",
            "actor_raw_action_upper_json",
        )
    }
    raw_action_soft_lower = exact_action_bounds["raw_action_soft_lower_json"]
    raw_action_soft_upper = exact_action_bounds["raw_action_soft_upper_json"]
    actor_raw_action_lower = exact_action_bounds["actor_raw_action_lower_json"]
    actor_raw_action_upper = exact_action_bounds["actor_raw_action_upper_json"]
    for index, (soft_lo, soft_hi, actor_lo, actor_hi) in enumerate(
        zip(
            raw_action_soft_lower,
            raw_action_soft_upper,
            actor_raw_action_lower,
            actor_raw_action_upper,
            strict=True,
        )
    ):
        if not soft_lo < actor_lo < 0.0 < actor_hi < soft_hi:
            raise PicoHybridPolicyContractError(
                "actor bounds must be strictly inside raw soft bounds with zero "
                f"strictly interior (action index {index})"
            )
    for name, actual, expected in (
        (
            "raw_action_soft_lower_json",
            raw_action_soft_lower,
            EXPECTED_RAW_ACTION_SOFT_LOWER,
        ),
        (
            "raw_action_soft_upper_json",
            raw_action_soft_upper,
            EXPECTED_RAW_ACTION_SOFT_UPPER,
        ),
        (
            "actor_raw_action_lower_json",
            actor_raw_action_lower,
            EXPECTED_ACTOR_RAW_ACTION_LOWER,
        ),
        (
            "actor_raw_action_upper_json",
            actor_raw_action_upper,
            EXPECTED_ACTOR_RAW_ACTION_UPPER,
        ),
    ):
        _require_tight_json_vector(name, actual, expected)

    foot_lower = _float_csv(metadata.get("foot_target_lower"), "foot_target_lower", 6)
    foot_upper = _float_csv(metadata.get("foot_target_upper"), "foot_target_upper", 6)
    simultaneous_both_feet_lower = _float_csv(
        metadata.get("simultaneous_both_feet_target_lower"),
        "simultaneous_both_feet_target_lower",
        6,
    )
    simultaneous_both_feet_upper = _float_csv(
        metadata.get("simultaneous_both_feet_target_upper"),
        "simultaneous_both_feet_target_upper",
        6,
    )
    hand_lower = _float_csv(metadata.get("hand_target_lower"), "hand_target_lower", 6)
    hand_upper = _float_csv(metadata.get("hand_target_upper"), "hand_target_upper", 6)
    if metadata.get("foot_target_frame") != "robot_trunk_xyz_forward_left_up":
        raise PicoHybridPolicyContractError("unsupported foot target frame")
    if metadata.get("hand_target_frame") != "robot_trunk_xyz_forward_left_up":
        raise PicoHybridPolicyContractError("unsupported hand target frame")
    if (
        metadata.get("foot_target_units") != "metres"
        or metadata.get("hand_target_units") != "metres"
    ):
        raise PicoHybridPolicyContractError("body targets must be expressed in metres")
    if metadata.get("foot_target_semantics") != (
        "left_xyz_then_right_xyz_trunk_frame_offset_from_episode_reset_"
        "reference_metres_periodic_command_resampling_does_not_move_reference"
    ):
        raise PicoHybridPolicyContractError("unsupported foot target semantics")
    if metadata.get("hand_target_semantics") != (
        "left_xyz_then_right_xyz_then_left_right_active_flags_"
        "trunk_frame_offset_from_episode_reset_reference_metres_"
        "periodic_command_resampling_does_not_move_reference"
    ):
        raise PicoHybridPolicyContractError("unsupported hand target semantics")
    if (
        metadata.get("simultaneous_both_feet_target_semantics")
        != EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_SEMANTICS
    ):
        raise PicoHybridPolicyContractError(
            "unsupported simultaneous-both-feet target semantics"
        )
    if metadata.get("simultaneous_both_feet_requires_zero_twist") != "true":
        raise PicoHybridPolicyContractError(
            "simultaneous-both-feet targets must require zero twist"
        )
    for name, actual, expected in (
        ("foot_target_lower", foot_lower, EXPECTED_FOOT_TARGET_LOWER),
        ("foot_target_upper", foot_upper, EXPECTED_FOOT_TARGET_UPPER),
        (
            "simultaneous_both_feet_target_lower",
            simultaneous_both_feet_lower,
            EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_LOWER,
        ),
        (
            "simultaneous_both_feet_target_upper",
            simultaneous_both_feet_upper,
            EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_UPPER,
        ),
        ("hand_target_lower", hand_lower, EXPECTED_HAND_TARGET_LOWER),
        ("hand_target_upper", hand_upper, EXPECTED_HAND_TARGET_UPPER),
    ):
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-9):
            raise PicoHybridPolicyContractError(
                f"{name} does not match the deployed command contract"
            )

    return _PolicyContract(
        input_name=inputs[0].name,
        observation_joint_names=observation_joints,
        observation_default_joint_pos=expected_observation_defaults,
        action_joint_names=action_joints,
        action_default_joint_pos=EXPECTED_ACTION_DEFAULT_JOINT_POS,
        action_scale=EXPECTED_ACTION_SCALE,
        soft_lower=EXPECTED_SOFT_JOINT_POS_LOWER,
        soft_upper=EXPECTED_SOFT_JOINT_POS_UPPER,
        raw_action_soft_lower=raw_action_soft_lower,
        raw_action_soft_upper=raw_action_soft_upper,
        actor_raw_action_lower=EXPECTED_ACTOR_RAW_ACTION_LOWER,
        actor_raw_action_upper=EXPECTED_ACTOR_RAW_ACTION_UPPER,
        foot_lower=foot_lower,
        foot_upper=foot_upper,
        simultaneous_both_feet_lower=simultaneous_both_feet_lower,
        simultaneous_both_feet_upper=simultaneous_both_feet_upper,
        hand_lower=hand_lower,
        hand_upper=hand_upper,
        checkpoint_filename=checkpoint_filename,
        checkpoint_iteration=checkpoint_iteration,
        checkpoint_completed_updates=checkpoint_completed_updates,
        checkpoint_sha256=checkpoint_sha256,
        training_provenance_sha256=training_provenance_sha256,
        training_source_tree_sha256=training_source_tree_sha256,
        training_resume_source_checkpoint_sha256=(
            training_resume_source_checkpoint_sha256
        ),
        training_resume_source_checkpoint_iteration=(
            training_resume_source_checkpoint_iteration
        ),
        migration_source_checkpoint_sha256=migration_source_checkpoint_sha256,
        migration_source_checkpoint_iteration=migration_source_checkpoint_iteration,
        migration_source_training_provenance_sha256=(
            migration_source_training_provenance_sha256
        ),
        migration_source_tree_sha256=migration_source_tree_sha256,
        migration_source_gate_sha256=migration_source_gate_sha256,
        migration_state_transfer=migration_state_transfer,
        migration_source_optimizer_learning_rate=(
            migration_source_optimizer_learning_rate
        ),
        training_fixed_learning_rate=training_fixed_learning_rate,
        acceptance_receipt_sha256=acceptance_receipt_sha256,
        acceptance_evaluator_source_sha256=(acceptance_evaluator_source_sha256),
        safe_velocity_source_checkpoint_sha256=(safe_velocity_source_checkpoint_sha256),
        safe_velocity_source_checkpoint_iteration=(
            safe_velocity_source_checkpoint_iteration
        ),
        safe_velocity_acceptance_receipt_sha256=(
            safe_velocity_acceptance_receipt_sha256
        ),
    )


def _strict_json_metadata(metadata: Mapping[str, str], name: str) -> Any:
    value = metadata.get(name)
    if value is None:
        raise PicoHybridPolicyContractError(f"missing ONNX metadata: {name}")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PicoHybridPolicyContractError(f"{name} must contain strict JSON") from exc


def _exact_json_value(actual: Any, expected: Any) -> bool:
    """Compare JSON trees without Python's bool/int/float equality coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_value(received, required)
            for received, required in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _require_exact_metadata(
    metadata: Mapping[str, str], name: str, expected: str
) -> str:
    actual = metadata.get(name)
    if actual != expected:
        raise PicoHybridPolicyContractError(
            f"{name} does not match the contract-v12 deployment contract"
        )
    return actual


def _require_v12_bounded_metric(
    metadata: Mapping[str, str], name: str, *, lower: float = 0.0, upper: float
) -> float:
    value = metadata.get(name)
    if value is None:
        raise PicoHybridPolicyContractError(f"missing ONNX metadata: {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PicoHybridPolicyContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < lower or result > upper:
        raise PicoHybridPolicyContractError(
            f"{name} is outside the authenticated contract-v12 gate tolerance"
        )
    return result


def _require_v12_float32_evidence_vector(
    metadata: Mapping[str, str], name: str
) -> tuple[float, ...]:
    values = _float_json_vector(metadata.get(name), name, EXPECTED_ACTION_WIDTH)
    with np.errstate(over="ignore", invalid="ignore"):
        float32_values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(float32_values).all():
        raise PicoHybridPolicyContractError(
            f"{name} contains a value outside the finite float32 action domain"
        )
    return values


def _parse_v12_raw_action_envelope(
    metadata: Mapping[str, str], action_joints: tuple[str, ...]
) -> tuple[tuple[float, ...], ...]:
    """Authenticate final tracking evidence and its gross-anomaly guard.

    The guard is not a joint-limit clamp.  Its symmetric per-joint bounds are
    derived only from the final, hash-bound v12/source/delta tracking extrema.
    A runtime escape raises before any target write so the selector can execute
    the legacy walk policy in the same control cycle.
    """

    if (
        _canonical_nonnegative_int(
            metadata.get("v12_raw_action_envelope_schema_version"),
            "v12_raw_action_envelope_schema_version",
        )
        != EXPECTED_V12_RAW_ACTION_ENVELOPE_SCHEMA_VERSION
    ):
        raise PicoHybridPolicyContractError(
            "unsupported contract-v12 raw-action envelope schema"
        )
    joint_names = _strict_json_metadata(metadata, "v12_raw_action_joint_names_json")
    if joint_names != list(action_joints):
        raise PicoHybridPolicyContractError(
            "v12 raw-action envelope joint order drifted"
        )

    triplets: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = []
    for prefix in (
        "v12_raw_action",
        "v12_source_raw_action",
        "v12_learned_source_delta",
    ):
        minimum = _require_v12_float32_evidence_vector(metadata, f"{prefix}_min_json")
        maximum = _require_v12_float32_evidence_vector(metadata, f"{prefix}_max_json")
        absolute_maximum = _require_v12_float32_evidence_vector(
            metadata, f"{prefix}_absmax_json"
        )
        for index, (lower, upper, absolute) in enumerate(
            zip(minimum, maximum, absolute_maximum, strict=True)
        ):
            expected_absolute = max(abs(lower), abs(upper))
            if lower > upper:
                raise PicoHybridPolicyContractError(
                    f"{prefix} evidence minimum exceeds maximum at action index {index}"
                )
            if absolute < 0.0 or absolute != expected_absolute:
                raise PicoHybridPolicyContractError(
                    f"{prefix} absolute maximum is inconsistent at action index {index}"
                )
        triplets.append((minimum, maximum, absolute_maximum))

    _require_exact_metadata(
        metadata,
        "runtime_raw_action_guard_formula",
        EXPECTED_V12_RAW_ACTION_GUARD_FORMULA,
    )
    _require_exact_finite_scalar(
        metadata,
        "runtime_raw_action_guard_multiplier",
        EXPECTED_V12_RAW_ACTION_GUARD_MULTIPLIER,
    )
    _require_exact_metadata(
        metadata,
        "runtime_raw_action_guard_semantics",
        EXPECTED_V12_RAW_ACTION_GUARD_SEMANTICS,
    )
    guard = _require_v12_float32_evidence_vector(
        metadata, "runtime_raw_action_guard_absmax_json"
    )
    v12_absolute = triplets[0][2]
    source_absolute = triplets[1][2]
    delta_absolute = triplets[2][2]
    expected_guard = tuple(
        max(v12_value, source_value + delta_value)
        * EXPECTED_V12_RAW_ACTION_GUARD_MULTIPLIER
        for v12_value, source_value, delta_value in zip(
            v12_absolute, source_absolute, delta_absolute, strict=True
        )
    )
    if not all(math.isfinite(value) for value in expected_guard):
        raise PicoHybridPolicyContractError(
            "v12 raw-action guard derivation is non-finite"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        expected_guard_float32 = np.asarray(expected_guard, dtype=np.float32)
    if not np.isfinite(expected_guard_float32).all():
        raise PicoHybridPolicyContractError(
            "v12 raw-action guard is outside the finite float32 domain"
        )
    for index, (actual, expected) in enumerate(zip(guard, expected_guard, strict=True)):
        if actual != expected:
            raise PicoHybridPolicyContractError(
                "runtime raw-action guard does not match the authenticated "
                f"formula at action index {index}"
            )

    return (
        *triplets[0],
        *triplets[1],
        *triplets[2],
        guard,
    )


def _parse_v12_contract(session: Any) -> _PolicyContract:
    """Parse the raw-action v12 contract without relaxing contract v10.

    V12 is a separate, hash-bound contract.  In particular, none of the v10
    bounded-distribution metadata is interpreted as v12 authority: the source
    legacy checkpoint and closed-loop probe identities are compiled in here,
    and a canonical v12 boundary must carry passing locomotion and ONNX gates.
    """

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise PicoHybridPolicyContractError(
            "policy must have exactly one input and output"
        )
    if _fixed_width(inputs[0], "policy input") != EXPECTED_OBSERVATION_WIDTH:
        raise PicoHybridPolicyContractError("policy input width must be exactly 83")
    if _fixed_width(outputs[0], "policy output") != EXPECTED_ACTION_WIDTH:
        raise PicoHybridPolicyContractError("policy output width must be exactly 18")
    if inputs[0].name != "obs" or outputs[0].name != "actions":
        raise PicoHybridPolicyContractError(
            "contract-v12 tensor names must be obs -> actions"
        )
    if (
        getattr(inputs[0], "type", None) != "tensor(float)"
        or getattr(outputs[0], "type", None) != "tensor(float)"
    ):
        raise PicoHybridPolicyContractError(
            "contract-v12 tensors must use float32 elements"
        )

    metadata = session.get_modelmeta().custom_metadata_map
    _require_exact_metadata(metadata, "policy_type", EXPECTED_POLICY_TYPE)
    _require_exact_metadata(
        metadata,
        "microban_teleop_training_contract_version",
        EXPECTED_V12_TRAINING_CONTRACT_VERSION,
    )
    _require_exact_metadata(
        metadata,
        "microban_teleop_recipe_revision",
        EXPECTED_V12_RECIPE_REVISION,
    )

    filename = metadata.get("checkpoint_filename", "")
    match = _CHECKPOINT_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise PicoHybridPolicyContractError(
            "checkpoint_filename must be canonical model_N.pt"
        )
    checkpoint_iteration = _canonical_nonnegative_int(
        metadata.get("checkpoint_iteration"), "checkpoint_iteration"
    )
    if checkpoint_iteration != int(match.group(1)):
        raise PicoHybridPolicyContractError(
            "checkpoint_iteration does not match checkpoint_filename"
        )
    _require_exact_metadata(
        metadata,
        "checkpoint_iteration_semantics",
        "zero_based_completed_update_index_from_model_filename",
    )
    checkpoint_completed_updates = _canonical_nonnegative_int(
        metadata.get("checkpoint_completed_updates"),
        "checkpoint_completed_updates",
    )
    if checkpoint_completed_updates != checkpoint_iteration + 1:
        raise PicoHybridPolicyContractError(
            "checkpoint_completed_updates must equal checkpoint_iteration + 1"
        )
    if (
        checkpoint_iteration != EXPECTED_V12_FINAL_CHECKPOINT_ITERATION
        or checkpoint_completed_updates != EXPECTED_V12_FINAL_COMPLETED_UPDATES
    ):
        raise PicoHybridPolicyContractError(
            "contract-v12 hardware deployment requires final model_14999.pt "
            "at the 15000-update boundary"
        )
    checkpoint_sha256 = _require_lowercase_sha256(metadata, "checkpoint_sha256")

    _require_exact_metadata(metadata, "deployment_accepted", "true")
    if (
        _canonical_nonnegative_int(
            metadata.get("v12_stage_gate_schema_version"),
            "v12_stage_gate_schema_version",
        )
        != 2
    ):
        raise PicoHybridPolicyContractError("unsupported v12 stage gate schema")
    _require_exact_metadata(metadata, "v12_stage_gate_name", EXPECTED_V12_STAGE_GATE)
    _require_exact_metadata(metadata, "v12_stage_gate_status", "pass")
    _require_exact_metadata(metadata, "v12_stage_gate_canonical_boundary", "true")
    stage_gate_sha256 = _require_lowercase_sha256(metadata, "v12_stage_gate_sha256")
    locomotion_report_sha256 = _require_lowercase_sha256(
        metadata, "v12_locomotion_report_sha256"
    )
    onnx_report_sha256 = _require_lowercase_sha256(metadata, "v12_onnx_report_sha256")
    tracking_report_sha256 = _require_lowercase_sha256(
        metadata, "v12_tracking_report_sha256"
    )
    _require_exact_metadata(
        metadata, "v12_tracking_profile", EXPECTED_V12_TRACKING_PROFILE
    )
    if metadata.get("v12_stage_gate_checkpoint_sha256") != checkpoint_sha256:
        raise PicoHybridPolicyContractError(
            "v12 stage gate checkpoint SHA-256 does not match the exported checkpoint"
        )
    if (
        _canonical_nonnegative_int(
            metadata.get("v12_stage_gate_checkpoint_iteration"),
            "v12_stage_gate_checkpoint_iteration",
        )
        != checkpoint_iteration
    ):
        raise PicoHybridPolicyContractError(
            "v12 stage gate iteration does not match the exported checkpoint"
        )
    if (
        _canonical_nonnegative_int(
            metadata.get("v12_stage_gate_completed_updates"),
            "v12_stage_gate_completed_updates",
        )
        != checkpoint_completed_updates
    ):
        raise PicoHybridPolicyContractError(
            "v12 stage gate update count does not match the exported checkpoint"
        )

    if (
        _canonical_nonnegative_int(
            metadata.get("v12_bootstrap_provenance_schema_version"),
            "v12_bootstrap_provenance_schema_version",
        )
        != EXPECTED_V12_BOOTSTRAP_PROVENANCE_SCHEMA_VERSION
    ):
        raise PicoHybridPolicyContractError(
            "unsupported v12 bootstrap provenance schema"
        )
    _require_exact_metadata(
        metadata,
        "v12_bootstrap_mapping_version",
        EXPECTED_V12_BOOTSTRAP_MAPPING_VERSION,
    )
    source_checkpoint_sha256 = _require_exact_sha256(
        metadata,
        "v12_legacy_source_checkpoint_sha256",
        EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256,
    )
    source_iteration = _canonical_nonnegative_int(
        metadata.get("v12_legacy_source_checkpoint_iteration"),
        "v12_legacy_source_checkpoint_iteration",
    )
    if source_iteration != EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_ITERATION:
        raise PicoHybridPolicyContractError("v12 legacy source iteration drifted")
    probe_sha256 = _require_exact_sha256(
        metadata,
        "v12_legacy_probe_sha256",
        EXPECTED_V12_LEGACY_PROBE_SHA256,
    )
    for name, expected in (
        ("v12_legacy_probe_scenario_count", 9),
        ("v12_legacy_probe_steps_per_scenario", 300),
        ("v12_legacy_probe_settle_steps", 50),
        ("v12_legacy_probe_seed", 42),
    ):
        if _canonical_nonnegative_int(metadata.get(name), name) != expected:
            raise PicoHybridPolicyContractError(f"{name} drifted")

    mapping = _strict_json_metadata(metadata, "v12_source_to_target_columns_json")
    expected_mapping = [list(pair) for pair in EXPECTED_V12_SOURCE_TO_TARGET_COLUMNS]
    if mapping != expected_mapping:
        raise PicoHybridPolicyContractError(
            "v12 source-to-target observation mapping drifted"
        )
    extra_columns = _strict_json_metadata(
        metadata, "v12_extra_observation_columns_json"
    )
    if extra_columns != list(EXPECTED_V12_EXTRA_OBSERVATION_COLUMNS):
        raise PicoHybridPolicyContractError(
            "v12 teleop-only observation columns drifted"
        )
    topology = _strict_json_metadata(metadata, "v12_actor_topology_json")
    if topology != list(EXPECTED_V12_ACTOR_TOPOLOGY):
        raise PicoHybridPolicyContractError("v12 actor topology drifted")
    _require_exact_finite_scalar(
        metadata, "v12_normalizer_eps", EXPECTED_V12_NORMALIZER_EPS
    )
    _require_exact_metadata(
        metadata,
        "v12_normalizer_semantics",
        EXPECTED_V12_NORMALIZER_SEMANTICS,
    )
    _require_exact_metadata(
        metadata,
        "v12_trainable_actor_parameters",
        "mlp.0.weight_extra_columns_only",
    )
    _require_exact_metadata(
        metadata,
        "adapter_gradient_schedule_revision",
        EXPECTED_V12_ADAPTER_GRADIENT_SCHEDULE_REVISION,
    )
    active_columns = _strict_json_metadata(
        metadata, "v12_active_actor_columns_at_save_json"
    )
    if active_columns != list(EXPECTED_V12_EXTRA_OBSERVATION_COLUMNS):
        raise PicoHybridPolicyContractError(
            "final v12 checkpoint did not enable all teleop adapter columns"
        )
    _require_exact_metadata(metadata, "v12_frozen_legacy_tensors_verified", "true")

    _require_exact_metadata(
        metadata, "v12_locomotion_gate", EXPECTED_V12_LOCOMOTION_GATE
    )
    _require_exact_metadata(metadata, "v12_locomotion_status", "pass")
    _require_exact_finite_scalar(
        metadata,
        "v12_actual_dynamic_soft_limit_overshoot_max_deg",
        EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_DEG,
    )
    _require_exact_finite_scalar(
        metadata,
        "v12_actual_dynamic_soft_limit_overshoot_max_rad",
        EXPECTED_V12_ACTUAL_DYNAMIC_SOFT_LIMIT_OVERSHOOT_MAX_RAD,
    )
    _require_exact_finite_scalar(
        metadata,
        "v12_commanded_target_soft_limit_excess_max_rad",
        EXPECTED_V12_COMMANDED_TARGET_SOFT_LIMIT_EXCESS_MAX_RAD,
    )
    for name, expected in (
        ("v12_locomotion_seed", 42),
        ("v12_locomotion_scenario_count", 9),
        ("v12_locomotion_steps_per_scenario", 300),
        ("v12_locomotion_settle_steps", 50),
        ("v12_locomotion_fall_scenario_count", 0),
        ("v12_locomotion_nonfinite_scenario_count", 0),
        ("v12_locomotion_actual_soft_limit_violation_scenario_count", 0),
        ("v12_locomotion_directionally_correct_scenario_count", 8),
        ("v12_locomotion_directional_scenario_count", 8),
    ):
        if _canonical_nonnegative_int(metadata.get(name), name) != expected:
            raise PicoHybridPolicyContractError(f"{name} does not describe a pass")
    _require_exact_metadata(
        metadata, "v12_locomotion_raw_action_recurrence_all_steps", "true"
    )

    _require_exact_metadata(metadata, "v12_onnx_gate", EXPECTED_V12_ONNX_GATE)
    _require_exact_metadata(metadata, "v12_onnx_verified", "true")
    _require_exact_metadata(
        metadata, "v12_onnx_parity_teleop_columns", "random_finite_not_zeroed"
    )
    if (
        _canonical_nonnegative_int(
            metadata.get("v12_onnx_parity_seed"), "v12_onnx_parity_seed"
        )
        != EXPECTED_V12_PARITY_SEED
    ):
        raise PicoHybridPolicyContractError("unsupported v12 ONNX parity seed")
    if (
        _canonical_nonnegative_int(
            metadata.get("v12_onnx_parity_sample_count"),
            "v12_onnx_parity_sample_count",
        )
        != EXPECTED_V12_PARITY_SAMPLE_COUNT
    ):
        raise PicoHybridPolicyContractError("unsupported v12 ONNX parity sample count")
    _require_exact_finite_scalar(
        metadata, "v12_onnx_parity_atol", EXPECTED_V12_PARITY_ATOL
    )
    for name in (
        "v12_onnx_reference_max_abs_error",
        "v12_onnxruntime_cpu_max_abs_error",
        "v12_neutral_legacy_parity_max_abs_error",
    ):
        _require_v12_bounded_metric(metadata, name, upper=EXPECTED_V12_PARITY_ATOL)
    if (
        _canonical_nonnegative_int(
            metadata.get("v12_neutral_legacy_parity_sample_count"),
            "v12_neutral_legacy_parity_sample_count",
        )
        != 10_000
    ):
        raise PicoHybridPolicyContractError(
            "v12 neutral legacy parity sample count drifted"
        )

    if metadata.get("observation_schema_version") != EXPECTED_SCHEMA_VERSION:
        raise PicoHybridPolicyContractError("unsupported observation schema version")
    if metadata.get("observation_width") != str(EXPECTED_OBSERVATION_WIDTH):
        raise PicoHybridPolicyContractError("observation_width metadata must be 83")
    if metadata.get("action_width") != str(EXPECTED_ACTION_WIDTH):
        raise PicoHybridPolicyContractError("action_width metadata must be 18")
    observation_schema = _strict_json_metadata(metadata, "observation_schema_json")
    if observation_schema != [
        [name, width]
        for name, width in zip(
            EXPECTED_OBSERVATION_TERMS, (3, 3, 21, 21, 18, 3, 6, 8), strict=True
        )
    ]:
        raise PicoHybridPolicyContractError("contract-v12 observation schema drifted")
    if _split_csv(metadata.get("observation_names"), "observation_names") != (
        EXPECTED_OBSERVATION_TERMS
    ):
        raise PicoHybridPolicyContractError("unsafe observation order")
    observation_joints = _split_csv(
        metadata.get("observation_joint_names"), "observation_joint_names"
    )
    if observation_joints != EXPECTED_V12_OBSERVATION_JOINT_NAMES:
        raise PicoHybridPolicyContractError("unsafe observation joint order")
    action_joints = _split_csv(metadata.get("action_joint_names"), "action_joint_names")
    if action_joints != tuple(OBSERVATION_DOF_ORDER):
        raise PicoHybridPolicyContractError("unsafe action joint order")
    (
        v12_raw_action_minimum,
        v12_raw_action_maximum,
        v12_raw_action_absolute_maximum,
        v12_source_raw_action_minimum,
        v12_source_raw_action_maximum,
        v12_source_raw_action_absolute_maximum,
        v12_learned_source_delta_minimum,
        v12_learned_source_delta_maximum,
        v12_learned_source_delta_absolute_maximum,
        v12_runtime_raw_action_guard_absolute_maximum,
    ) = _parse_v12_raw_action_envelope(metadata, action_joints)

    if metadata.get("base_ang_vel_frame") != "robot_body_xyz":
        raise PicoHybridPolicyContractError("base_ang_vel_frame must be robot_body_xyz")
    if metadata.get("base_ang_vel_units") != "rad_s":
        raise PicoHybridPolicyContractError("base_ang_vel_units must be rad_s")
    if _split_csv(
        metadata.get("locomotion_command_order"), "locomotion_command_order"
    ) != (
        "linear_velocity_x",
        "linear_velocity_y",
        "angular_velocity_z",
    ):
        raise PicoHybridPolicyContractError("unsupported locomotion command order")
    if _split_csv(
        metadata.get("locomotion_command_units"), "locomotion_command_units"
    ) != ("m_s", "m_s", "rad_s"):
        raise PicoHybridPolicyContractError("unsupported locomotion command units")
    _require_exact_metadata(
        metadata,
        "locomotion_command_frame",
        "robot_body_forward_left_yaw_up",
    )
    _require_exact_metadata(
        metadata,
        "previous_action_semantics",
        EXPECTED_V12_PREVIOUS_ACTION_SEMANTICS,
    )
    _require_exact_metadata(
        metadata,
        "action_target_semantics",
        "default_joint_pos_plus_raw_action_times_scale",
    )
    _require_exact_metadata(
        metadata, "action_clip_semantics", EXPECTED_V12_ACTION_CLIP_SEMANTICS
    )
    _require_exact_metadata(
        metadata,
        "action_distribution_semantics",
        EXPECTED_V12_ACTION_DISTRIBUTION_SEMANTICS,
    )
    _require_exact_metadata(
        metadata,
        "runtime_action_semantics",
        EXPECTED_V12_RUNTIME_ACTION_SEMANTICS,
    )
    _require_exact_finite_scalar(metadata, "control_hz", 50.0)

    observation_defaults = _float_csv(
        metadata.get("observation_default_joint_pos"),
        "observation_default_joint_pos",
        len(observation_joints),
    )
    action_defaults = _float_csv(
        metadata.get("default_joint_pos"), "default_joint_pos", len(action_joints)
    )
    action_scale = _float_csv(
        metadata.get("action_scale"), "action_scale", len(action_joints)
    )
    expected_observation_defaults = tuple(
        float(PICO_TELEOP_HOME_POSE[name]) for name in observation_joints
    )
    _require_fixed_metadata_vector(
        "observation_default_joint_pos",
        observation_defaults,
        expected_observation_defaults,
    )
    _require_fixed_metadata_vector(
        "default_joint_pos", action_defaults, EXPECTED_ACTION_DEFAULT_JOINT_POS
    )
    _require_fixed_metadata_vector("action_scale", action_scale, EXPECTED_ACTION_SCALE)

    # The current v12 deployment exporter does not serialize soft limits because
    # its actor semantics intentionally have no environment target clip.  Older
    # base graphs and future exporters may nevertheless carry both vectors.  If
    # present, authenticate them against the same compiled contract as v10;
    # otherwise the constructor validates the compiled fallback before inference.
    soft_limit_keys = ("soft_joint_pos_lower", "soft_joint_pos_upper")
    soft_limit_presence = tuple(key in metadata for key in soft_limit_keys)
    if any(soft_limit_presence) and not all(soft_limit_presence):
        raise PicoHybridPolicyContractError(
            "contract-v12 soft-limit metadata must provide both lower and upper"
        )
    if all(soft_limit_presence):
        metadata_soft_lower = _float_csv(
            metadata.get("soft_joint_pos_lower"),
            "soft_joint_pos_lower",
            len(action_joints),
        )
        metadata_soft_upper = _float_csv(
            metadata.get("soft_joint_pos_upper"),
            "soft_joint_pos_upper",
            len(action_joints),
        )
        _require_fixed_metadata_vector(
            "soft_joint_pos_lower",
            metadata_soft_lower,
            EXPECTED_SOFT_JOINT_POS_LOWER,
        )
        _require_fixed_metadata_vector(
            "soft_joint_pos_upper",
            metadata_soft_upper,
            EXPECTED_SOFT_JOINT_POS_UPPER,
        )

    foot_lower = _float_csv(metadata.get("foot_target_lower"), "foot_target_lower", 6)
    foot_upper = _float_csv(metadata.get("foot_target_upper"), "foot_target_upper", 6)
    both_lower = _float_csv(
        metadata.get("simultaneous_both_feet_target_lower"),
        "simultaneous_both_feet_target_lower",
        6,
    )
    both_upper = _float_csv(
        metadata.get("simultaneous_both_feet_target_upper"),
        "simultaneous_both_feet_target_upper",
        6,
    )
    hand_lower = _float_csv(metadata.get("hand_target_lower"), "hand_target_lower", 6)
    hand_upper = _float_csv(metadata.get("hand_target_upper"), "hand_target_upper", 6)
    hand_target_fk = _strict_json_metadata(metadata, "hand_target_fk")
    if not _exact_json_value(hand_target_fk, EXPECTED_V12_HAND_TARGET_FK):
        raise PicoHybridPolicyContractError(
            "hand_target_fk does not match the contract-v12 deployment contract"
        )
    _require_exact_metadata(
        metadata, "foot_target_frame", "robot_trunk_xyz_forward_left_up"
    )
    _require_exact_metadata(
        metadata, "hand_target_frame", "robot_trunk_xyz_forward_left_up"
    )
    _require_exact_metadata(metadata, "foot_target_units", "metres")
    _require_exact_metadata(metadata, "hand_target_units", "metres")
    _require_exact_metadata(
        metadata,
        "foot_target_semantics",
        "left_xyz_then_right_xyz_trunk_frame_offset_from_episode_reset_"
        "reference_metres_periodic_command_resampling_does_not_move_reference",
    )
    _require_exact_metadata(
        metadata,
        "hand_target_semantics",
        "left_xyz_then_right_xyz_then_left_right_active_flags_"
        "trunk_frame_offset_from_episode_reset_reference_metres_"
        "periodic_command_resampling_does_not_move_reference",
    )
    _require_exact_metadata(
        metadata,
        "simultaneous_both_feet_target_semantics",
        EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_SEMANTICS,
    )
    _require_exact_metadata(
        metadata, "simultaneous_both_feet_requires_zero_twist", "true"
    )
    for name, actual, expected in (
        ("foot_target_lower", foot_lower, EXPECTED_FOOT_TARGET_LOWER),
        ("foot_target_upper", foot_upper, EXPECTED_FOOT_TARGET_UPPER),
        (
            "simultaneous_both_feet_target_lower",
            both_lower,
            EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_LOWER,
        ),
        (
            "simultaneous_both_feet_target_upper",
            both_upper,
            EXPECTED_SIMULTANEOUS_BOTH_FEET_TARGET_UPPER,
        ),
        ("hand_target_lower", hand_lower, EXPECTED_HAND_TARGET_LOWER),
        ("hand_target_upper", hand_upper, EXPECTED_HAND_TARGET_UPPER),
    ):
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-9):
            raise PicoHybridPolicyContractError(
                f"{name} does not match the deployed command contract"
            )

    return _PolicyContract(
        input_name=inputs[0].name,
        observation_joint_names=observation_joints,
        observation_default_joint_pos=expected_observation_defaults,
        action_joint_names=action_joints,
        action_default_joint_pos=EXPECTED_ACTION_DEFAULT_JOINT_POS,
        action_scale=EXPECTED_ACTION_SCALE,
        soft_lower=EXPECTED_SOFT_JOINT_POS_LOWER,
        soft_upper=EXPECTED_SOFT_JOINT_POS_UPPER,
        raw_action_soft_lower=(),
        raw_action_soft_upper=(),
        actor_raw_action_lower=(),
        actor_raw_action_upper=(),
        foot_lower=foot_lower,
        foot_upper=foot_upper,
        simultaneous_both_feet_lower=both_lower,
        simultaneous_both_feet_upper=both_upper,
        hand_lower=hand_lower,
        hand_upper=hand_upper,
        checkpoint_filename=filename,
        checkpoint_iteration=checkpoint_iteration,
        checkpoint_completed_updates=checkpoint_completed_updates,
        checkpoint_sha256=checkpoint_sha256,
        training_provenance_sha256=stage_gate_sha256,
        training_source_tree_sha256="",
        training_resume_source_checkpoint_sha256="",
        training_resume_source_checkpoint_iteration=-1,
        migration_source_checkpoint_sha256="",
        migration_source_checkpoint_iteration=-1,
        migration_source_training_provenance_sha256="",
        migration_source_tree_sha256="",
        migration_source_gate_sha256="",
        migration_state_transfer="",
        migration_source_optimizer_learning_rate=0.0,
        training_fixed_learning_rate=0.0,
        acceptance_receipt_sha256=locomotion_report_sha256,
        acceptance_evaluator_source_sha256="",
        safe_velocity_source_checkpoint_sha256=source_checkpoint_sha256,
        safe_velocity_source_checkpoint_iteration=source_iteration,
        safe_velocity_acceptance_receipt_sha256=probe_sha256,
        training_contract_version=EXPECTED_V12_TRAINING_CONTRACT_VERSION,
        runtime_action_semantics=EXPECTED_V12_RUNTIME_ACTION_SEMANTICS,
        v12_legacy_source_checkpoint_sha256=source_checkpoint_sha256,
        v12_legacy_probe_sha256=probe_sha256,
        v12_stage_gate_sha256=stage_gate_sha256,
        v12_locomotion_report_sha256=locomotion_report_sha256,
        v12_onnx_report_sha256=onnx_report_sha256,
        v12_tracking_report_sha256=tracking_report_sha256,
        v12_raw_action_minimum=v12_raw_action_minimum,
        v12_raw_action_maximum=v12_raw_action_maximum,
        v12_raw_action_absolute_maximum=v12_raw_action_absolute_maximum,
        v12_source_raw_action_minimum=v12_source_raw_action_minimum,
        v12_source_raw_action_maximum=v12_source_raw_action_maximum,
        v12_source_raw_action_absolute_maximum=(v12_source_raw_action_absolute_maximum),
        v12_learned_source_delta_minimum=v12_learned_source_delta_minimum,
        v12_learned_source_delta_maximum=v12_learned_source_delta_maximum,
        v12_learned_source_delta_absolute_maximum=(
            v12_learned_source_delta_absolute_maximum
        ),
        v12_runtime_raw_action_guard_absolute_maximum=(
            v12_runtime_raw_action_guard_absolute_maximum
        ),
    )


def _parse_contract(session: Any) -> _PolicyContract:
    try:
        metadata = session.get_modelmeta().custom_metadata_map
    except Exception as exc:
        raise PicoHybridPolicyContractError("failed to read ONNX metadata") from exc
    version = metadata.get("microban_teleop_training_contract_version")
    if version == EXPECTED_V12_TRAINING_CONTRACT_VERSION:
        return _parse_v12_contract(session)
    # Preserve the complete, independently reviewed v10 parser unchanged.  It
    # remains responsible for rejecting missing and historical v1-v9 models.
    return _parse_v10_contract(session)


def _validate_physical_motor_target_contract(contract: _PolicyContract) -> None:
    """Fail closed before inference if the actuator clamp is not well-defined.

    Contract v10 authenticates exporter-provided soft-limit metadata.  Contract
    v12 authenticates it when present and otherwise uses the robot-local
    constants.  This final constructor check is deliberately common to both so
    no live target can encounter malformed geometry after torque is active.
    """

    vectors = {
        "action_default_joint_pos": contract.action_default_joint_pos,
        "action_scale": contract.action_scale,
        "soft_joint_pos_lower": contract.soft_lower,
        "soft_joint_pos_upper": contract.soft_upper,
    }
    if (
        len(contract.action_joint_names) != EXPECTED_ACTION_WIDTH
        or len(set(contract.action_joint_names)) != EXPECTED_ACTION_WIDTH
        or any(len(values) != EXPECTED_ACTION_WIDTH for values in vectors.values())
    ):
        raise PicoHybridPolicyContractError(
            "physical motor-target guard has an invalid 18-joint geometry"
        )

    for index, name in enumerate(contract.action_joint_names):
        try:
            default = float(contract.action_default_joint_pos[index])
            scale = float(contract.action_scale[index])
            lower = float(contract.soft_lower[index])
            upper = float(contract.soft_upper[index])
            neutral = float(NEUTRAL_POSE[name])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise PicoHybridPolicyContractError(
                f"physical motor-target guard is malformed for {name}"
            ) from exc
        if not all(
            math.isfinite(value)
            for value in (default, scale, lower, upper, neutral)
        ):
            raise PicoHybridPolicyContractError(
                f"physical motor-target guard is non-finite for {name}"
            )
        if scale <= 0.0 or lower >= upper:
            raise PicoHybridPolicyContractError(
                f"physical motor-target guard has invalid bounds for {name}"
            )
        if not lower <= default <= upper:
            raise PicoHybridPolicyContractError(
                f"policy default is outside physical soft limits for {name}"
            )
        if not lower <= neutral <= upper:
            raise PicoHybridPolicyContractError(
                f"global neutral is outside physical soft limits for {name}"
            )


def _bounded_targets(
    values: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> list[float]:
    if len(values) != len(lower) or len(lower) != len(upper):
        raise PicoHybridPolicyRuntimeError("body target has the wrong width")
    result: list[float] = []
    for value, lo, hi in zip(values, lower, upper):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise PicoHybridPolicyRuntimeError(
                "body target contains a non-finite value"
            )
        # Clipping at the policy's training support is safer than extrapolating a
        # human-size or corrupt target into an unseen command.
        result.append(max(lo, min(hi, numeric)))
    return result


class PicoHybridMove(Move):
    """Run the independently trained 83-observation PICO policy.

    The existing ``hmd_head`` move continues to own head/neck joints.  This move
    owns exactly ``OBSERVATION_DOF_ORDER`` and uses the same smooth return and
    get-up hand-off behavior as ``WalkMove``.
    """

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        policy_path: str | Path = Path("src/agents") / AGENT_NAME,
        neutral_return_duration_s: float = 0.8,
        *,
        session: Any | None = None,
        gyro_transform: Callable[
            [Sequence[float]], Sequence[float]
        ] = sensor_gyro_to_body,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._neutral_return_duration_s = neutral_return_duration_s
        self._gyro_transform = gyro_transform
        self._session = session or ort.InferenceSession(str(policy_path))
        self._contract = _parse_contract(self._session)
        _validate_physical_motor_target_contract(self._contract)
        if (
            self._contract.training_contract_version
            == EXPECTED_V12_TRAINING_CONTRACT_VERSION
        ):
            self._compatibility_smoke_sample_count = (
                validate_v12_onnxruntime_compatibility(
                    self._session,
                    self._contract.input_name,
                    self._contract.v12_runtime_raw_action_guard_absolute_maximum,
                )
            )
        else:
            self._compatibility_smoke_sample_count = validate_onnxruntime_compatibility(
                self._session,
                self._contract.input_name,
                self._contract.actor_raw_action_lower,
                self._contract.actor_raw_action_upper,
                EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER,
                EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER,
            )
        self._last_action = np.zeros(EXPECTED_ACTION_WIDTH, dtype=np.float32)
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}

    def _physical_target(self, index: int, value: float) -> float:
        """Continuously saturate one finite actuator target to compiled limits."""

        numeric = float(value)
        if not math.isfinite(numeric):
            raise PicoHybridPolicyRuntimeError(
                "physical motor target became non-finite before clamping"
            )
        return max(
            self._contract.soft_lower[index],
            min(self._contract.soft_upper[index], numeric),
        )

    def _measured_action_positions(self, obs: Observation) -> dict[str, float]:
        """Return a complete finite measured pose before any command is written."""

        positions: dict[str, float] = {}
        for name in self._contract.action_joint_names:
            try:
                value = float(
                    obs.robot_state.motor_positions.get(
                        name, PICO_TELEOP_HOME_POSE[name]
                    )
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise PicoHybridPolicyRuntimeError(
                    f"motor position for {name} is not numeric"
                ) from exc
            if not math.isfinite(value):
                raise PicoHybridPolicyRuntimeError(
                    f"motor position for {name} is non-finite"
                )
            positions[name] = value
        return positions

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        measured_positions = self._measured_action_positions(obs)
        if self._controller is not None:
            ids = [MOTOR_TO_ID[name] for name in self._contract.action_joint_names]
            self._controller.sync_write_kp(ids, [KP_RL] * len(ids))
        for index, name in enumerate(self._contract.action_joint_names):
            command.target_angles[name] = self._physical_target(
                index, measured_positions[name]
            )
        self._last_action.fill(0.0)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self.state = MoveState.ACTIVE

    def _body_targets(self, obs: Observation) -> tuple[list[float], list[float]]:
        foot_mapping: Mapping[str, Sequence[float]] = obs.user_input.foot_target or {}
        foot_values: list[float] = []
        for side in ("left", "right"):
            foot_values.extend(foot_mapping.get(side, (0.0, 0.0, 0.0)))
        for start in (0, 3):
            vector = tuple(float(value) for value in foot_values[start : start + 3])
            if vector[2] <= SUPPORT_FOOT_FLOOR_BAND_M and vector != (
                0.0,
                0.0,
                0.0,
            ):
                raise PicoHybridPolicyRuntimeError(
                    "support-foot floor-band target must be exact XYZ zero"
                )
        active_feet = [
            any(abs(value) > 1.0e-12 for value in foot_values[start : start + 3])
            for start in (0, 3)
        ]
        if all(active_feet):
            live_lower = tuple(
                value * LIVE_BODY_TARGET_SAFETY_MARGIN
                for value in self._contract.simultaneous_both_feet_lower
            )
            live_upper = tuple(
                value * LIVE_BODY_TARGET_SAFETY_MARGIN
                for value in self._contract.simultaneous_both_feet_upper
            )
            if any(
                value < live_lower[index] or value > live_upper[index]
                for index, value in enumerate(foot_values)
            ):
                raise PicoHybridPolicyRuntimeError(
                    "simultaneous foot targets exceed their conservative live bound"
                )
            if any(
                float(obs.user_input.velocity[axis]) != 0.0
                for axis in ("vx", "vy", "vtheta")
            ):
                raise PicoHybridPolicyRuntimeError(
                    "simultaneous foot targets require zero locomotion command"
                )
        feet = _bounded_targets(
            foot_values, self._contract.foot_lower, self._contract.foot_upper
        )

        hand_mapping: Mapping[str, Sequence[float] | None] = (
            obs.user_input.hand_target or {}
        )
        hand_values: list[float] = []
        active: list[float] = []
        for side in ("left", "right"):
            target = hand_mapping.get(side)
            active.append(0.0 if target is None else 1.0)
            hand_values.extend((0.0, 0.0, 0.0) if target is None else target)
        hands = _bounded_targets(
            hand_values, self._contract.hand_lower, self._contract.hand_upper
        )
        hands.extend(active)
        return feet, hands

    def build_observation(self, obs: Observation) -> list[float]:
        try:
            gyro_body = [
                float(value) for value in self._gyro_transform(obs.robot_state.gyro)
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise PicoHybridPolicyRuntimeError(
                "failed to transform gyro to body frame"
            ) from exc
        gravity = [float(value) for value in obs.robot_state.projected_gravity]
        if len(gyro_body) != 3 or len(gravity) != 3:
            raise PicoHybridPolicyRuntimeError(
                "gyro and projected gravity must be 3-vectors"
            )

        values: list[float] = gyro_body + gravity
        for name, default in zip(
            self._contract.observation_joint_names,
            self._contract.observation_default_joint_pos,
        ):
            values.append(float(obs.robot_state.motor_positions[name]) - default)
        for name in self._contract.observation_joint_names:
            values.append(float(obs.robot_state.motor_velocities[name]))
        values.extend(float(value) for value in self._last_action)
        values.extend(
            float(obs.user_input.velocity[axis]) for axis in ("vx", "vy", "vtheta")
        )
        feet, hands = self._body_targets(obs)
        values.extend(feet)
        values.extend(hands)
        if len(values) != EXPECTED_OBSERVATION_WIDTH or not all(
            math.isfinite(value) for value in values
        ):
            raise PicoHybridPolicyRuntimeError(
                f"unsafe policy observation (width={len(values)})"
            )
        return values

    def step(self, obs: Observation, command: MotorCommand) -> None:
        # Match the existing fall gate.  Scheduler/get-up arbitration owns the
        # sustained-fall transition; this tick only holds measured action joints.
        gravity = obs.robot_state.projected_gravity
        if len(gravity) != 3 or not all(
            math.isfinite(float(value)) for value in gravity
        ):
            raise PicoHybridPolicyRuntimeError("projected gravity is invalid")
        if float(gravity[2]) > -0.5:
            measured_positions = self._measured_action_positions(obs)
            for index, name in enumerate(self._contract.action_joint_names):
                command.target_angles[name] = self._physical_target(
                    index, measured_positions[name]
                )
            return

        with np.errstate(over="ignore", invalid="ignore"):
            policy_input = np.asarray([self.build_observation(obs)], dtype=np.float32)
        if not np.isfinite(policy_input).all():
            raise PicoHybridPolicyRuntimeError(
                "policy observation is not finite in float32"
            )
        outputs = self._session.run(None, {self._contract.input_name: policy_input})
        if len(outputs) != 1:
            raise PicoHybridPolicyRuntimeError("policy returned more than one output")
        try:
            action = np.asarray(outputs[0], dtype=np.float64)
            finite = bool(np.isfinite(action).all())
        except (TypeError, ValueError, OverflowError) as exc:
            raise PicoHybridPolicyRuntimeError(
                "policy returned a non-numeric output"
            ) from exc
        if action.shape != (1, EXPECTED_ACTION_WIDTH) or not finite:
            raise PicoHybridPolicyRuntimeError(
                f"unsafe policy output shape/value: {action.shape}"
            )
        raw_action = action[0]
        if (
            self._contract.training_contract_version
            == EXPECTED_V12_TRAINING_CONTRACT_VERSION
        ):
            # Preserve the source locomotion MDP recurrence exactly: no actor
            # transform and the same raw float32 action recurs in the next
            # observation.  The separate actuator boundary continuously clamps
            # finite absolute MotorCommand targets to compiled physical limits;
            # saturation is not fed back into the actor recurrence.
            with np.errstate(over="ignore", invalid="ignore"):
                raw_action_float32 = raw_action.astype(np.float32)
            if not np.isfinite(raw_action_float32).all():
                raise PicoHybridPolicyRuntimeError(
                    "contract-v12 raw action is not finite in float32"
                )
            for index, (name, raw_value, absolute_maximum) in enumerate(
                zip(
                    self._contract.action_joint_names,
                    raw_action_float32,
                    self._contract.v12_runtime_raw_action_guard_absolute_maximum,
                    strict=True,
                )
            ):
                if abs(float(raw_value)) > absolute_maximum:
                    raise PicoHybridPolicyRuntimeError(
                        "contract-v12 raw action escaped the authenticated "
                        f"finite-amplitude guard for {name} at action index {index}"
                    )
            targets = [
                self._contract.action_default_joint_pos[index]
                + float(raw_action_float32[index]) * self._contract.action_scale[index]
                for index in range(EXPECTED_ACTION_WIDTH)
            ]
            if not all(math.isfinite(target) for target in targets):
                raise PicoHybridPolicyRuntimeError(
                    "contract-v12 raw action produced a non-finite target"
                )
            for index, (name, target) in enumerate(
                zip(self._contract.action_joint_names, targets, strict=True)
            ):
                command.target_angles[name] = self._physical_target(index, target)
            self._last_action = raw_action_float32.copy()
            return

        targets: list[float] = []
        effective_action = np.empty(EXPECTED_ACTION_WIDTH, dtype=np.float32)
        for index, name in enumerate(self._contract.action_joint_names):
            raw_value = float(raw_action[index])
            if not (
                self._contract.actor_raw_action_lower[index]
                < raw_value
                < self._contract.actor_raw_action_upper[index]
            ):
                raise PicoHybridPolicyRuntimeError(
                    "policy output reached or exceeded its guarded actor bound "
                    f"for {name}"
                )
            if not (
                EXPECTED_DETERMINISTIC_RAW_ACTION_LOWER[index]
                <= raw_value
                <= EXPECTED_DETERMINISTIC_RAW_ACTION_UPPER[index]
            ):
                raise PicoHybridPolicyRuntimeError(
                    "policy output escaped the current deterministic transform "
                    f"envelope for {name}"
                )
            target = (
                self._contract.action_default_joint_pos[index]
                + raw_value * self._contract.action_scale[index]
            )
            if not math.isfinite(target) or not (
                self._contract.soft_lower[index]
                < target
                < self._contract.soft_upper[index]
            ):
                raise PicoHybridPolicyRuntimeError(
                    "policy output produced a target outside the compiled-in "
                    f"soft limits for {name}"
                )
            clipped_target = max(
                self._contract.soft_lower[index],
                min(self._contract.soft_upper[index], target),
            )
            targets.append(clipped_target)
            # V2 observes the action that actually survived the absolute target
            # soft clip, expressed back in the actor's raw delta coordinates.
            # This is algebraically the same formula as the training MDP term.
            effective_action[index] = (
                clipped_target - self._contract.action_default_joint_pos[index]
            ) / self._contract.action_scale[index]
        # Validate the complete 18-joint output before writing any target so a
        # bad later joint cannot leave a partial motor command behind.
        for index, (name, target) in enumerate(
            zip(self._contract.action_joint_names, targets, strict=True)
        ):
            command.target_angles[name] = self._physical_target(index, target)
        self._last_action = effective_action

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if "getup" in obs.user_input.active_moves:
            self._finish_stop()
            return
        if self._stop_start_time_s is None:
            measured_positions = self._measured_action_positions(obs)
            self._stop_start_time_s = float(obs.robot_state.time_s)
            self._stop_start_angles = measured_positions
        elapsed = max(0.0, float(obs.robot_state.time_s) - self._stop_start_time_s)
        duration = max(1e-6, self._neutral_return_duration_s)
        fraction = min(1.0, elapsed / duration)
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        for index, name in enumerate(self._contract.action_joint_names):
            start = self._stop_start_angles[name]
            target = (
                NEUTRAL_POSE[name]
                if fraction >= 1.0
                else start + (NEUTRAL_POSE[name] - start) * blend
            )
            command.target_angles[name] = self._physical_target(
                index,
                target,
            )
        if fraction >= 1.0:
            if self._controller is not None:
                ids = [MOTOR_TO_ID[name] for name in self._contract.action_joint_names]
                self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
            self._finish_stop()

    def _finish_stop(self) -> None:
        self._last_action.fill(0.0)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        _ = obs
        self._last_action.fill(0.0)
        self._stop_start_time_s = None
        self._stop_start_angles = {}
