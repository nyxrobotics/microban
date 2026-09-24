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
EXPECTED_TRAINING_CONTRACT_VERSION = "8"
# Keep the recipe in one deployment-side constant: a deliberately promoted
# training recipe then requires one reviewed line change here.  It must match
# the exporter exactly; accepting a different marker would attach current
# runtime semantics to weights trained under another reward/config recipe.
EXPECTED_ACTOR_INITIALIZATION = (
    "clean_random_except_inward_shoulder_roll_v1_nonshoulder_std_1_v1"
)
EXPECTED_RECIPE_REVISION = (
    "v8g_clean_shoulder_std1_intermediate_commands_tracking_l1x2_v1"
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
EXPECTED_ONNX_PARITY_GATE_VERSION = "1"
EXPECTED_ONNX_PARITY_RUNTIME = "onnx.reference.ReferenceEvaluator"
EXPECTED_ONNX_PARITY_SEED = 20260924
EXPECTED_ONNX_PARITY_SAMPLE_COUNT = 16
EXPECTED_ONNX_PARITY_ATOL = 1e-5
EXPECTED_ONNX_PARITY_RTOL = 1e-4
EXPECTED_TRAINING_PROVENANCE_SCHEMA_VERSION = 1
EXPECTED_TRAINING_PROVENANCE_MODE = "canonical_v8_stage"
EXPECTED_FINAL_TRAINING_STAGE_START_BOUNDARY = 18_000
EXPECTED_FINAL_TRAINING_STAGE_TARGET_BOUNDARY = 20_000
EXPECTED_ACCEPTANCE_RECEIPT_SCHEMA_VERSION = 3
EXPECTED_ACCEPTANCE_EVALUATOR_REVISION = (
    "microban_teleop_deterministic_evaluator_v8_1"
)
EXPECTED_ACCEPTANCE_REVISION = "microban_teleop_acceptance_v8_1"
EXPECTED_ACCEPTANCE_NOMINAL_REPORT_COUNT = 3
EXPECTED_ACCEPTANCE_MOVING_HMD_REPORT_COUNT = 3

_CHECKPOINT_FILENAME_RE = re.compile(r"model_(0|[1-9][0-9]*)\.pt\Z")
_CHECKPOINT_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

# The exporter serializes numeric metadata to three decimal places.  Keep the
# full-precision, robot-side action contract here and only use the serialized
# form when checking the ONNX metadata.  Inference always uses these values,
# never model-provided limits, so altered metadata cannot widen motor targets.
EXPECTED_ACTION_DEFAULT_JOINT_POS = tuple(
    float(NEUTRAL_POSE[name]) for name in OBSERVATION_DOF_ORDER
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
) -> None:
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
) -> tuple[str, str, str, str]:
    """Require a canonical final-stage checkpoint and its schema-3 pass receipt.

    Diagnostic and automatic exports deliberately carry
    ``deployment_accepted=false``.  The robot must never infer deployability
    merely from a v8 recipe label or a final-looking checkpoint filename.
    """

    schema_version = _canonical_nonnegative_int(
        metadata.get("training_provenance_schema_version"),
        "training_provenance_schema_version",
    )
    if schema_version != EXPECTED_TRAINING_PROVENANCE_SCHEMA_VERSION:
        raise PicoHybridPolicyContractError(
            "unsupported training provenance schema version"
        )
    training_sha256 = _require_lowercase_sha256(
        metadata, "training_provenance_sha256"
    )
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
            "training provenance is not from the canonical v8 stage driver"
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
            "deployment requires canonical training stage 18000->20000"
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
    receipt_sha256 = _require_lowercase_sha256(
        metadata, "acceptance_receipt_sha256"
    )
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
        raise PicoHybridPolicyContractError(
            "unsupported acceptance evaluator revision"
        )
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
    )


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
    V7's exported deterministic graph must also keep every result strictly
    inside its open actor interval; reaching either endpoint fails the load.
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
    acceptance_receipt_sha256: str
    acceptance_evaluator_source_sha256: str


def _parse_contract(session: Any) -> _PolicyContract:
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
        training_provenance_sha256,
        training_source_tree_sha256,
        acceptance_receipt_sha256,
        acceptance_evaluator_source_sha256,
    ) = _require_final_deployment_provenance(
        metadata,
        checkpoint_iteration=checkpoint_iteration,
        checkpoint_completed_updates=checkpoint_completed_updates,
        checkpoint_sha256=checkpoint_sha256,
    )
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
        float(NEUTRAL_POSE[name]) for name in observation_joints
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
        acceptance_receipt_sha256=acceptance_receipt_sha256,
        acceptance_evaluator_source_sha256=(
            acceptance_evaluator_source_sha256
        ),
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

    def _measured_action_positions(self, obs: Observation) -> dict[str, float]:
        """Return a complete finite measured pose before any command is written."""

        positions: dict[str, float] = {}
        for name in self._contract.action_joint_names:
            try:
                value = float(
                    obs.robot_state.motor_positions.get(name, NEUTRAL_POSE[name])
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
        for name in self._contract.action_joint_names:
            command.target_angles[name] = measured_positions[name]
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
            for name in self._contract.action_joint_names:
                command.target_angles[name] = measured_positions[name]
            return

        policy_input = np.asarray([self.build_observation(obs)], dtype=np.float32)
        outputs = self._session.run(None, {self._contract.input_name: policy_input})
        if len(outputs) != 1:
            raise PicoHybridPolicyRuntimeError("policy returned more than one output")
        action = np.asarray(outputs[0], dtype=np.float64)
        if action.shape != (1, EXPECTED_ACTION_WIDTH) or not np.isfinite(action).all():
            raise PicoHybridPolicyRuntimeError(
                f"unsafe policy output shape/value: {action.shape}"
            )
        raw_action = action[0]
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
        for name, target in zip(
            self._contract.action_joint_names, targets, strict=True
        ):
            command.target_angles[name] = target
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
        for name in self._contract.action_joint_names:
            start = self._stop_start_angles[name]
            command.target_angles[name] = start + (NEUTRAL_POSE[name] - start) * blend
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
