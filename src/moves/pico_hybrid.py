# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Opt-in runtime for the Microban PICO hybrid teleoperation policy.

This move deliberately has a stricter contract than :mod:`moves.walk`.  A
PICO policy observes all 21 encoders and body-tracking targets but controls
only the 18 non-head joints.  Loading a legacy walking policy, or an export
whose observation order changed, therefore fails before motor torque is used.
"""

from __future__ import annotations

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
EXPECTED_TRAINING_CONTRACT_VERSION = "4"
EXPECTED_SCHEMA_VERSION = "2"
EXPECTED_PREVIOUS_ACTION_SEMANTICS = (
    "effective_action_after_absolute_target_soft_clip_in_raw_delta_coordinates"
)
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


def _canonical_nonnegative_int(value: str | None, name: str) -> int:
    if value is None or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise PicoHybridPolicyContractError(
            f"{name} must be a canonical non-negative integer"
        )
    return int(value)


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

    checkpoint_sha256 = metadata.get("checkpoint_sha256", "")
    if _CHECKPOINT_SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise PicoHybridPolicyContractError(
            "checkpoint_sha256 must be 64 lowercase hexadecimal characters"
        )
    return filename, iteration, completed_updates, checkpoint_sha256


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


def validate_onnxruntime_compatibility(session: Any, input_name: str) -> int:
    """Run the fixed corpus through ONNX Runtime and validate its outputs.

    Returns the number of observations executed.  This deliberately does not
    compare against PyTorch: the export gate owns that numerical parity check.
    """

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
        effective_action = np.empty(EXPECTED_ACTION_WIDTH, dtype=np.float32)
        for index, name in enumerate(self._contract.action_joint_names):
            target = (
                self._contract.action_default_joint_pos[index]
                + float(raw_action[index]) * self._contract.action_scale[index]
            )
            clipped_target = max(
                self._contract.soft_lower[index],
                min(self._contract.soft_upper[index], target),
            )
            command.target_angles[name] = clipped_target
            # V2 observes the action that actually survived the absolute target
            # soft clip, expressed back in the actor's raw delta coordinates.
            # This is algebraically the same formula as the training MDP term.
            effective_action[index] = (
                clipped_target - self._contract.action_default_joint_pos[index]
            ) / self._contract.action_scale[index]
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
