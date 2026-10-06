# SPDX-License-Identifier: GPL-3.0-or-later
"""The one contract every learned policy (walk, get-up, PICO) is checked against.

The training repository (mjlab_microban) writes the ONNX metadata described
here and ``src/agents/manifest.json``; the robot reads them with this module
only.  Compatibility is established by

* the contract version ``POLICY_CONTRACT`` (raised by hand on both sides when
  an observation, action or target meaning changes) and the reviewed recipe
  ids ``RECIPES``;
* the HOME stamp, which must be this robot's config/home_pose.yaml;
* a startup self-test: observations recorded from the exported checkpoint's
  evaluation rollouts are run through ONNX Runtime here and must reproduce the
  actions the training actor produced for them;
* the manifest, which binds the installed files of one release together (file
  and checkpoint SHA-256, and PICO's frozen walker to the installed walk).

Run-specific values live only in the manifest: the installer writes data
files and never edits robot code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from constants import NEUTRAL_POSE, OBSERVATION_DOF_ORDER, SERVO_TARGET_RANGE_RAD
from home_pose import (
    HOME_POSE,
    HOME_TAG,
    HOME_TRUNK_PITCH_RAD,
    hand_target_fk_contract,
    home_pose_stamp_matches,
)

POLICY_CONTRACT = "microban-policy-1"
# The training recipe each kind must come from.  A recipe id changes when the
# reward or the target meaning changes (one reviewed line here); fixing a
# failed run inside the same recipe family does not change it.
RECIPES: Mapping[str, str] = {
    "walk": "microban-walk-twist-ratio-1",
    "getup": "microban-getup-single-run-1",
    "pico": "microban-pico-pose-release-twist-ratio-1",
}
KINDS = tuple(RECIPES)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPOSITORY_ROOT / "src" / "agents"
MANIFEST_NAME = "manifest.json"
POLICY_FILES: Mapping[str, str] = {
    "walk": "walk.onnx",
    "getup": "getup.onnx",
    "pico": "pico_teleop.onnx",
}

# A dry run of the training pipeline (a few updates, gates forced) marks its
# package; only the pipeline's own validator and test calls accept it.
DRY_RUN_METADATA_KEY = "dry_run_not_deployable"
DRY_RUN_POLICY_ALLOW_ENV = "MICROBAN_ALLOW_DRYRUN_POLICY"

ACTION_WIDTH = len(OBSERVATION_DOF_ORDER)
HEAD_JOINTS = ("head", "neck_roll", "neck_pitch")
OBSERVATION_SCHEMAS: Mapping[str, tuple[tuple[str, int], ...]] = {
    "walk": (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", ACTION_WIDTH),
        ("joint_vel", ACTION_WIDTH),
        ("actions", ACTION_WIDTH),
        ("command", 3),
    ),
    "getup": (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", ACTION_WIDTH),
        ("joint_vel", ACTION_WIDTH),
        ("actions", ACTION_WIDTH),
    ),
    "pico": (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("joint_pos", 21),
        ("joint_vel", 21),
        ("actions", ACTION_WIDTH),
        ("command", 3),
        ("foot_target", 6),
        ("hand_target", 8),
    ),
}
OBSERVATION_JOINT_NAMES: Mapping[str, tuple[str, ...]] = {
    "walk": tuple(OBSERVATION_DOF_ORDER),
    "getup": tuple(OBSERVATION_DOF_ORDER),
    "pico": (*HEAD_JOINTS, *OBSERVATION_DOF_ORDER),
}
PREVIOUS_ACTION_SEMANTICS = "raw_policy_output"
BASE_ANG_VEL_FRAME = "imu_sensor_xyz"
CONTROL_HZ = 50.0
# default_joint_pos and the +-pi clip are written at full precision.
DEFAULT_POSE_TOLERANCE_RAD = 1.0e-6
CLIP_TOLERANCE_RAD = 1.0e-6

# Midpoint-centered 0.9 soft limits of the deployed MJCF joint ranges.  The
# self-test checks recorded observations against the full ranges (these
# divided back by 0.9) and every HOME must lie inside them.
SOFT_JOINT_POS_LIMITS: Mapping[str, tuple[float, float]] = {
    "right_shoulder_pitch": (-2.82743338823, 2.82743338823),
    "right_shoulder_roll": (-2.98451302091, -0.157079632679),
    "right_elbow": (-1.96349540849, 1.96349540849),
    "right_hip_yaw": (-3.92699081699, 0.785398163397),
    "right_hip_roll": (-0.3926988, 0.3926988),
    "right_hip_pitch": (-1.41371669412, 1.41371669412),
    "right_knee": (-0.628318530718, 2.19911485751),
    "right_ankle_pitch": (-1.46171324855, 0.501782159948),
    "right_ankle_roll": (-0.549778714378, 0.549778714378),
    "left_shoulder_pitch": (-2.82743338823, 2.82743338823),
    "left_shoulder_roll": (0.157079632679, 2.98451302091),
    "left_elbow": (-1.96349540849, 1.96349540849),
    "left_hip_yaw": (-0.785398163397, 3.92699081699),
    "left_hip_roll": (-0.3926988, 0.3926988),
    "left_hip_pitch": (-1.41371669412, 1.41371669412),
    "left_knee": (-0.628318530718, 2.19911485751),
    "left_ankle_pitch": (-1.46171324855, 0.501782159948),
    "left_ankle_roll": (-0.549778714378, 0.549778714378),
}

# Startup self-test: rows recorded from evaluation rollouts of the exported
# checkpoint, with the training actor's deterministic output for each.
SELF_TEST_MIN_ROWS = 8
SELF_TEST_MAX_ROWS = 64
SELF_TEST_ATOL = 1.0e-4
SELF_TEST_RTOL = 1.0e-5
SELF_TEST_GRAVITY_NORM_TOLERANCE = 0.05
# XC330 no-load speed at a full 3S pack (12.6 V / kt 1.0425 V*s/rad).
SELF_TEST_MAX_JOINT_SPEED_RAD_S = 12.1
SELF_TEST_JOINT_RANGE_MARGIN_RAD = math.radians(5.0)

# PICO targets: the command support of the training task.  Inference clips
# the live targets to these, never to model-provided values.
PICO_FOOT_TARGET_LOWER = (-0.03, -0.03, 0.0) * 2
PICO_FOOT_TARGET_UPPER = (0.03, 0.03, 0.05) * 2
PICO_BOTH_FEET_TARGET_LOWER = (-0.01, -0.01, 0.0) * 2
PICO_BOTH_FEET_TARGET_UPPER = (0.01, 0.01, 0.02) * 2
PICO_HAND_TARGET_LOWER = (-0.08,) * 6
PICO_HAND_TARGET_UPPER = (0.08,) * 6
# Observation columns the PICO adapter learns on top of the frozen walker
# (head/neck joint_pos and joint_vel, foot and hand targets); all of them must
# have been trainable when the deployed checkpoint was saved.
PICO_ADAPTER_COLUMNS = (6, 7, 8, 27, 28, 29, *range(69, 83))
# Frame of the PICO hand/foot target columns: the trunk frame with HOME's
# forward lean rotated out (gravity-levelled at HOME), which is what the PICO
# bridge sends.  With a vertical trunk it is the trunk frame itself.
PICO_TARGET_FRAME = (
    "robot_trunk_xyz_forward_left_up"
    if HOME_TRUNK_PITCH_RAD == 0.0
    else "robot_home_levelled_trunk_xyz_forward_left_up"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_RE = re.compile(r"model_(0|[1-9][0-9]*)\.pt\Z")


class PolicyContractError(ValueError):
    """An ONNX policy or the manifest does not describe this robot's contract."""


class PolicySelfTestError(RuntimeError):
    """ONNX Runtime did not reproduce the recorded training outputs."""


@dataclass(frozen=True)
class PicoTargets:
    """The PICO-only part of the contract."""

    walk_checkpoint_sha256: str
    foot_lower: tuple[float, ...]
    foot_upper: tuple[float, ...]
    both_feet_lower: tuple[float, ...]
    both_feet_upper: tuple[float, ...]
    hand_lower: tuple[float, ...]
    hand_upper: tuple[float, ...]
    raw_action_guard: tuple[float, ...]
    curriculum: Mapping[str, int]


@dataclass(frozen=True)
class PolicyContract:
    kind: str
    recipe: str
    input_name: str
    input_width: int
    observation_joint_names: tuple[str, ...]
    checkpoint_filename: str
    checkpoint_iteration: int
    checkpoint_sha256: str
    gate_report_sha256: str
    dry_run: bool
    self_test_observations: np.ndarray  # (N, 1, input_width) float32
    self_test_actions: np.ndarray  # (N, 18) float64
    pico: PicoTargets | None = None


@dataclass(frozen=True)
class LoadedPolicy:
    session: Any
    contract: PolicyContract
    self_test_rows: int


# ---------------------------------------------------------------- metadata
def _exact(metadata: Mapping[str, str], key: str, expected: str) -> str:
    actual = metadata.get(key)
    if actual != expected:
        raise PolicyContractError(f"metadata {key}={actual!r}, expected {expected!r}")
    return actual


def _json(metadata: Mapping[str, str], key: str) -> Any:
    value = metadata.get(key)
    if value is None:
        raise PolicyContractError(f"metadata lacks {key}")

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        raise PolicyContractError(f"metadata {key} is not strict JSON") from exc


def _numbers(value: Any, key: str, count: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise PolicyContractError(f"{key} must be a list of {count} numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise PolicyContractError(f"{key} must contain only numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise PolicyContractError(f"{key} contains a non-finite value")
    return result


def _csv_names(metadata: Mapping[str, str], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not value:
        raise PolicyContractError(f"metadata lacks {key}")
    return tuple(item.strip() for item in value.split(","))


def _csv_floats(metadata: Mapping[str, str], key: str) -> tuple[float, ...]:
    names = _csv_names(metadata, key)
    try:
        values = tuple(float(item) for item in names)
    except ValueError as exc:
        raise PolicyContractError(f"metadata {key} is not numeric") from exc
    if not all(math.isfinite(item) for item in values):
        raise PolicyContractError(f"metadata {key} is not finite")
    return values


def _sha256_value(metadata: Mapping[str, str], key: str) -> str:
    value = metadata.get(key, "")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PolicyContractError(f"metadata {key} must be a lowercase SHA-256")
    return value


def _tensor_width(node: Any, name: str, tensor: str) -> int:
    if getattr(node, "name", None) != name:
        raise PolicyContractError(f"ONNX {tensor} must be named {name!r}")
    node_type = getattr(node, "type", "tensor(float)")
    if node_type != "tensor(float)":
        raise PolicyContractError(f"ONNX {tensor} must be float32")
    shape = list(getattr(node, "shape", ()))
    if len(shape) != 2 or shape[0] not in (1, "1") or not isinstance(shape[1], int):
        raise PolicyContractError(f"ONNX {tensor} must have the fixed shape [1, N]")
    return int(shape[1])


def dry_run_allowed() -> bool:
    return os.environ.get(DRY_RUN_POLICY_ALLOW_ENV) == "1"


def parse_policy(kind: str, session: Any) -> PolicyContract:
    """Validate one ONNX session's tensors and metadata against the contract."""

    if kind not in RECIPES:
        raise PolicyContractError(f"unknown policy kind {kind!r}")
    try:
        metadata = dict(session.get_modelmeta().custom_metadata_map)
    except Exception as exc:  # noqa: BLE001 - ORT/fake session boundary
        raise PolicyContractError("cannot read the ONNX metadata") from exc

    dry_run = DRY_RUN_METADATA_KEY in metadata
    if dry_run and (metadata[DRY_RUN_METADATA_KEY] != "true" or not dry_run_allowed()):
        raise PolicyContractError(
            "policy is a DRY RUN package (dry_run_not_deployable); it is not deployable"
        )
    _exact(metadata, "microban_policy_contract", POLICY_CONTRACT)
    _exact(metadata, "microban_policy_kind", kind)
    _exact(metadata, "microban_recipe", RECIPES[kind])

    # HOME: the full-precision stamp and the joint defaults are this robot's.
    if not home_pose_stamp_matches(metadata.get("home_pose")):
        raise PolicyContractError(
            "home_pose stamp is missing or differs from config/home_pose.yaml "
            "(trained at another HOME?)"
        )
    joint_names = _csv_names(metadata, "joint_names")
    defaults = _csv_floats(metadata, "default_joint_pos")
    if (
        len(joint_names) != len(NEUTRAL_POSE)
        or set(joint_names) != set(NEUTRAL_POSE)
        or len(defaults) != len(joint_names)
    ):
        raise PolicyContractError("joint_names/default_joint_pos must name the 21 joints once")
    for name, value in zip(joint_names, defaults, strict=True):
        if abs(value - NEUTRAL_POSE[name]) > DEFAULT_POSE_TOLERANCE_RAD:
            raise PolicyContractError(
                f"default_joint_pos[{name}]={value!r} differs from the HOME "
                f"{NEUTRAL_POSE[name]!r}"
            )

    # Action: target = clip(HOME + raw * 1.0, -pi, +pi) on the 18 body joints,
    # and the observation feeds back the raw output.
    if _csv_names(metadata, "action_joint_names") != tuple(OBSERVATION_DOF_ORDER):
        raise PolicyContractError("action_joint_names differ from OBSERVATION_DOF_ORDER")
    scale = _csv_floats(metadata, "action_scale")
    if len(scale) not in (1, ACTION_WIDTH) or any(abs(value - 1.0) > 1.0e-6 for value in scale):
        raise PolicyContractError("action_scale must be 1.0")
    lower = _csv_floats(metadata, "action_clip_lower")
    upper = _csv_floats(metadata, "action_clip_upper")
    if (
        len(lower) != ACTION_WIDTH
        or len(upper) != ACTION_WIDTH
        or any(abs(value + SERVO_TARGET_RANGE_RAD) > CLIP_TOLERANCE_RAD for value in lower)
        or any(abs(value - SERVO_TARGET_RANGE_RAD) > CLIP_TOLERANCE_RAD for value in upper)
    ):
        raise PolicyContractError(
            f"action_clip_lower/upper must be the servo range +-{SERVO_TARGET_RANGE_RAD!r}"
        )
    _exact(metadata, "previous_action_semantics", PREVIOUS_ACTION_SEMANTICS)
    _exact(metadata, "base_ang_vel_frame", BASE_ANG_VEL_FRAME)
    try:
        control_hz = float(metadata.get("control_hz", "nan"))
    except ValueError as exc:
        raise PolicyContractError("control_hz must be numeric") from exc
    if control_hz != CONTROL_HZ:
        raise PolicyContractError(f"control_hz must be {CONTROL_HZ:g}")

    # Observation layout.
    schema = OBSERVATION_SCHEMAS[kind]
    if _json(metadata, "observation_schema_json") != [list(term) for term in schema]:
        raise PolicyContractError(f"observation_schema_json is not the {kind} layout")
    observation_joints = _csv_names(metadata, "observation_joint_names")
    if observation_joints != OBSERVATION_JOINT_NAMES[kind]:
        raise PolicyContractError("observation_joint_names drifted")
    width = sum(term_width for _, term_width in schema)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise PolicyContractError("policy must have exactly one input and one output")
    if _tensor_width(inputs[0], "obs", "input") != width:
        raise PolicyContractError(f"{kind} policy input width must be {width}")
    if _tensor_width(outputs[0], "actions", "output") != ACTION_WIDTH:
        raise PolicyContractError(f"policy output width must be {ACTION_WIDTH}")

    # Checkpoint and its passed gate.
    filename = metadata.get("checkpoint_filename", "")
    match = _CHECKPOINT_RE.fullmatch(filename)
    if match is None or metadata.get("checkpoint_iteration") != match.group(1):
        raise PolicyContractError("checkpoint_filename/checkpoint_iteration are not model_N.pt / N")
    checkpoint_sha256 = _sha256_value(metadata, "checkpoint_sha256")
    _exact(metadata, "gate_status", "pass")
    gate_report_sha256 = _sha256_value(metadata, "gate_report_sha256")

    observations, actions = _parse_self_test(kind, metadata, width, observation_joints)
    pico = _parse_pico(metadata, int(match.group(1))) if kind == "pico" else None
    return PolicyContract(
        kind=kind,
        recipe=RECIPES[kind],
        input_name=inputs[0].name,
        input_width=width,
        observation_joint_names=observation_joints,
        checkpoint_filename=filename,
        checkpoint_iteration=int(match.group(1)),
        checkpoint_sha256=checkpoint_sha256,
        gate_report_sha256=gate_report_sha256,
        dry_run=dry_run,
        self_test_observations=observations,
        self_test_actions=actions,
        pico=pico,
    )


def _term_offsets(kind: str) -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for name, width in OBSERVATION_SCHEMAS[kind]:
        offsets[name] = (start, start + width)
        start += width
    return offsets


def _parse_self_test(
    kind: str,
    metadata: Mapping[str, str],
    width: int,
    observation_joints: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows = _json(metadata, "self_test_observations_json")
    expected = _json(metadata, "self_test_actions_json")
    if (
        not isinstance(rows, list)
        or not isinstance(expected, list)
        or not SELF_TEST_MIN_ROWS <= len(rows) <= SELF_TEST_MAX_ROWS
        or len(expected) != len(rows)
    ):
        raise PolicyContractError(
            f"self-test must record {SELF_TEST_MIN_ROWS}..{SELF_TEST_MAX_ROWS} "
            "observations and one action row for each"
        )
    observations = np.asarray(
        [_numbers(row, "self_test_observations_json", width) for row in rows],
        dtype=np.float64,
    )
    actions = np.asarray(
        [_numbers(row, "self_test_actions_json", ACTION_WIDTH) for row in expected],
        dtype=np.float64,
    )
    # Each row must be a physically possible robot state, not a synthetic one.
    offsets = _term_offsets(kind)
    gravity = observations[:, slice(*offsets["projected_gravity"])]
    positions = observations[:, slice(*offsets["joint_pos"])]
    velocities = observations[:, slice(*offsets["joint_vel"])]
    for index, name in enumerate(observation_joints):
        if name not in SOFT_JOINT_POS_LIMITS:
            continue
        soft_lower, soft_upper = SOFT_JOINT_POS_LIMITS[name]
        middle = 0.5 * (soft_lower + soft_upper)
        half_range = 0.5 * (soft_upper - soft_lower) / 0.9
        angles = positions[:, index] + NEUTRAL_POSE[name]
        if np.any(np.abs(angles - middle) > half_range + SELF_TEST_JOINT_RANGE_MARGIN_RAD):
            raise PolicyContractError(f"self-test observation puts {name} outside its range")
    if np.any(
        np.abs(np.linalg.norm(gravity, axis=1) - 1.0) > SELF_TEST_GRAVITY_NORM_TOLERANCE
    ) or np.any(np.abs(velocities) > SELF_TEST_MAX_JOINT_SPEED_RAD_S):
        raise PolicyContractError("self-test observations are not physically possible")
    return observations.astype(np.float32).reshape(-1, 1, width), actions


def _parse_pico(metadata: Mapping[str, str], checkpoint_iteration: int) -> PicoTargets:
    _exact(metadata, "pico_target_frame", PICO_TARGET_FRAME)
    if not _same_json(_json(metadata, "pico_hand_target_fk_json"), hand_target_fk_contract()):
        raise PolicyContractError(
            "pico_hand_target_fk_json differs from config/home_pose.yaml hand_target_fk"
        )
    bounds = {}
    for key, expected in (
        ("pico_foot_target_lower_json", PICO_FOOT_TARGET_LOWER),
        ("pico_foot_target_upper_json", PICO_FOOT_TARGET_UPPER),
        ("pico_both_feet_target_lower_json", PICO_BOTH_FEET_TARGET_LOWER),
        ("pico_both_feet_target_upper_json", PICO_BOTH_FEET_TARGET_UPPER),
        ("pico_hand_target_lower_json", PICO_HAND_TARGET_LOWER),
        ("pico_hand_target_upper_json", PICO_HAND_TARGET_UPPER),
    ):
        values = _numbers(_json(metadata, key), key, 6)
        if not np.allclose(values, expected, rtol=0.0, atol=1.0e-9):
            raise PolicyContractError(f"{key} differs from the trained command support")
        bounds[key] = tuple(float(value) for value in expected)
    guard = _numbers(_json(metadata, "pico_raw_action_guard_json"), "pico_raw_action_guard_json", ACTION_WIDTH)
    with np.errstate(over="ignore"):
        guard_float32 = np.asarray(guard, dtype=np.float32)
    if any(value <= 0.0 for value in guard) or not np.isfinite(guard_float32).all():
        raise PolicyContractError("pico_raw_action_guard_json must be positive finite float32 values")
    curriculum = _json(metadata, "pico_curriculum_json")
    keys = ("critic_warmup", "hand_start", "hand_tighten", "foot_start", "foot_tighten", "total")
    if (
        not isinstance(curriculum, dict)
        or set(curriculum) != set(keys)
        or any(isinstance(curriculum[key], bool) or not isinstance(curriculum[key], int) for key in keys)
        or not 0 < curriculum["hand_start"] <= curriculum["foot_start"]
        <= curriculum["foot_tighten"] <= curriculum["total"]
    ):
        raise PolicyContractError("pico_curriculum_json is malformed")
    if not curriculum["foot_tighten"] <= checkpoint_iteration + 1 <= curriculum["total"]:
        raise PolicyContractError("PICO checkpoint is not from the final curriculum stage")
    if _json(metadata, "pico_active_adapter_columns_json") != list(PICO_ADAPTER_COLUMNS):
        raise PolicyContractError("PICO checkpoint did not train every adapter column")
    return PicoTargets(
        walk_checkpoint_sha256=_sha256_value(metadata, "pico_walk_checkpoint_sha256"),
        foot_lower=bounds["pico_foot_target_lower_json"],
        foot_upper=bounds["pico_foot_target_upper_json"],
        both_feet_lower=bounds["pico_both_feet_target_lower_json"],
        both_feet_upper=bounds["pico_both_feet_target_upper_json"],
        hand_lower=bounds["pico_hand_target_lower_json"],
        hand_upper=bounds["pico_hand_target_upper_json"],
        raw_action_guard=guard,
        curriculum=dict(curriculum),
    )


def _same_json(actual: Any, expected: Any) -> bool:
    """JSON equality without Python's bool/int/float coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_json(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_json(a, e) for a, e in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


# ---------------------------------------------------------------- self-test
def run_self_test(session: Any, contract: PolicyContract) -> int:
    """Run the recorded observations through ONNX Runtime; return the row count."""

    guard = (
        np.asarray(contract.pico.raw_action_guard, dtype=np.float64)
        if contract.pico is not None
        else None
    )
    for index, (observation, expected) in enumerate(
        zip(contract.self_test_observations, contract.self_test_actions, strict=True)
    ):
        try:
            outputs = session.run(None, {contract.input_name: observation})
            output = np.asarray(outputs[0], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - ORT boundary
            raise PolicySelfTestError(f"{contract.kind} self-test failed to run row {index}") from exc
        if len(outputs) != 1 or output.shape != (1, ACTION_WIDTH) or not np.isfinite(output).all():
            raise PolicySelfTestError(f"{contract.kind} self-test row {index} returned an unsafe output")
        error = float(np.max(np.abs(output[0] - expected)))
        bound = SELF_TEST_ATOL + SELF_TEST_RTOL * float(np.max(np.abs(expected)))
        if error > bound:
            raise PolicySelfTestError(
                f"{contract.kind} self-test row {index}: ONNX Runtime differs from the "
                f"recorded training output by {error:.3g} (limit {bound:.3g})"
            )
        if guard is not None and np.any(np.abs(output[0]) > guard):
            raise PolicySelfTestError(
                f"pico self-test row {index} exceeds the raw-action guard"
            )
    return len(contract.self_test_observations)


# ---------------------------------------------------------------- manifest
def load_manifest(agents_dir: Path | str = AGENTS_DIR) -> dict[str, Any]:
    """Read and check src/agents/manifest.json (the installer writes it)."""

    path = Path(agents_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyContractError(f"cannot read the policy manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PolicyContractError(f"{path} must hold a JSON object")
    if manifest.get("contract") != POLICY_CONTRACT:
        raise PolicyContractError(f"{path}: contract must be {POLICY_CONTRACT!r}")
    if manifest.get("home_tag") != HOME_TAG:
        raise PolicyContractError(
            f"{path}: home_tag {manifest.get('home_tag')!r} is not this robot's HOME "
            f"{HOME_TAG!r} (config/home_pose.yaml)"
        )
    if not isinstance(manifest.get("dry_run"), bool):
        raise PolicyContractError(f"{path}: dry_run must be true or false")
    if manifest["dry_run"] and not dry_run_allowed():
        raise PolicyContractError(f"{path} describes a DRY RUN package; it is not deployable")
    policies = manifest.get("policies")
    if not isinstance(policies, dict) or set(policies) != set(KINDS):
        raise PolicyContractError(f"{path}: policies must describe {list(KINDS)}")
    for kind, entry in policies.items():
        if (
            not isinstance(entry, dict)
            or entry.get("file") != POLICY_FILES[kind]
            or not isinstance(entry.get("sha256"), str)
            or _SHA256_RE.fullmatch(entry["sha256"]) is None
            or not isinstance(entry.get("checkpoint_sha256"), str)
            or _SHA256_RE.fullmatch(entry["checkpoint_sha256"]) is None
        ):
            raise PolicyContractError(f"{path}: policies.{kind} is malformed")
    return manifest


def check_manifest_file(kind: str, policy_path: Path, manifest: Mapping[str, Any], file_sha256: str) -> None:
    """The file is the one the manifest installed for ``kind``."""

    entry = manifest["policies"][kind]
    if Path(policy_path).name != entry["file"]:
        raise PolicyContractError(f"{policy_path} is not the manifest's {entry['file']}")
    if file_sha256 != entry["sha256"]:
        raise PolicyContractError(
            f"{policy_path} is not the file the manifest installed (SHA-256 differs)"
        )


def check_manifest(
    contract: PolicyContract,
    policy_path: Path,
    manifest: Mapping[str, Any],
    file_sha256: str,
) -> None:
    """Bind one installed policy to the release the manifest describes."""

    check_manifest_file(contract.kind, policy_path, manifest, file_sha256)
    entry = manifest["policies"][contract.kind]
    if contract.checkpoint_sha256 != entry["checkpoint_sha256"]:
        raise PolicyContractError(f"{policy_path} checkpoint SHA-256 differs from the manifest")
    if contract.dry_run != manifest["dry_run"]:
        raise PolicyContractError("the manifest and the ONNX disagree about the dry run")
    if (
        contract.pico is not None
        and contract.pico.walk_checkpoint_sha256
        != manifest["policies"]["walk"]["checkpoint_sha256"]
    ):
        raise PolicyContractError(
            "the PICO policy was not trained on the installed walking checkpoint"
        )


def open_session(model: Path | str | bytes, providers: Sequence[str] | None = None) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    source = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(source, sess_options=options, providers=providers)


def load_installed_policy(
    kind: str,
    policy_path: Path | str | None = None,
    *,
    providers: Sequence[str] | None = None,
) -> LoadedPolicy:
    """Open an installed policy, check it and its manifest, run its self-test."""

    path = Path(policy_path) if policy_path is not None else AGENTS_DIR / POLICY_FILES[kind]
    data = path.read_bytes()
    file_sha256 = hashlib.sha256(data).hexdigest()
    manifest = load_manifest(path.parent)
    check_manifest_file(kind, path, manifest, file_sha256)
    # Hash and load the same bytes: a file replaced meanwhile cannot slip in.
    session = open_session(data, providers)
    contract = parse_policy(kind, session)
    check_manifest(contract, path, manifest, file_sha256)
    rows = run_self_test(session, contract)
    return LoadedPolicy(session=session, contract=contract, self_test_rows=rows)


def home_within_soft_limits() -> bool:
    """Every body joint's HOME angle lies inside its soft limits."""

    return all(
        lower <= NEUTRAL_POSE[name] <= upper
        for name, (lower, upper) in SOFT_JOINT_POS_LIMITS.items()
    )


if not home_within_soft_limits():
    raise RuntimeError(f"HOME {HOME_POSE['path']} puts a body joint outside its soft limits")
