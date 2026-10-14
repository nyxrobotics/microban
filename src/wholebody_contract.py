# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Contract ``microban-wholebody-1``: the whole-body tracking policy.

A contract of its own beside ``microban-policy-1`` (walk, get-up, PICO; see
policy_contract.py), written by mjlab_microban
``src/mjlab_microban/wholebody/contract.py`` and
``scripts/wholebody_export.py``.  It reuses that contract's parts: the HOME
stamp, the SHA-256 records and the startup self-test (recorded observation
histories with the training actor's deterministic output, re-run here in ONNX
Runtime within the same bounds).

Input ``obs_history`` [1, 10, 118]: the last 10 frames, oldest first, the
current one last; after a reset every row is the first frame.  A frame is the
12 terms of OBSERVATION_SCHEMA (wholebody/observation.py builds it).
Output ``actions`` [1, 18]: target = clip(HOME + raw, -pi, +pi) on
OBSERVATION_DOF_ORDER, arms included; the raw output recurs as the previous
action.  The neck is not the policy's: its targets are the reference's neck
angles clamped to ``neck_limits_rad_json`` and slewed at
``neck_slew_rates_rad_s_json`` (the training data's neck is HOME).

The installed file is ``src/agents/wholebody.onnx`` with its own manifest
``src/agents/wholebody_manifest.json`` (tools/install_wholebody_policy.py
writes both); the walk/get-up/PICO manifest is not touched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from constants import NEUTRAL_POSE, OBSERVATION_DOF_ORDER, SERVO_TARGET_RANGE_RAD
from home_pose import HOME_TAG, home_pose_stamp_matches
from policy_contract import (
    AGENTS_DIR,
    BASE_ANG_VEL_FRAME,
    CLIP_TOLERANCE_RAD,
    CONTROL_HZ,
    DEFAULT_POSE_TOLERANCE_RAD,
    DRY_RUN_METADATA_KEY,
    HEAD_JOINTS,
    PREVIOUS_ACTION_SEMANTICS,
    SELF_TEST_ATOL,
    SELF_TEST_GRAVITY_NORM_TOLERANCE,
    SELF_TEST_JOINT_RANGE_MARGIN_RAD,
    SELF_TEST_MAX_JOINT_SPEED_RAD_S,
    SELF_TEST_MAX_ROWS,
    SELF_TEST_MIN_ROWS,
    SELF_TEST_RTOL,
    SOFT_JOINT_POS_LIMITS,
    PolicyContractError,
    PolicySelfTestError,
    _csv_floats,
    _csv_names,
    _exact,
    _json,
    _numbers,
    _sha256_value,
    _CHECKPOINT_RE,
    dry_run_allowed,
    open_session,
)

CONTRACT = "microban-wholebody-1"
KIND = "wholebody"
RECIPE = "microban-wholebody-teleopit-track-1"
POLICY_FILE = "wholebody.onnx"
MANIFEST_FILE = "wholebody_manifest.json"
INPUT_NAME = "obs_history"
OUTPUT_NAME = "actions"
HISTORY_LENGTH = 10
ACTION_WIDTH = len(OBSERVATION_DOF_ORDER)
OBSERVATION_JOINT_NAMES = (*HEAD_JOINTS, *OBSERVATION_DOF_ORDER)
REFERENCE_JOINT_NAMES = tuple(OBSERVATION_DOF_ORDER)
OBSERVATION_SCHEMA: tuple[tuple[str, int], ...] = (
    ("ref_joint_pos", 18),
    ("ref_joint_vel", 18),
    ("ref_anchor_ori_b", 6),
    ("base_ang_vel", 3),
    ("joint_pos", 21),
    ("joint_vel", 21),
    ("actions", 18),
    ("projected_gravity", 3),
    ("ref_anchor_lin_vel_b", 3),
    ("ref_anchor_ang_vel_b", 3),
    ("ref_projected_gravity_b", 3),
    ("ref_anchor_height", 1),
)
OBSERVATION_WIDTH = sum(width for _, width in OBSERVATION_SCHEMA)
# The meaning of the timing and frame fields; any other value is another contract.
FIXED_FIELDS = {
    "input_names": INPUT_NAME,
    "output_name": OUTPUT_NAME,
    "history_length": str(HISTORY_LENGTH),
    "history_order": "oldest_first_current_last",
    "history_after_reset": "every_row_the_first_frame",
    "reference_time": "one_control_period_ahead",
    "reference_yaw": "aligned_to_imu_yaw_at_start",
    "neck_joint_names": ",".join(HEAD_JOINTS),
    "neck_control": "reference_clamped_slewed",
    "previous_action_semantics": PREVIOUS_ACTION_SEMANTICS,
    "base_ang_vel_frame": BASE_ANG_VEL_FRAME,
}


def term_slices() -> dict[str, slice]:
    out, start = {}, 0
    for name, width in OBSERVATION_SCHEMA:
        out[name] = slice(start, start + width)
        start += width
    return out


@dataclass(frozen=True)
class WholebodyContract:
    recipe: str
    checkpoint_filename: str
    checkpoint_iteration: int
    checkpoint_sha256: str
    benchmark_report_sha256: str
    eval_only_benchmark_report_sha256: str
    dry_run: bool
    neck_limits_rad: Mapping[str, tuple[float, float]]
    neck_slew_rates_rad_s: Mapping[str, float]
    self_test_histories: np.ndarray  # (N, 1, 10, 118) float32
    self_test_actions: np.ndarray  # (N, 18) float64


@dataclass(frozen=True)
class LoadedWholebodyPolicy:
    session: Any
    contract: WholebodyContract
    self_test_rows: int


def frame_problem(frame: np.ndarray) -> str | None:
    """Why one observation frame is not a possible robot state, or None."""

    terms = term_slices()
    if frame.shape != (OBSERVATION_WIDTH,) or not np.isfinite(frame).all():
        return "not a finite frame of the schema's width"
    for name in ("projected_gravity", "ref_projected_gravity_b"):
        if abs(float(np.linalg.norm(frame[terms[name]])) - 1.0) > SELF_TEST_GRAVITY_NORM_TOLERANCE:
            return f"{name} is not a unit vector"
    if np.any(np.abs(frame[terms["joint_vel"]]) > SELF_TEST_MAX_JOINT_SPEED_RAD_S):
        return "joint speed above the servo's no-load speed"
    for name, residual in zip(OBSERVATION_JOINT_NAMES, frame[terms["joint_pos"]], strict=True):
        if name not in SOFT_JOINT_POS_LIMITS:
            continue
        soft_lower, soft_upper = SOFT_JOINT_POS_LIMITS[name]
        middle = 0.5 * (soft_lower + soft_upper)
        half_range = 0.5 * (soft_upper - soft_lower) / 0.9
        if abs(float(residual) + NEUTRAL_POSE[name] - middle) > half_range + SELF_TEST_JOINT_RANGE_MARGIN_RAD:
            return f"{name} outside its range"
    return None


def _shape(node: Any, name: str, expected: tuple[int, ...]) -> None:
    if getattr(node, "name", None) != name:
        raise PolicyContractError(f"ONNX tensor must be named {name!r}")
    if getattr(node, "type", "tensor(float)") != "tensor(float)":
        raise PolicyContractError(f"ONNX {name} must be float32")
    shape = tuple(int(d) if isinstance(d, int) or (isinstance(d, str) and d.isdigit()) else -1
                  for d in getattr(node, "shape", ()))
    if shape != expected:
        raise PolicyContractError(f"ONNX {name} has shape {shape}, expected {expected}")


def _neck_fields(metadata: Mapping[str, str]) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    limits_json = _json(metadata, "neck_limits_rad_json")
    rates_json = _json(metadata, "neck_slew_rates_rad_s_json")
    if not isinstance(limits_json, dict) or set(limits_json) != set(HEAD_JOINTS):
        raise PolicyContractError("neck_limits_rad_json must give the three neck joints")
    if not isinstance(rates_json, dict) or set(rates_json) != set(HEAD_JOINTS):
        raise PolicyContractError("neck_slew_rates_rad_s_json must give the three neck joints")
    limits, rates = {}, {}
    for name in HEAD_JOINTS:
        lower, upper = _numbers(limits_json[name], f"neck_limits_rad_json.{name}", 2)
        if not lower <= NEUTRAL_POSE[name] <= upper:
            raise PolicyContractError(f"neck limit of {name} excludes HOME")
        (rate,) = _numbers([rates_json[name]], f"neck_slew_rates_rad_s_json.{name}", 1)
        if rate <= 0.0:
            raise PolicyContractError(f"neck slew rate of {name} must be positive")
        limits[name], rates[name] = (lower, upper), rate
    return limits, rates


def parse_wholebody_policy(session: Any) -> WholebodyContract:
    """Validate one ONNX session's tensors and metadata against microban-wholebody-1."""

    try:
        metadata = dict(session.get_modelmeta().custom_metadata_map)
    except Exception as exc:  # noqa: BLE001 - ORT/fake session boundary
        raise PolicyContractError("cannot read the ONNX metadata") from exc

    dry_run = DRY_RUN_METADATA_KEY in metadata
    if dry_run and (metadata[DRY_RUN_METADATA_KEY] != "true" or not dry_run_allowed()):
        raise PolicyContractError(
            "whole-body policy is a DRY RUN package (dry_run_not_deployable); it is not deployable"
        )
    _exact(metadata, "microban_policy_contract", CONTRACT)
    _exact(metadata, "microban_policy_kind", KIND)
    _exact(metadata, "microban_recipe", RECIPE)
    for key, expected in FIXED_FIELDS.items():
        _exact(metadata, key, expected)

    if not home_pose_stamp_matches(metadata.get("home_pose")):
        raise PolicyContractError(
            "home_pose stamp is missing or differs from config/home_pose.yaml (trained at another HOME?)"
        )
    joint_names = _csv_names(metadata, "joint_names")
    defaults = _csv_floats(metadata, "default_joint_pos")
    if len(joint_names) != len(NEUTRAL_POSE) or set(joint_names) != set(NEUTRAL_POSE) or len(defaults) != len(
        joint_names
    ):
        raise PolicyContractError("joint_names/default_joint_pos must name the 21 joints once")
    for name, value in zip(joint_names, defaults, strict=True):
        if abs(value - NEUTRAL_POSE[name]) > DEFAULT_POSE_TOLERANCE_RAD:
            raise PolicyContractError(f"default_joint_pos[{name}]={value!r} differs from the HOME")

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
        raise PolicyContractError(f"action_clip_lower/upper must be +-{SERVO_TARGET_RANGE_RAD!r}")
    try:
        control_hz = float(metadata.get("control_hz", "nan"))
    except ValueError as exc:
        raise PolicyContractError("control_hz must be numeric") from exc
    if control_hz != CONTROL_HZ:
        raise PolicyContractError(f"control_hz must be {CONTROL_HZ:g}")
    if _json(metadata, "observation_schema_json") != [list(term) for term in OBSERVATION_SCHEMA]:
        raise PolicyContractError("observation_schema_json is not the whole-body layout")
    if _csv_names(metadata, "observation_joint_names") != OBSERVATION_JOINT_NAMES:
        raise PolicyContractError("observation_joint_names drifted")
    if _csv_names(metadata, "reference_joint_names") != REFERENCE_JOINT_NAMES:
        raise PolicyContractError("reference_joint_names drifted")
    neck_limits, neck_rates = _neck_fields(metadata)

    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise PolicyContractError("policy must have exactly one input and one output")
    _shape(inputs[0], INPUT_NAME, (1, HISTORY_LENGTH, OBSERVATION_WIDTH))
    _shape(outputs[0], OUTPUT_NAME, (1, ACTION_WIDTH))

    filename = metadata.get("checkpoint_filename", "")
    match = _CHECKPOINT_RE.fullmatch(filename)
    if match is None or metadata.get("checkpoint_iteration") != match.group(1):
        raise PolicyContractError("checkpoint_filename/checkpoint_iteration are not model_N.pt / N")

    histories = _json(metadata, "self_test_histories_json")
    expected = _json(metadata, "self_test_actions_json")
    if (
        not isinstance(histories, list)
        or not isinstance(expected, list)
        or not SELF_TEST_MIN_ROWS <= len(histories) <= SELF_TEST_MAX_ROWS
        or len(expected) != len(histories)
    ):
        raise PolicyContractError(
            f"self-test must record {SELF_TEST_MIN_ROWS}..{SELF_TEST_MAX_ROWS} histories and one action each"
        )
    rows = []
    for index, history in enumerate(histories):
        if not isinstance(history, list) or len(history) != HISTORY_LENGTH:
            raise PolicyContractError(f"self-test history {index} is not {HISTORY_LENGTH} frames")
        frames = np.asarray(
            [_numbers(frame, "self_test_histories_json", OBSERVATION_WIDTH) for frame in history], dtype=np.float64
        )
        for frame in frames:
            problem = frame_problem(frame)
            if problem:
                raise PolicyContractError(f"self-test history {index}: {problem}")
        rows.append(frames)
    actions = np.asarray([_numbers(row, "self_test_actions_json", ACTION_WIDTH) for row in expected], dtype=np.float64)
    return WholebodyContract(
        recipe=RECIPE,
        checkpoint_filename=filename,
        checkpoint_iteration=int(match.group(1)),
        checkpoint_sha256=_sha256_value(metadata, "checkpoint_sha256"),
        benchmark_report_sha256=_sha256_value(metadata, "benchmark_report_sha256"),
        eval_only_benchmark_report_sha256=_sha256_value(metadata, "eval_only_benchmark_report_sha256"),
        dry_run=dry_run,
        neck_limits_rad=neck_limits,
        neck_slew_rates_rad_s=neck_rates,
        self_test_histories=np.asarray(rows, dtype=np.float32).reshape(-1, 1, HISTORY_LENGTH, OBSERVATION_WIDTH),
        self_test_actions=actions,
    )


def run_self_test(session: Any, contract: WholebodyContract) -> int:
    """Run the recorded histories through ONNX Runtime; return the row count."""

    for index, (history, expected) in enumerate(
        zip(contract.self_test_histories, contract.self_test_actions, strict=True)
    ):
        try:
            outputs = session.run(None, {INPUT_NAME: history})
            output = np.asarray(outputs[0], dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - ORT boundary
            raise PolicySelfTestError(f"wholebody self-test failed to run row {index}") from exc
        if len(outputs) != 1 or output.shape != (1, ACTION_WIDTH) or not np.isfinite(output).all():
            raise PolicySelfTestError(f"wholebody self-test row {index} returned an unsafe output")
        error = float(np.max(np.abs(output[0] - expected)))
        bound = SELF_TEST_ATOL + SELF_TEST_RTOL * float(np.max(np.abs(expected)))
        if error > bound:
            raise PolicySelfTestError(
                f"wholebody self-test row {index}: ONNX Runtime differs from the recorded training "
                f"output by {error:.3g} (limit {bound:.3g})"
            )
    return len(contract.self_test_histories)


def load_manifest(agents_dir: Path | str = AGENTS_DIR) -> dict[str, Any]:
    path = Path(agents_dir) / MANIFEST_FILE
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicyContractError(f"cannot read the whole-body manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("contract") != CONTRACT:
        raise PolicyContractError(f"{path}: contract must be {CONTRACT!r}")
    if manifest.get("home_tag") != HOME_TAG:
        raise PolicyContractError(f"{path}: home_tag {manifest.get('home_tag')!r} is not this robot's HOME {HOME_TAG!r}")
    if not isinstance(manifest.get("dry_run"), bool):
        raise PolicyContractError(f"{path}: dry_run must be true or false")
    if manifest["dry_run"] and not dry_run_allowed():
        raise PolicyContractError(f"{path} describes a DRY RUN package; it is not deployable")
    for key in ("sha256", "checkpoint_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise PolicyContractError(f"{path}: {key} must be a lowercase SHA-256")
    if manifest.get("file") != POLICY_FILE:
        raise PolicyContractError(f"{path}: file must be {POLICY_FILE}")
    return manifest


def load_wholebody_policy(
    policy_path: Path | str | None = None,
    *,
    providers: Sequence[str] | None = None,
) -> LoadedWholebodyPolicy:
    """Open the installed whole-body policy, check it and its manifest, run its self-test."""

    path = Path(policy_path) if policy_path is not None else AGENTS_DIR / POLICY_FILE
    data = path.read_bytes()
    file_sha256 = hashlib.sha256(data).hexdigest()
    manifest = load_manifest(path.parent)
    if path.name != POLICY_FILE or file_sha256 != manifest["sha256"]:
        raise PolicyContractError(f"{path} is not the file the whole-body manifest installed")
    session = open_session(data, providers)
    contract = parse_wholebody_policy(session)
    if contract.checkpoint_sha256 != manifest["checkpoint_sha256"]:
        raise PolicyContractError(f"{path} checkpoint SHA-256 differs from the manifest")
    if contract.dry_run != manifest["dry_run"]:
        raise PolicyContractError("the manifest and the ONNX disagree about the dry run")
    rows = run_self_test(session, contract)
    return LoadedWholebodyPolicy(session=session, contract=contract, self_test_rows=rows)


def check_onnx_file(path: Path | str) -> tuple[bytes, WholebodyContract, int]:
    """An exported file's bytes, contract and self-test rows (the installer's check)."""

    data = Path(path).read_bytes()
    session = open_session(data, ["CPUExecutionProvider"])
    contract = parse_wholebody_policy(session)
    return data, contract, run_self_test(session, contract)
