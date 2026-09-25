#!/usr/bin/env python3
"""Validate the PICO ONNX and pinned walk fallback without opening interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from moves.pico_hybrid import (
    EXPECTED_ONNX_PARITY_GATE_VERSION,
    EXPECTED_V12_RAW_ACTION_GUARD_FORMULA,
    EXPECTED_V12_RAW_ACTION_GUARD_MULTIPLIER,
    EXPECTED_V12_RAW_ACTION_GUARD_SEMANTICS,
    EXPECTED_V12_TRAINING_CONTRACT_VERSION,
    PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS,
    PicoHybridMove,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WALK_FALLBACK_POLICY = REPOSITORY_ROOT / "src" / "agents" / "walk.onnx"
EXPECTED_WALK_FALLBACK_SHA256 = (
    "10c58a63c66337669c3d4c588732d541a6a07eea3291c0401f79893c7f60f15d"
)
WALK_FALLBACK_SMOKE_SAMPLE_COUNT = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_fallback_smoke_inputs() -> np.ndarray:
    """Return a deterministic, exactly representable finite 63-column corpus."""

    integers = np.arange(
        WALK_FALLBACK_SMOKE_SAMPLE_COUNT * 63,
        dtype=np.int32,
    ).reshape(WALK_FALLBACK_SMOKE_SAMPLE_COUNT, 63)
    observations = ((integers % 29) - 14).astype(np.float32) / np.float32(16.0)
    observations[0].fill(0.0)
    if observations.shape != (WALK_FALLBACK_SMOKE_SAMPLE_COUNT, 63) or not bool(
        np.isfinite(observations).all()
    ):
        raise AssertionError("walk fallback smoke corpus is invalid")
    return observations


def validate_walk_fallback(
    policy_path: Path = WALK_FALLBACK_POLICY,
) -> dict[str, object]:
    """Authenticate and CPU-smoke the exact fallback used by ``WalkMove``."""

    policy_path = policy_path.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"walk fallback policy not found: {policy_path}")
    digest = _sha256(policy_path)
    if digest != EXPECTED_WALK_FALLBACK_SHA256:
        raise RuntimeError(
            "walk fallback SHA-256 mismatch: "
            f"expected {EXPECTED_WALK_FALLBACK_SHA256}, got {digest}"
        )

    session = ort.InferenceSession(
        str(policy_path),
        providers=["CPUExecutionProvider"],
    )
    providers = session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise RuntimeError("walk fallback validation requires CPUExecutionProvider only")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("walk fallback must have exactly one input and one output")
    input_value = inputs[0]
    output_value = outputs[0]
    if (
        input_value.name != "obs"
        or input_value.shape != [1, 63]
        or input_value.type != "tensor(float)"
    ):
        raise RuntimeError(
            "walk fallback input must be float32 obs[1,63], got "
            f"{input_value.name!r} {input_value.shape!r} {input_value.type!r}"
        )
    if (
        output_value.name != "actions"
        or output_value.shape != [1, 18]
        or output_value.type != "tensor(float)"
    ):
        raise RuntimeError(
            "walk fallback output must be float32 actions[1,18], got "
            f"{output_value.name!r} {output_value.shape!r} {output_value.type!r}"
        )

    maximum_absolute_output = 0.0
    observations = _walk_fallback_smoke_inputs()
    for observation in observations:
        inference_outputs = session.run(
            None,
            {input_value.name: observation.reshape(1, 63)},
        )
        if len(inference_outputs) != 1:
            raise RuntimeError("walk fallback inference returned multiple outputs")
        action = np.asarray(inference_outputs[0])
        if (
            action.shape != (1, 18)
            or action.dtype != np.float32
            or not bool(np.isfinite(action).all())
        ):
            raise RuntimeError(
                "walk fallback smoke returned an unsafe output: "
                f"shape={action.shape}, dtype={action.dtype}"
            )
        maximum_absolute_output = max(
            maximum_absolute_output,
            float(np.max(np.abs(action))),
        )

    return {
        "status": "pass",
        "policy": str(policy_path),
        "sha256": digest,
        "providers": providers,
        "input": {
            "name": input_value.name,
            "shape": input_value.shape,
            "type": input_value.type,
        },
        "output": {
            "name": output_value.name,
            "shape": output_value.shape,
            "type": output_value.type,
        },
        "smoke": {
            "status": "pass",
            "sample_count": len(observations),
            "corpus": "deterministic_exact_float32_mod29_v1",
            "all_outputs_finite": True,
            "maximum_absolute_output": maximum_absolute_output,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "policy",
        nargs="?",
        type=Path,
        default=Path("src/agents/pico_teleop.onnx"),
    )
    args = parser.parse_args()
    if not args.policy.is_file():
        raise FileNotFoundError(f"PICO policy not found: {args.policy}")

    # This is the robot admission check, not a performance benchmark.  Pin the
    # provider so a workstation with CUDA installed cannot accidentally certify
    # a graph that the robot's CPU runtime has never loaded and executed.
    session = ort.InferenceSession(
        str(args.policy), providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("PICO policy validation requires CPUExecutionProvider only")
    move = PicoHybridMove(
        controller=None,
        policy_path=args.policy,
        # Contract validation does not read a sensor. Keep the hardware-frame
        # transform out of this offline command entirely.
        gyro_transform=lambda values: values,
        session=session,
    )
    contract = move._contract
    runtime_smoke_samples = move._compatibility_smoke_sample_count
    is_v12 = (
        contract.training_contract_version == EXPECTED_V12_TRAINING_CONTRACT_VERSION
    )
    walk_fallback = validate_walk_fallback()
    print(
        json.dumps(
            {
                "status": "pass",
                "policy": str(args.policy.resolve()),
                "input_width": 83,
                "output_width": len(contract.action_joint_names),
                "checkpoint_filename": contract.checkpoint_filename,
                "checkpoint_iteration": contract.checkpoint_iteration,
                "checkpoint_completed_updates": contract.checkpoint_completed_updates,
                "checkpoint_sha256": contract.checkpoint_sha256,
                "training_contract_version": contract.training_contract_version,
                "runtime_action_semantics": contract.runtime_action_semantics,
                "onnx_parity_gate_version": (
                    "v12" if is_v12 else EXPECTED_ONNX_PARITY_GATE_VERSION
                ),
                "onnxruntime_compatibility_smoke": {
                    "status": "pass",
                    "sample_count": runtime_smoke_samples,
                    "providers": move._session.get_providers(),
                    "scope": (
                        "load_run_output_shape_float32_finiteness_and_"
                        "authenticated_finite_amplitude_guard"
                        if is_v12
                        else "load_run_output_shape_finiteness_and_open_actor_bounds"
                    ),
                },
                "observation_joint_names": contract.observation_joint_names,
                "action_joint_names": contract.action_joint_names,
                "enforced_default_joint_pos": contract.action_default_joint_pos,
                "enforced_action_scale": contract.action_scale,
                "enforced_soft_joint_pos_lower": contract.soft_lower,
                "enforced_soft_joint_pos_upper": contract.soft_upper,
                "physical_motor_target_guard_semantics": (
                    PHYSICAL_MOTOR_TARGET_GUARD_SEMANTICS
                ),
                "enforced_actor_raw_action_lower": (
                    None if is_v12 else contract.actor_raw_action_lower
                ),
                "enforced_actor_raw_action_upper": (
                    None if is_v12 else contract.actor_raw_action_upper
                ),
                "v12_legacy_source_checkpoint_sha256": (
                    contract.v12_legacy_source_checkpoint_sha256
                ),
                "v12_legacy_probe_sha256": contract.v12_legacy_probe_sha256,
                "v12_stage_gate_sha256": contract.v12_stage_gate_sha256,
                "v12_tracking_report_sha256": (contract.v12_tracking_report_sha256),
                "v12_raw_action_guard": (
                    {
                        "formula": EXPECTED_V12_RAW_ACTION_GUARD_FORMULA,
                        "multiplier": EXPECTED_V12_RAW_ACTION_GUARD_MULTIPLIER,
                        "semantics": EXPECTED_V12_RAW_ACTION_GUARD_SEMANTICS,
                        "absolute_maximum": (
                            contract.v12_runtime_raw_action_guard_absolute_maximum
                        ),
                    }
                    if is_v12
                    else None
                ),
                "walk_fallback": walk_fallback,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
