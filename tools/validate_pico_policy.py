#!/usr/bin/env python3
"""Validate a PICO hybrid ONNX without opening the motor or network interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
