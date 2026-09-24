#!/usr/bin/env python3
"""Validate a PICO hybrid ONNX without opening the motor or network interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moves.pico_hybrid import PicoHybridMove


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

    move = PicoHybridMove(
        controller=None,
        policy_path=args.policy,
        # Contract validation does not read a sensor. Keep the hardware-frame
        # transform out of this offline command entirely.
        gyro_transform=lambda values: values,
    )
    contract = move._contract
    print(
        json.dumps(
            {
                "status": "pass",
                "policy": str(args.policy.resolve()),
                "input_width": 83,
                "output_width": len(contract.action_joint_names),
                "observation_joint_names": contract.observation_joint_names,
                "action_joint_names": contract.action_joint_names,
                "enforced_default_joint_pos": contract.action_default_joint_pos,
                "enforced_action_scale": contract.action_scale,
                "enforced_soft_joint_pos_lower": contract.soft_lower,
                "enforced_soft_joint_pos_upper": contract.soft_upper,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
