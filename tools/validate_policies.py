#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Validate the installed walk, get-up and PICO policies without opening interfaces.

Checks src/agents/manifest.json, each ONNX against contract microban-policy-1
(src/policy_contract.py) and runs each policy's startup self-test under ONNX
Runtime's CPU provider, as the robot does.  Prints a JSON report; exits non-zero
on the first failure.

    PYTHONPATH=src uv run --locked python tools/validate_policies.py [src/agents]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from policy_contract import (
    AGENTS_DIR,
    KINDS,
    POLICY_CONTRACT,
    POLICY_FILES,
    PolicyContractError,
    PolicySelfTestError,
    load_installed_policy,
    load_manifest,
)


def validate_policies(agents_dir: Path = AGENTS_DIR) -> dict[str, object]:
    manifest = load_manifest(agents_dir)
    report: dict[str, object] = {
        "status": "pass",
        "contract": POLICY_CONTRACT,
        "agents_dir": str(Path(agents_dir).resolve()),
        "home_tag": manifest["home_tag"],
        "training_commit": manifest.get("training_commit"),
        "dry_run": manifest["dry_run"],
        "policies": {},
    }
    for kind in KINDS:
        loaded = load_installed_policy(
            kind, Path(agents_dir) / POLICY_FILES[kind], providers=["CPUExecutionProvider"]
        )
        if loaded.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("policy validation requires CPUExecutionProvider only")
        contract = loaded.contract
        report["policies"][kind] = {
            "file": POLICY_FILES[kind],
            "sha256": manifest["policies"][kind]["sha256"],
            "recipe": contract.recipe,
            "checkpoint": contract.checkpoint_filename,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "gate_report_sha256": contract.gate_report_sha256,
            "input_width": contract.input_width,
            "self_test_rows": loaded.self_test_rows,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agents_dir", nargs="?", type=Path, default=AGENTS_DIR)
    args = parser.parse_args()
    try:
        report = validate_policies(args.agents_dir)
    except (OSError, PolicyContractError, PolicySelfTestError, RuntimeError) as exc:
        print(json.dumps({"status": "fail", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
