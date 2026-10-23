#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Install an exported whole-body policy (contract microban-wholebody-1).

    PYTHONPATH=src uv run --locked python tools/install_wholebody_policy.py <wholebody.onnx> \\
        [--agents-dir src/agents] [--source "mjlab_microban <commit> <run>"]

Checks the file against the contract (HOME stamp, observation layout, timing
fields) and runs its self-test in ONNX Runtime (CPU), then writes
``wholebody.onnx`` and ``wholebody_manifest.json`` into the agents directory,
each atomically (the ONNX first: a manifest that names another file is
refused at start) and loads the installed pair as the robot does.  The
walk/get-up/IK tracking files and their manifest are not touched.  A trial export
(``dry_run_not_deployable``) installs only with MICROBAN_ALLOW_DRYRUN_POLICY=1
and then also runs only with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from home_pose import HOME_TAG
from policy_contract import AGENTS_DIR, PolicyContractError, PolicySelfTestError
from wholebody_contract import (
    CONTRACT,
    MANIFEST_FILE,
    POLICY_FILE,
    check_onnx_file,
    load_wholebody_policy,
)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def install(onnx: Path, agents_dir: Path = AGENTS_DIR, source: str | None = None) -> dict:
    data, contract, rows = check_onnx_file(onnx)
    manifest = {
        "contract": CONTRACT,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": contract.dry_run,
        "home_tag": HOME_TAG,
        "file": POLICY_FILE,
        "sha256": hashlib.sha256(data).hexdigest(),
        "checkpoint": contract.checkpoint_filename,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "benchmark_report_sha256": contract.benchmark_report_sha256,
        "eval_only_benchmark_report_sha256": contract.eval_only_benchmark_report_sha256,
        "self_test_rows": rows,
        "source": source,
    }
    agents_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(agents_dir / POLICY_FILE, data)
    _atomic_write(agents_dir / MANIFEST_FILE, (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode())
    loaded = load_wholebody_policy(agents_dir / POLICY_FILE, providers=["CPUExecutionProvider"])
    if loaded.contract.checkpoint_sha256 != contract.checkpoint_sha256:
        raise PolicyContractError("the installed file is not the one checked")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--agents-dir", type=Path, default=AGENTS_DIR)
    parser.add_argument("--source", default=None, help="Where the file came from (training commit, run)")
    args = parser.parse_args(argv)
    try:
        manifest = install(args.onnx, args.agents_dir, args.source)
    except (OSError, PolicyContractError, PolicySelfTestError) as exc:
        print(json.dumps({"status": "fail", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps({"status": "installed", **manifest}, indent=2))
    if manifest["dry_run"]:
        print("WARNING: a DRY RUN (trial) policy: it runs only with MICROBAN_ALLOW_DRYRUN_POLICY=1", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
