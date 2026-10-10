#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Install one judged PICO policy without replacing walk or get-up.

The full HOME pipeline installs all three policies together.  This tool is
for a later PICO-only run: it checks the packaged ONNX and its stage gate,
builds a complete candidate agents directory around the already-installed
walk/get-up files, and validates that bundle before changing the worktree.

    PYTHONPATH=src uv run --locked python tools/install_pico_policy.py \
        <release>/pico_teleop.onnx <release>/model_14999_gate.json \
        --training-repo ../mjlab_microban \
        --training-branch tracking-v13c \
        --training-commit <40-hex-commit>

Only ``src/agents/pico_teleop.onnx`` and ``src/agents/manifest.json`` are
published.  Existing walk/get-up bytes and their manifest entries must pass
the runtime contract and are preserved exactly.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from policy_contract import (
    AGENTS_DIR,
    KINDS,
    POLICY_FILES,
    PolicyContract,
    PolicyContractError,
    PolicySelfTestError,
    load_installed_policy,
    load_manifest,
    open_session,
    parse_policy,
    run_self_test,
)


_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REPO_PATH_PREFIX = "${MJLAB_REPO}/"
_TRANSACTION_SCHEMA_VERSION = 1
_TRANSACTION_FILE = ".pico-policy-install.json"
_PICO_BACKUP_FILE = ".pico-policy-install.pico.bak"
_MANIFEST_BACKUP_FILE = ".pico-policy-install.manifest.bak"
_TRAINING_GIT_METADATA_KEYS = (
    "training_git_commit",
    "training_git_branch",
    "training_git_record_sha256",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(data: bytes, label: str) -> dict[str, Any]:
    """Decode one immutable byte snapshot as strict JSON."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise PolicyContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PolicyContractError(f"{label} must hold a JSON object")
    return value


def _git(repository: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _training_home_sha256(repository: Path, commit: str) -> str:
    resolved = _git(repository, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        raise ValueError(f"training commit resolves to {resolved!r}, expected {commit!r}")
    home_yaml = _git(repository, "show", f"{commit}:config/home_pose.yaml", binary=True)
    assert isinstance(home_yaml, bytes)
    return sha256_bytes(home_yaml)


def _training_branch_tip(repository: Path, branch: str, commit: str) -> tuple[str, str]:
    """Return a branch ref/tip that contains the declared training commit."""

    if (
        not branch
        or branch != branch.strip()
        or branch.startswith("-")
        or any(ord(character) < 32 for character in branch)
    ):
        raise ValueError("--training-branch must be one canonical branch name")
    checked = subprocess.run(
        ["git", "-C", str(repository), "check-ref-format", "--branch", branch],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if checked.returncode != 0:
        raise ValueError(f"--training-branch is invalid: {branch!r}")

    existing: list[str] = []
    for reference in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}"):
        resolved = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", f"{reference}^{{commit}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if resolved.returncode != 0:
            continue
        tip = resolved.stdout.strip()
        existing.append(reference)
        ancestor = subprocess.run(
            ["git", "-C", str(repository), "merge-base", "--is-ancestor", commit, tip],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if ancestor.returncode == 0:
            return reference, tip
        if ancestor.returncode not in (0, 1):
            raise subprocess.CalledProcessError(
                ancestor.returncode,
                ancestor.args,
                output=ancestor.stdout,
                stderr=ancestor.stderr,
            )
    if not existing:
        raise ValueError(f"training branch {branch!r} does not exist locally or at origin")
    raise ValueError(
        f"training commit {commit} is not reachable from training branch {branch!r}"
    )


def _load_pico(path: Path) -> tuple[bytes, PolicyContract, int, dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"PICO policy must be a regular file, not a symlink: {path}")
    data = path.read_bytes()
    session = open_session(data, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("PICO installation check requires CPUExecutionProvider only")
    contract = parse_policy("pico", session)
    if contract.dry_run:
        raise PolicyContractError("a dry-run PICO policy cannot be installed")
    rows = run_self_test(session, contract)
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    provenance = {key: metadata.get(key, "") for key in _TRAINING_GIT_METADATA_KEYS}
    if (
        _GIT_COMMIT_RE.fullmatch(provenance["training_git_commit"]) is None
        or not provenance["training_git_branch"].strip()
        or _SHA256_RE.fullmatch(provenance["training_git_record_sha256"]) is None
    ):
        raise PolicyContractError("PICO ONNX lacks its training Git provenance")
    return data, contract, rows, provenance


def _load_stage_gate(path: Path, contract: PolicyContract) -> dict[str, Any]:
    try:
        gate_data = path.read_bytes()
    except OSError as exc:
        raise PolicyContractError(f"cannot read PICO stage gate {path}: {exc}") from exc
    gate_sha256 = sha256_bytes(gate_data)
    if gate_sha256 != contract.gate_report_sha256:
        raise PolicyContractError(
            "PICO ONNX gate_report_sha256 does not match the supplied stage gate"
        )
    gate = _strict_json_object(gate_data, f"PICO stage gate {path}")
    assert contract.pico is not None
    expected = {
        "schema_version": 3,
        "gate": "microban_teleop_v12_stage",
        "status": "pass",
        "checkpoint_sha256": contract.checkpoint_sha256,
        "iteration": contract.checkpoint_iteration,
        "completed_updates": contract.pico.curriculum["total"],
    }
    wrong = [key for key, value in expected.items() if gate.get(key) != value]
    if wrong or contract.checkpoint_iteration + 1 != gate.get("completed_updates"):
        raise PolicyContractError(
            "PICO stage gate does not describe the packaged final checkpoint"
            + (f" ({', '.join(wrong)})" if wrong else "")
        )
    return gate


def _gate_checkpoint_path(training_repo: Path, gate: dict[str, Any]) -> tuple[Path, str]:
    """Resolve the canonical gate checkpoint inside the supplied training repo."""

    value = gate.get("checkpoint")
    if not isinstance(value, str) or not value:
        raise PolicyContractError("PICO stage gate lacks its checkpoint path")
    repository = training_repo.resolve()
    if value.startswith(_REPO_PATH_PREFIX):
        relative = Path(value.removeprefix(_REPO_PATH_PREFIX))
        if relative.is_absolute() or ".." in relative.parts:
            raise PolicyContractError("PICO stage gate checkpoint path is unsafe")
        checkpoint = (repository / relative).resolve()
    else:
        checkpoint = Path(value).expanduser().resolve()
    try:
        relative = checkpoint.relative_to(repository)
    except ValueError as exc:
        raise PolicyContractError(
            "PICO stage gate checkpoint is outside --training-repo"
        ) from exc
    return checkpoint, relative.as_posix()


def _training_provenance(
    *,
    repository: Path,
    branch: str,
    commit: str,
    gate: dict[str, Any],
    contract: PolicyContract,
    pico_sha256: str,
    onnx_provenance: dict[str, str],
) -> dict[str, Any]:
    """Bind the PICO artifact to its checkpoint and captured run Git state."""

    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("--training-commit must be a lowercase 40-digit Git commit")
    repository = repository.resolve()
    repository_root = Path(str(_git(repository, "rev-parse", "--show-toplevel"))).resolve()
    if repository_root != repository:
        raise ValueError("--training-repo must name the Git worktree root")
    resolved = _git(repository, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        raise ValueError(f"training commit resolves to {resolved!r}, expected {commit!r}")
    branch_ref, branch_tip = _training_branch_tip(repository, branch, commit)
    training_home_sha256 = _training_home_sha256(repository, commit)

    checkpoint, checkpoint_relative = _gate_checkpoint_path(repository, gate)
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or checkpoint.name != contract.checkpoint_filename
    ):
        raise PolicyContractError(
            "PICO stage gate checkpoint is missing, a symlink, or has the wrong filename"
        )
    if sha256_file(checkpoint) != contract.checkpoint_sha256:
        raise PolicyContractError(
            "PICO stage gate checkpoint bytes differ from the packaged ONNX"
        )

    git_snapshot = checkpoint.parent / "git" / "mjlab_microban.diff"
    if not git_snapshot.is_file() or git_snapshot.is_symlink():
        raise PolicyContractError(f"training run lacks its Git snapshot: {git_snapshot}")
    snapshot_data = git_snapshot.read_bytes()
    try:
        snapshot = snapshot_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyContractError("training run Git snapshot is not UTF-8") from exc
    commit_match = re.search(
        r"(?m)^--- git commit ---\r?$\n([0-9a-f]{40})\r?$", snapshot
    )
    branch_match = re.search(r"(?m)^On branch ([^\r\n]+)\r?$", snapshot)
    if commit_match is None or commit_match.group(1) != commit:
        raise PolicyContractError(
            "--training-commit differs from the commit captured when training started"
        )
    if branch_match is None or branch_match.group(1) != branch:
        raise PolicyContractError(
            "--training-branch differs from the branch captured when training started"
        )
    if (
        "\ndiff --git " in snapshot
        or "\nChanges not staged for commit:" in snapshot
        or "\nChanges to be committed:" in snapshot
    ):
        raise PolicyContractError(
            "training started with tracked Git changes; a commit cannot truthfully identify it"
        )
    expected_onnx_provenance = {
        "training_git_commit": commit,
        "training_git_branch": branch,
        "training_git_record_sha256": sha256_bytes(snapshot_data),
    }
    if onnx_provenance != expected_onnx_provenance:
        raise PolicyContractError(
            "PICO ONNX training provenance differs from its run Git snapshot"
        )

    return {
        "scope": "pico_only",
        "policy_sha256": pico_sha256,
        "checkpoint": contract.checkpoint_filename,
        "checkpoint_path": checkpoint_relative,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "stage_gate_sha256": contract.gate_report_sha256,
        "training_branch": branch,
        "training_branch_ref": branch_ref,
        "training_branch_tip_at_install": branch_tip,
        "training_commit": commit,
        "training_home_yaml_sha256": training_home_sha256,
        "run_git_snapshot": git_snapshot.relative_to(repository).as_posix(),
        "run_git_snapshot_sha256": onnx_provenance["training_git_record_sha256"],
        "installed_at": f"{datetime.now():%F %T}",
    }


def _updated_manifest(
    current: dict[str, Any],
    *,
    pico_sha256: str,
    contract: PolicyContract,
    pico_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Return a mixed release manifest while preserving walk/get-up entries."""

    result = copy.deepcopy(current)
    result["dry_run"] = False
    result["policies"]["pico"] = {
        "file": POLICY_FILES["pico"],
        "sha256": pico_sha256,
        "checkpoint_sha256": contract.checkpoint_sha256,
    }
    checkpoints = result.setdefault("checkpoints", {})
    if not isinstance(checkpoints, dict):
        raise PolicyContractError("manifest checkpoints record is malformed")
    checkpoints["pico"] = contract.checkpoint_filename
    judgments = result.setdefault("judgments", {})
    if not isinstance(judgments, dict):
        raise PolicyContractError("manifest judgments record is malformed")
    judgments["pico"] = {
        "failures": [],
        "passed": True,
        "checkpoint": contract.checkpoint_filename,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "gate_report_sha256": contract.gate_report_sha256,
    }
    policy_provenance = result.setdefault("policy_provenance", {})
    if not isinstance(policy_provenance, dict):
        raise PolicyContractError("manifest policy_provenance record is malformed")
    policy_provenance["pico"] = copy.deepcopy(pico_provenance)
    return result


def _validate_bundle(agents_dir: Path) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        loaded = load_installed_policy(
            kind,
            agents_dir / POLICY_FILES[kind],
            providers=["CPUExecutionProvider"],
        )
        if loaded.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("bundle validation requires CPUExecutionProvider only")
        policies[kind] = {
            "file": POLICY_FILES[kind],
            "sha256": sha256_file(agents_dir / POLICY_FILES[kind]),
            "checkpoint": loaded.contract.checkpoint_filename,
            "checkpoint_sha256": loaded.contract.checkpoint_sha256,
            "self_test_rows": loaded.self_test_rows,
        }
    return policies


def _fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def _installation_lock(agents_dir: Path) -> Iterator[None]:
    """Serialize installers without leaving a file in the Git worktree."""

    identity = sha256_bytes(str(agents_dir.resolve()).encode("utf-8"))[:24]
    lock_path = Path(tempfile.gettempdir()) / f"microban-pico-policy-{identity}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        information = os.fstat(descriptor)
        if not stat.S_ISREG(information.st_mode) or information.st_uid != os.geteuid():
            raise RuntimeError(f"unsafe PICO installer lock file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _remove_stale_atomic_temps(agents_dir: Path) -> None:
    """Remove harmless incomplete temp files left before an atomic rename."""

    bases = (
        _PICO_BACKUP_FILE,
        _MANIFEST_BACKUP_FILE,
        _TRANSACTION_FILE,
        POLICY_FILES["pico"],
        "manifest.json",
    )
    changed = False
    if not agents_dir.is_dir():
        return
    for path in agents_dir.iterdir():
        if not any(
            path.name.startswith(f".{base}.") and path.name.endswith(".tmp")
            for base in bases
        ):
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"unsafe stale PICO installer temporary: {path}")
        path.unlink()
        changed = True
    if changed:
        _fsync_directory(agents_dir)


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _transaction_paths(agents_dir: Path) -> tuple[Path, Path, Path]:
    return (
        agents_dir / _TRANSACTION_FILE,
        agents_dir / _PICO_BACKUP_FILE,
        agents_dir / _MANIFEST_BACKUP_FILE,
    )


def _transaction_document(
    *, old_pico: bytes, old_manifest: bytes, new_pico: bytes, new_manifest: bytes
) -> dict[str, Any]:
    return {
        "schema_version": _TRANSACTION_SCHEMA_VERSION,
        "operation": "install_pico_policy",
        "old": {
            "pico_sha256": sha256_bytes(old_pico),
            "manifest_sha256": sha256_bytes(old_manifest),
        },
        "new": {
            "pico_sha256": sha256_bytes(new_pico),
            "manifest_sha256": sha256_bytes(new_manifest),
        },
    }


def _validated_transaction(data: bytes, path: Path) -> dict[str, Any]:
    transaction = _strict_json_object(data, f"PICO install transaction {path}")
    if set(transaction) != {"schema_version", "operation", "old", "new"} or (
        transaction.get("schema_version") != _TRANSACTION_SCHEMA_VERSION
        or transaction.get("operation") != "install_pico_policy"
    ):
        raise PolicyContractError("PICO install transaction header is malformed")
    for state in ("old", "new"):
        value = transaction.get(state)
        if not isinstance(value, dict) or set(value) != {
            "pico_sha256",
            "manifest_sha256",
        }:
            raise PolicyContractError(f"PICO install transaction {state} state is malformed")
        if any(
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
            for digest in value.values()
        ):
            raise PolicyContractError(
                f"PICO install transaction {state} hashes are malformed"
            )
    return transaction


def _remove_transaction_files(agents_dir: Path) -> None:
    journal, pico_backup, manifest_backup = _transaction_paths(agents_dir)
    changed = False
    # The journal is the commit marker: remove backups first and it last.
    for path in (pico_backup, manifest_backup, journal):
        try:
            path.unlink()
            changed = True
        except FileNotFoundError:
            pass
    if changed:
        _fsync_directory(agents_dir)


def _restore_transaction_old(
    agents_dir: Path, transaction: dict[str, Any]
) -> None:
    journal, pico_backup, manifest_backup = _transaction_paths(agents_dir)
    del journal
    try:
        old_pico = pico_backup.read_bytes()
        old_manifest = manifest_backup.read_bytes()
    except OSError as exc:
        raise RuntimeError("PICO install recovery lacks an intact backup") from exc
    expected = transaction["old"]
    if (
        sha256_bytes(old_pico) != expected["pico_sha256"]
        or sha256_bytes(old_manifest) != expected["manifest_sha256"]
    ):
        raise RuntimeError("PICO install recovery backup hash differs from its journal")
    pico_path = agents_dir / POLICY_FILES["pico"]
    manifest_path = agents_dir / "manifest.json"
    _atomic_write(pico_path, old_pico)
    _atomic_write(manifest_path, old_manifest)
    if (
        sha256_file(pico_path) != expected["pico_sha256"]
        or sha256_file(manifest_path) != expected["manifest_sha256"]
    ):
        raise RuntimeError("PICO install recovery could not restore the previous pair")


def _recover_pending_install(agents_dir: Path) -> str:
    """Finish or roll back a durable transaction left by an interrupted run."""

    journal, pico_backup, manifest_backup = _transaction_paths(agents_dir)
    pico_path = agents_dir / POLICY_FILES["pico"]
    manifest_path = agents_dir / "manifest.json"
    if not journal.exists():
        orphans = [path for path in (pico_backup, manifest_backup) if path.exists()]
        if not orphans:
            return "none"
        live_by_backup = {
            pico_backup: pico_path,
            manifest_backup: manifest_path,
        }
        for backup in orphans:
            live = live_by_backup[backup]
            if not live.is_file() or sha256_file(backup) != sha256_file(live):
                raise RuntimeError(
                    "orphaned PICO install backup differs from the live file; refusing to guess"
                )
        _remove_transaction_files(agents_dir)
        return "discarded_orphan_backups"

    transaction = _validated_transaction(journal.read_bytes(), journal)
    actual = {
        "pico_sha256": sha256_file(pico_path) if pico_path.is_file() else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
    }
    if actual == transaction["new"]:
        try:
            _validate_bundle(agents_dir)
        except Exception:
            _restore_transaction_old(agents_dir, transaction)
            _remove_transaction_files(agents_dir)
            return "rolled_back"
        _remove_transaction_files(agents_dir)
        return "finalized"
    if actual == transaction["old"]:
        _remove_transaction_files(agents_dir)
        return "rolled_back"

    _restore_transaction_old(agents_dir, transaction)
    _remove_transaction_files(agents_dir)
    return "rolled_back"


def _publish_bundle(
    *,
    agents_dir: Path,
    pico_data: bytes,
    manifest_data: bytes,
    expected_old_pico_sha256: str,
    expected_old_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Publish the model/manifest pair with a durable recovery journal."""

    journal, pico_backup, manifest_backup = _transaction_paths(agents_dir)
    if any(path.exists() for path in (journal, pico_backup, manifest_backup)):
        raise RuntimeError("PICO install recovery must run before publication")
    pico_path = agents_dir / POLICY_FILES["pico"]
    manifest_path = agents_dir / "manifest.json"
    old_pico = pico_path.read_bytes()
    old_manifest = manifest_path.read_bytes()
    if (
        sha256_bytes(old_pico) != expected_old_pico_sha256
        or sha256_bytes(old_manifest) != expected_old_manifest_sha256
    ):
        raise RuntimeError("installed policy bundle changed while the candidate was checked")

    transaction = _transaction_document(
        old_pico=old_pico,
        old_manifest=old_manifest,
        new_pico=pico_data,
        new_manifest=manifest_data,
    )
    transaction_data = (
        json.dumps(transaction, indent=1, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        _atomic_write(pico_backup, old_pico)
        _atomic_write(manifest_backup, old_manifest)
        _atomic_write(journal, transaction_data)
        _atomic_write(pico_path, pico_data)
        _atomic_write(manifest_path, manifest_data)
        policies = _validate_bundle(agents_dir)
    except Exception:
        try:
            _recover_pending_install(agents_dir)
        except Exception as recovery_error:
            raise RuntimeError(
                "PICO publication failed and automatic rollback also failed; "
                f"keep {journal} and its backups for recovery"
            ) from recovery_error
        raise
    _remove_transaction_files(agents_dir)
    return policies


def _install_locked(
    *,
    source: Path,
    stage_gate: Path,
    agents_dir: Path,
    training_repo: Path,
    training_branch: str,
    training_commit: str,
    check_only: bool,
) -> dict[str, Any]:
    # A previous process may have stopped between the two atomic renames.
    # Recover before reading either member of the installed pair.
    _remove_stale_atomic_temps(agents_dir)
    startup_recovery = _recover_pending_install(agents_dir)

    current = load_manifest(agents_dir)
    if current["dry_run"]:
        raise PolicyContractError("cannot base a production PICO release on a dry-run bundle")
    manifest_path = agents_dir / "manifest.json"
    current_manifest_data = manifest_path.read_bytes()
    if _strict_json_object(current_manifest_data, f"policy manifest {manifest_path}") != current:
        raise RuntimeError("installed manifest changed while it was validated")
    pico_path = agents_dir / POLICY_FILES["pico"]
    old_pico_sha256 = sha256_file(pico_path)
    old_manifest_sha256 = sha256_bytes(current_manifest_data)
    # Validate the two files that this tool promises not to replace.
    preserved: dict[str, bytes] = {}
    for kind in ("walk", "getup"):
        path = agents_dir / POLICY_FILES[kind]
        load_installed_policy(kind, path, providers=["CPUExecutionProvider"])
        data = path.read_bytes()
        if sha256_bytes(data) != current["policies"][kind]["sha256"]:
            raise RuntimeError(f"installed {kind} changed while it was validated")
        preserved[kind] = data

    pico_data, contract, source_self_test_rows, onnx_provenance = _load_pico(source)
    gate = _load_stage_gate(stage_gate, contract)
    assert contract.pico is not None
    installed_walker = current["policies"]["walk"]["checkpoint_sha256"]
    if contract.pico.walk_checkpoint_sha256 != installed_walker:
        raise PolicyContractError(
            "PICO frozen-walker checkpoint differs from the installed walk policy"
        )

    pico_sha256 = sha256_bytes(pico_data)
    pico_provenance = _training_provenance(
        repository=training_repo,
        branch=training_branch,
        commit=training_commit,
        gate=gate,
        contract=contract,
        pico_sha256=pico_sha256,
        onnx_provenance=onnx_provenance,
    )
    candidate_manifest = _updated_manifest(
        current,
        pico_sha256=pico_sha256,
        contract=contract,
        pico_provenance=pico_provenance,
    )
    manifest_data = (
        json.dumps(candidate_manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    agents_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pico-policy-candidate-", dir=agents_dir.parent
    ) as temporary_root:
        candidate = Path(temporary_root) / "agents"
        candidate.mkdir()
        for kind in ("walk", "getup"):
            (candidate / POLICY_FILES[kind]).write_bytes(preserved[kind])
        (candidate / POLICY_FILES["pico"]).write_bytes(pico_data)
        (candidate / "manifest.json").write_bytes(manifest_data)
        policies = _validate_bundle(candidate)

    if not check_only:
        policies = _publish_bundle(
            agents_dir=agents_dir,
            pico_data=pico_data,
            manifest_data=manifest_data,
            expected_old_pico_sha256=old_pico_sha256,
            expected_old_manifest_sha256=old_manifest_sha256,
        )

    return {
        "status": "pass",
        "check_only": check_only,
        "startup_recovery": startup_recovery,
        "agents_dir": str(agents_dir.resolve()),
        "training_branch": training_branch,
        "training_commit": training_commit,
        "training_home_yaml_sha256": pico_provenance["training_home_yaml_sha256"],
        "training_git_record_sha256": pico_provenance["run_git_snapshot_sha256"],
        "stage_gate_sha256": contract.gate_report_sha256,
        "source_self_test_rows": source_self_test_rows,
        "policies": policies,
    }


def install(
    *,
    source: Path,
    stage_gate: Path,
    agents_dir: Path,
    training_repo: Path,
    training_branch: str,
    training_commit: str,
    check_only: bool,
) -> dict[str, Any]:
    with _installation_lock(agents_dir):
        return _install_locked(
            source=source,
            stage_gate=stage_gate,
            agents_dir=agents_dir,
            training_repo=training_repo,
            training_branch=training_branch,
            training_commit=training_commit,
            check_only=check_only,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="packaged pico_teleop.onnx")
    parser.add_argument("stage_gate", type=Path, help="the stage gate used by the packager")
    parser.add_argument("--agents-dir", type=Path, default=AGENTS_DIR)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--training-branch", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument(
        "--check-only", action="store_true", help="validate and print the receipt without writing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = install(
            source=args.source,
            stage_gate=args.stage_gate,
            agents_dir=args.agents_dir,
            training_repo=args.training_repo,
            training_branch=args.training_branch,
            training_commit=args.training_commit,
            check_only=args.check_only,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        ValueError,
        PolicyContractError,
        PolicySelfTestError,
        RuntimeError,
    ) as exc:
        print(
            json.dumps(
                {"status": "fail", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
