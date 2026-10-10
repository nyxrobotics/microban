import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "install_pico_policy.py"
SPEC = importlib.util.spec_from_file_location("install_pico_policy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


WALK_SHA = "a7" * 32
PICO_SHA = "b4" * 32
GATE_SHA = "c3" * 32
TRAINING_COMMIT = "d5" * 20


def contract(*, gate_sha=GATE_SHA, checkpoint_sha=PICO_SHA):
    return SimpleNamespace(
        checkpoint_filename="model_14999.pt",
        checkpoint_iteration=14999,
        checkpoint_sha256=checkpoint_sha,
        gate_report_sha256=gate_sha,
        pico=SimpleNamespace(
            curriculum={
                "critic_warmup": 1,
                "arm_start": 2,
                "foot_start": 3,
                "foot_tighten": 4,
                "total": 15000,
            }
        ),
    )


class PicoPolicyInstallerTest(unittest.TestCase):
    def manifest(self):
        return {
            "contract": "microban-policy-1",
            "home_tag": "forward_lean_home",
            "dry_run": False,
            "training_branch": "home-config",
            "training_commit": "0" * 40,
            "home_yaml_sha256": "9" * 64,
            "created": "2026-10-08 23:25:37",
            "policies": {
                "walk": {
                    "file": "walk.onnx",
                    "sha256": "1" * 64,
                    "checkpoint_sha256": WALK_SHA,
                },
                "getup": {
                    "file": "getup.onnx",
                    "sha256": "2" * 64,
                    "checkpoint_sha256": "3" * 64,
                },
                "pico": {
                    "file": "pico_teleop.onnx",
                    "sha256": "4" * 64,
                    "checkpoint_sha256": "5" * 64,
                },
            },
            "checkpoints": {
                "walk": "model_29000.pt",
                "getup": "model_17999.pt",
                "pico": "model_8999.pt",
            },
            "judgments": {
                "walk": {"passed": True},
                "getup": {"passed": True},
                "pico": {"passed": True},
            },
        }

    def test_manifest_replaces_only_pico_policy_entry(self):
        before = self.manifest()
        provenance = {
            "scope": "pico_only",
            "training_branch": "tracking-v13c",
            "training_commit": TRAINING_COMMIT,
            "training_home_yaml_sha256": "7" * 64,
        }
        after = installer._updated_manifest(
            before,
            pico_sha256="6" * 64,
            contract=contract(),
            pico_provenance=provenance,
        )

        self.assertEqual(after["policies"]["walk"], before["policies"]["walk"])
        self.assertEqual(after["policies"]["getup"], before["policies"]["getup"])
        self.assertEqual(after["checkpoints"]["walk"], "model_29000.pt")
        self.assertEqual(after["checkpoints"]["getup"], "model_17999.pt")
        self.assertEqual(after["policies"]["pico"]["sha256"], "6" * 64)
        self.assertEqual(after["policies"]["pico"]["checkpoint_sha256"], PICO_SHA)
        self.assertEqual(after["checkpoints"]["pico"], "model_14999.pt")
        # The top-level fields still identify the preserved three-policy base
        # release.  The later PICO has its own, truthful provenance record.
        for key in ("training_commit", "training_branch", "home_yaml_sha256", "created"):
            self.assertEqual(after[key], before[key])
        self.assertEqual(after["policy_provenance"]["pico"], provenance)

    def test_stage_gate_must_be_the_one_embedded_in_onnx(self):
        gate = {
            "schema_version": 3,
            "gate": "microban_teleop_v12_stage",
            "status": "pass",
            "checkpoint_sha256": PICO_SHA,
            "iteration": 14999,
            "completed_updates": 15000,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(gate), encoding="utf-8")
            gate_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(installer._load_stage_gate(path, contract(gate_sha=gate_sha)), gate)

            gate["completed_updates"] = 14999
            path.write_text(json.dumps(gate), encoding="utf-8")
            gate_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(installer.PolicyContractError):
                installer._load_stage_gate(path, contract(gate_sha=gate_sha))

    def test_stage_gate_hash_and_json_use_one_byte_snapshot(self):
        gate = {
            "schema_version": 3,
            "gate": "microban_teleop_v12_stage",
            "status": "pass",
            "checkpoint_sha256": PICO_SHA,
            "iteration": 14999,
            "completed_updates": 15000,
        }
        data = json.dumps(gate).encode()

        class CountingPath:
            calls = 0

            def read_bytes(self):
                self.calls += 1
                return data

            def __str__(self):
                return "counting-gate.json"

        path = CountingPath()
        loaded = installer._load_stage_gate(
            path, contract(gate_sha=hashlib.sha256(data).hexdigest())
        )
        self.assertEqual(loaded, gate)
        self.assertEqual(path.calls, 1)

    def test_training_provenance_is_checkpoint_and_git_record_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "training"
            repository.mkdir()
            subprocess.run(
                ["git", "init", "-b", "tracking-v13c"],
                cwd=repository,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=repository,
                check=True,
            )
            (repository / "config").mkdir()
            (repository / "config" / "home_pose.yaml").write_text("home: forward\n")
            subprocess.run(["git", "add", "config/home_pose.yaml"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-m", "training source"],
                cwd=repository,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            checkpoint = repository / "logs" / "run" / "model_14999.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            snapshot = checkpoint.parent / "git" / "mjlab_microban.diff"
            snapshot.parent.mkdir()
            snapshot.write_text(
                "--- git commit ---\n"
                + commit
                + "\n\n--- git status ---\n"
                + "On branch tracking-v13c\n"
                + "nothing to commit, working tree clean\n\n"
            )
            # A later documentation commit may advance the branch while the
            # model still truthfully belongs to the captured training commit.
            (repository / "README.md").write_text("later documentation\n")
            subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-m", "later docs"],
                cwd=repository,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            branch_tip = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            record_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            provenance = installer._training_provenance(
                repository=repository,
                branch="tracking-v13c",
                commit=commit,
                gate={"checkpoint": "${MJLAB_REPO}/logs/run/model_14999.pt"},
                contract=contract(checkpoint_sha=checkpoint_sha),
                pico_sha256="6" * 64,
                onnx_provenance={
                    "training_git_commit": commit,
                    "training_git_branch": "tracking-v13c",
                    "training_git_record_sha256": record_sha,
                },
            )
            self.assertEqual(provenance["training_commit"], commit)
            self.assertNotEqual(branch_tip, commit)
            self.assertEqual(provenance["training_branch_tip_at_install"], branch_tip)
            self.assertEqual(provenance["run_git_snapshot_sha256"], record_sha)

            with self.assertRaisesRegex(
                installer.PolicyContractError, "ONNX training provenance"
            ):
                installer._training_provenance(
                    repository=repository,
                    branch="tracking-v13c",
                    commit=commit,
                    gate={"checkpoint": "${MJLAB_REPO}/logs/run/model_14999.pt"},
                    contract=contract(checkpoint_sha=checkpoint_sha),
                    pico_sha256="6" * 64,
                    onnx_provenance={
                        "training_git_commit": commit,
                        "training_git_branch": "tracking-v13c",
                        "training_git_record_sha256": "f" * 64,
                    },
                )

    def test_every_publication_crash_point_is_recovered(self):
        class SimulatedCrash(BaseException):
            pass

        old_pico = b"old-pico"
        old_manifest = b"old-manifest"
        new_pico = b"new-pico"
        new_manifest = b"new-manifest"
        for crash_after_write in range(1, 6):
            with self.subTest(crash_after_write=crash_after_write), tempfile.TemporaryDirectory() as directory:
                agents = Path(directory)
                pico_path = agents / installer.POLICY_FILES["pico"]
                manifest_path = agents / "manifest.json"
                pico_path.write_bytes(old_pico)
                manifest_path.write_bytes(old_manifest)
                real_atomic_write = installer._atomic_write
                calls = 0

                def crash_write(path, data):
                    nonlocal calls
                    real_atomic_write(path, data)
                    calls += 1
                    if calls == crash_after_write:
                        raise SimulatedCrash()

                with mock.patch.object(
                    installer, "_validate_bundle", return_value={"pico": {}}
                ):
                    with mock.patch.object(installer, "_atomic_write", side_effect=crash_write):
                        with self.assertRaises(SimulatedCrash):
                            installer._publish_bundle(
                                agents_dir=agents,
                                pico_data=new_pico,
                                manifest_data=new_manifest,
                                expected_old_pico_sha256=hashlib.sha256(old_pico).hexdigest(),
                                expected_old_manifest_sha256=hashlib.sha256(old_manifest).hexdigest(),
                            )
                    recovered = installer._recover_pending_install(agents)

                if crash_after_write == 5:
                    self.assertEqual(recovered, "finalized")
                    self.assertEqual(pico_path.read_bytes(), new_pico)
                    self.assertEqual(manifest_path.read_bytes(), new_manifest)
                else:
                    self.assertIn(recovered, {"discarded_orphan_backups", "rolled_back"})
                    self.assertEqual(pico_path.read_bytes(), old_pico)
                    self.assertEqual(manifest_path.read_bytes(), old_manifest)
                self.assertFalse(any(path.name.startswith(".pico-policy-install") for path in agents.iterdir()))

    def test_validation_failure_rolls_back_before_returning(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = Path(directory)
            pico_path = agents / installer.POLICY_FILES["pico"]
            manifest_path = agents / "manifest.json"
            pico_path.write_bytes(b"old-pico")
            manifest_path.write_bytes(b"old-manifest")
            with mock.patch.object(
                installer, "_validate_bundle", side_effect=RuntimeError("bad bundle")
            ):
                with self.assertRaisesRegex(RuntimeError, "bad bundle"):
                    installer._publish_bundle(
                        agents_dir=agents,
                        pico_data=b"new-pico",
                        manifest_data=b"new-manifest",
                        expected_old_pico_sha256=hashlib.sha256(b"old-pico").hexdigest(),
                        expected_old_manifest_sha256=hashlib.sha256(b"old-manifest").hexdigest(),
                    )
            self.assertEqual(pico_path.read_bytes(), b"old-pico")
            self.assertEqual(manifest_path.read_bytes(), b"old-manifest")
            self.assertFalse(any(path.name.startswith(".pico-policy-install") for path in agents.iterdir()))

    def test_startup_removes_only_its_stale_atomic_temps(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = Path(directory)
            stale = agents / ".pico_teleop.onnx.interrupted.tmp"
            unrelated = agents / "keep.tmp"
            stale.write_bytes(b"incomplete")
            unrelated.write_bytes(b"keep")
            installer._remove_stale_atomic_temps(agents)
            self.assertFalse(stale.exists())
            self.assertEqual(unrelated.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
