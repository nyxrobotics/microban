"""Contract microban-policy-1: every policy's metadata, self-test and manifest.

The synthetic sessions (policy_fixtures) carry this process's HOME.  The real
package of a 64-env training dry run (tests/fixtures/policies, forward-lean
HOME) is checked in subprocesses that select that HOME.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


import policy_contract as pc
from constants import NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from home_pose import HOME_TAG
from policy_fixtures import (
    ACTION_COUNT,
    CENTERED_HOME_YAML,
    LEAN_HOME_YAML,
    OLD_HOME,
    POLICY_PACKAGE,
    WALK_CHECKPOINT_SHA256,
    WIDTHS,
    contract_metadata,
    csv,
    fake_session,
    home_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse(kind, metadata=None, **kwargs):
    return pc.parse_policy(kind, fake_session(kind, metadata, **kwargs))


class ContractMetadataTest(unittest.TestCase):
    def test_every_kind_parses_at_this_home(self):
        for kind in pc.KINDS:
            with self.subTest(kind=kind):
                contract = parse(kind)
                self.assertEqual(contract.kind, kind)
                self.assertEqual(contract.recipe, pc.RECIPES[kind])
                self.assertEqual(contract.input_width, WIDTHS[kind])
                self.assertEqual(contract.self_test_observations.shape, (8, 1, WIDTHS[kind]))
                self.assertEqual(contract.pico is not None, kind == "pico")

    def test_one_contract_and_recipe_per_kind(self):
        self.assertEqual(pc.POLICY_CONTRACT, "microban-policy-1")
        self.assertEqual(set(pc.RECIPES), {"walk", "getup", "pico"})
        self.assertEqual({WIDTHS[kind] for kind in ("walk", "getup", "pico")}, {63, 60, 83})

    def test_missing_fields_fail_closed(self):
        for kind in pc.KINDS:
            for key in contract_metadata(kind):
                metadata = contract_metadata(kind)
                del metadata[key]
                with self.subTest(kind=kind, missing=key), self.assertRaises(pc.PolicyContractError):
                    parse(kind, metadata)

    def test_drifted_common_fields_fail_closed(self):
        three_decimal = ",".join(
            f"{NEUTRAL_POSE[name]:.3f}"
            for name in contract_metadata("walk")["joint_names"].split(",")
        )
        cases = {
            "contract": {"microban_policy_contract": "microban-policy-0"},
            "kind": {"microban_policy_kind": "pico"},
            "recipe": {"microban_recipe": "microban-walk-exp-tracking-1"},
            "home_stamp": {"home_pose": contract_metadata("walk", OLD_HOME)["home_pose"]},
            "home_defaults": {"default_joint_pos": contract_metadata("walk", OLD_HOME)["default_joint_pos"]},
            "three_decimal_defaults": {"default_joint_pos": three_decimal},
            "joint_names": {"joint_names": ",".join(reversed(contract_metadata("walk")["joint_names"].split(",")[:-1]))},
            "action_order": {"action_joint_names": ",".join(reversed(OBSERVATION_DOF_ORDER))},
            "scale": {"action_scale": "0.5"},
            "clip157": {"action_clip_upper": csv([1.57] * ACTION_COUNT)},
            "clip_three_decimal": {"action_clip_upper": ",".join(["3.142"] * ACTION_COUNT)},
            "clip_wide": {"action_clip_lower": csv([-math.pi - 1.0e-5] * ACTION_COUNT)},
            "clip_short": {"action_clip_lower": csv([-math.pi] * (ACTION_COUNT - 1))},
            "clip_nan": {"action_clip_lower": ",".join(["nan"] * ACTION_COUNT)},
            "previous_action": {"previous_action_semantics": "clipped_target"},
            "gyro_frame": {"base_ang_vel_frame": "robot_body_xyz"},
            "control_hz": {"control_hz": "100"},
            "schema": {"observation_schema_json": json.dumps([["base_ang_vel", 3]])},
            "observation_joints": {"observation_joint_names": ",".join(pc.OBSERVATION_JOINT_NAMES["pico"])},
            "checkpoint_name": {"checkpoint_filename": "best.pt"},
            "checkpoint_iteration": {"checkpoint_iteration": "8998"},
            "checkpoint_sha": {"checkpoint_sha256": "A" * 64},
            "gate": {"gate_status": "fail"},
            "gate_sha": {"gate_report_sha256": "0" * 63},
            "self_test_json": {"self_test_actions_json": "[[NaN]]"},
        }
        for case, change in cases.items():
            with self.subTest(case=case), self.assertRaises(pc.PolicyContractError):
                parse("walk", {**contract_metadata("walk"), **change})

    def test_tensor_names_types_and_widths_are_fixed(self):
        for case, kwargs in (
            ("input_name", {"input_name": "input"}),
            ("output_name", {"output_name": "output"}),
            ("input_type", {"input_type": "tensor(double)"}),
            ("output_type", {"output_type": "tensor(int64)"}),
            ("reference_phase_width", {"input_width": WIDTHS["walk"] + 2}),
            ("output_width", {"output_width": ACTION_COUNT - 1}),
        ):
            with self.subTest(case=case), self.assertRaises(pc.PolicyContractError):
                parse("walk", **kwargs)

    def test_pico_fields_fail_closed(self):
        curriculum = json.loads(contract_metadata("pico")["pico_curriculum_json"])
        cases = {
            "frame": {"pico_target_frame": "robot_imu_xyz"},
            "hand_fk": {"pico_hand_target_fk_json": json.dumps({"revision": "other"})},
            "wide_foot": {"pico_foot_target_upper_json": json.dumps([0.03, 0.03, 0.06] * 2)},
            "wide_both_feet": {"pico_both_feet_target_upper_json": json.dumps([0.02, 0.01, 0.02] * 2)},
            "wide_hands": {"pico_hand_target_upper_json": json.dumps([0.1] * 6)},
            "guard_zero": {"pico_raw_action_guard_json": json.dumps([0.0] * 18)},
            "guard_short": {"pico_raw_action_guard_json": json.dumps([1.0] * 17)},
            "guard_overflow": {"pico_raw_action_guard_json": json.dumps([1.0e40] * 18)},
            "curriculum_order": {
                "pico_curriculum_json": json.dumps({**curriculum, "foot_start": 7000})
            },
            "checkpoint_before_final_stage": {
                "checkpoint_filename": "model_4999.pt",
                "checkpoint_iteration": "4999",
            },
            "checkpoint_after_total": {
                "checkpoint_filename": "model_14999.pt",
                "checkpoint_iteration": "14999",
            },
            "adapter_columns": {"pico_active_adapter_columns_json": json.dumps([6, 7, 8])},
            "walker_sha": {"pico_walk_checkpoint_sha256": "x" * 64},
        }
        for case, change in cases.items():
            with self.subTest(case=case), self.assertRaises(pc.PolicyContractError):
                parse("pico", {**contract_metadata("pico"), **change})

    def test_dry_run_package_needs_the_pipeline_switch(self):
        metadata = {**contract_metadata("getup"), pc.DRY_RUN_METADATA_KEY: "true"}
        with mock.patch.dict(os.environ, {pc.DRY_RUN_POLICY_ALLOW_ENV: "0"}):
            with self.assertRaisesRegex(pc.PolicyContractError, "DRY RUN"):
                parse("getup", metadata)
        with mock.patch.dict(os.environ, {pc.DRY_RUN_POLICY_ALLOW_ENV: "1"}):
            self.assertTrue(parse("getup", metadata).dry_run)


class SelfTestTest(unittest.TestCase):
    def rows(self, kind, mutate):
        row = home_observation(kind)
        mutate(row)
        return json.dumps([row] * 8)

    def test_recorded_observations_must_be_physical(self):
        offsets = pc._term_offsets("pico")
        position = offsets["joint_pos"][0] + 3  # first body joint (after head/neck)
        velocity = offsets["joint_vel"][0]
        cases = {
            "too_few": json.dumps([home_observation("pico")] * 7),
            "too_many": json.dumps([home_observation("pico")] * 65),
            "width": json.dumps([home_observation("walk")] * 8),
            "gravity": self.rows("pico", lambda row: row.__setitem__(5, -0.5)),
            "joint_range": self.rows("pico", lambda row: row.__setitem__(position, 4.0)),
            "joint_speed": self.rows("pico", lambda row: row.__setitem__(velocity, 13.0)),
            "nan": json.dumps([[math.nan] * 83] * 8),
        }
        for case, rows in cases.items():
            metadata = {**contract_metadata("pico"), "self_test_observations_json": rows}
            with self.subTest(case=case), self.assertRaises(pc.PolicyContractError):
                parse("pico", metadata)

    def test_onnxruntime_must_reproduce_the_recorded_actions(self):
        recorded = [[0.5 * index for index in range(ACTION_COUNT)]] * 8
        metadata = contract_metadata("getup", self_test_actions=recorded)
        bound = pc.SELF_TEST_ATOL + pc.SELF_TEST_RTOL * 8.5
        for case, output, passes in (
            ("exact", recorded[0], True),
            ("within", [value + 0.9 * bound for value in recorded[0]], True),
            ("outside", [value + 1.1 * bound for value in recorded[0]], False),
            ("nan", [math.nan] * ACTION_COUNT, False),
        ):
            session = fake_session("getup", metadata, outputs=[output])
            contract = pc.parse_policy("getup", session)
            with self.subTest(case=case):
                if passes:
                    self.assertEqual(pc.run_self_test(session, contract), 8)
                else:
                    with self.assertRaises(pc.PolicySelfTestError):
                        pc.run_self_test(session, contract)

    def test_pico_self_test_respects_the_raw_action_guard(self):
        big = [30.0] * ACTION_COUNT
        session = fake_session(
            "pico", contract_metadata("pico", self_test_actions=[big] * 8), outputs=[big]
        )
        contract = pc.parse_policy("pico", session)
        with self.assertRaisesRegex(pc.PolicySelfTestError, "guard"):
            pc.run_self_test(session, contract)


class ManifestTest(unittest.TestCase):
    def manifest(self, **changes):
        manifest = {
            "contract": pc.POLICY_CONTRACT,
            "home_tag": HOME_TAG,
            "training_commit": "0" * 40,
            "dry_run": False,
            "policies": {
                kind: {"file": pc.POLICY_FILES[kind], "sha256": "a" * 64, "checkpoint_sha256": parse(kind).checkpoint_sha256}
                for kind in pc.KINDS
            },
        }
        manifest.update(changes)
        return manifest

    def load(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / pc.MANIFEST_NAME).write_text(json.dumps(manifest))
            return pc.load_manifest(directory)

    def test_manifest_is_checked(self):
        self.load(self.manifest())
        base = self.manifest()
        cases = {
            "contract": self.manifest(contract="microban-policy-0"),
            "home": self.manifest(home_tag="other_home"),
            "dry_run_type": self.manifest(dry_run="no"),
            "dry_run": self.manifest(dry_run=True),
            "kinds": self.manifest(policies={"walk": base["policies"]["walk"]}),
            "file": self.manifest(policies={**base["policies"], "pico": {**base["policies"]["pico"], "file": "pico.onnx"}}),
            "sha": self.manifest(policies={**base["policies"], "walk": {**base["policies"]["walk"], "sha256": "z"}}),
        }
        for case, manifest in cases.items():
            with self.subTest(case=case), self.assertRaises(pc.PolicyContractError):
                with mock.patch.dict(os.environ, {pc.DRY_RUN_POLICY_ALLOW_ENV: "0"}):
                    self.load(manifest)
        with self.assertRaises(pc.PolicyContractError):
            pc.load_manifest(tempfile.gettempdir() + "/no-such-agents-dir")

    def test_installed_file_checkpoint_and_walker_must_match(self):
        manifest = self.manifest()
        pico = parse("pico")
        pc.check_manifest(pico, Path("pico_teleop.onnx"), manifest, "a" * 64)
        self.assertEqual(pico.pico.walk_checkpoint_sha256, WALK_CHECKPOINT_SHA256)
        other_walker = json.loads(json.dumps(manifest))
        other_walker["policies"]["walk"]["checkpoint_sha256"] = "9" * 64
        other_checkpoint = json.loads(json.dumps(manifest))
        other_checkpoint["policies"]["pico"]["checkpoint_sha256"] = "9" * 64
        for case, (contract, path, document, digest) in {
            "file_sha": (pico, "pico_teleop.onnx", manifest, "b" * 64),
            "file_name": (pico, "other.onnx", manifest, "a" * 64),
            "checkpoint": (pico, "pico_teleop.onnx", other_checkpoint, "a" * 64),
            "walker": (pico, "pico_teleop.onnx", other_walker, "a" * 64),
            "dry_run": (pico, "pico_teleop.onnx", {**manifest, "dry_run": True}, "a" * 64),
        }.items():
            with self.subTest(case=case), self.assertRaises(pc.PolicyContractError):
                pc.check_manifest(contract, Path(path), document, digest)


PACKAGE_PROBE = r"""
import json, sys
from pathlib import Path
import numpy as np
import policy_contract as pc
from constants import NEUTRAL_POSE, MOTOR_TO_ID, HOME_PROJECTED_GRAVITY
from input.input_source import UserInput
from moves.getup import GetupMove
from moves.move import MotorCommand
from moves.pico_hybrid import PicoHybridMove
from moves.walk import WalkMove
from observer import Observation, RobotState

agents = Path(sys.argv[1])
result = {}
try:
    walk = WalkMove(controller=None, policy_path=agents / "walk.onnx")
    getup = GetupMove(controller=None, policy_path=agents / "getup.onnx")
    pico = PicoHybridMove(controller=None, policy_path=agents / "pico_teleop.onnx")
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
    sys.exit(0)
obs = Observation(
    robot_state=RobotState(
        time_s=0.0, gyro=[0.0, 0.0, 0.0], projected_gravity=list(HOME_PROJECTED_GRAVITY),
        body_quat=[1.0, 0.0, 0.0, 0.0], motor_positions=dict(NEUTRAL_POSE),
        motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
    ),
    user_input=UserInput(active_moves={"walk", "getup"}, torque_enabled=True, getup_armed=True),
)
for name, move in (("walk", walk), ("getup", getup), ("pico", pico)):
    command = MotorCommand()
    move.on_start(obs, command)
    move.step(obs, command)
    result[name] = all(np.isfinite(list(command.target_angles.values())))
result["getup_ready"] = getup.model_ready
result["pico_self_test_rows"] = pico.self_test_rows
print(json.dumps(result))
"""


def run_in_home(yaml_path, args, *, dry_run_allowed=True, script=None):
    environment = dict(os.environ)
    environment["MICROBAN_HOME_POSE_YAML"] = str(yaml_path)
    environment[pc.DRY_RUN_POLICY_ALLOW_ENV] = "1" if dry_run_allowed else "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    command = [sys.executable, "-c", script, *args] if script else [sys.executable, *args]
    return subprocess.run(
        command, env=environment, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False
    )


class DryRunPackageTest(unittest.TestCase):
    """The 64-env dry-run package of the training pipeline (forward-lean HOME)."""

    def validate(self, agents, yaml_path=LEAN_HOME_YAML, **kwargs):
        completed = run_in_home(
            yaml_path, [str(REPO_ROOT / "tools" / "validate_policies.py"), str(agents)], **kwargs
        )
        return completed.returncode, json.loads(completed.stdout)

    def test_validator_accepts_the_package_at_its_home(self):
        code, report = self.validate(POLICY_PACKAGE)
        self.assertEqual(code, 0, report)
        self.assertEqual(report["contract"], "microban-policy-1")
        self.assertEqual(report["home_tag"], "forward_lean_home")
        self.assertEqual(set(report["policies"]), {"walk", "getup", "pico"})
        for entry in report["policies"].values():
            self.assertGreaterEqual(entry["self_test_rows"], 8)

    def test_moves_load_and_step_the_package(self):
        completed = run_in_home(LEAN_HOME_YAML, [str(POLICY_PACKAGE)], script=PACKAGE_PROBE)
        self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(
            result, {"walk": True, "getup": True, "pico": True, "getup_ready": True, "pico_self_test_rows": 16}
        )

    def test_package_is_refused_at_another_home_or_outside_the_dry_run(self):
        code, report = self.validate(POLICY_PACKAGE, CENTERED_HOME_YAML)
        self.assertEqual(code, 1)
        self.assertIn("home_tag", report["error"])
        code, report = self.validate(POLICY_PACKAGE, dry_run_allowed=False)
        self.assertEqual(code, 1)
        self.assertIn("DRY RUN", report["error"])

    def test_tampered_or_mixed_packages_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = Path(directory)

            def fresh():
                for path in POLICY_PACKAGE.iterdir():
                    shutil.copy(path, agents / path.name)

            fresh()
            data = bytearray((agents / "getup.onnx").read_bytes())
            data[-1] ^= 1
            (agents / "getup.onnx").write_bytes(bytes(data))
            code, report = self.validate(agents)
            self.assertEqual(code, 1)
            self.assertIn("getup.onnx", report["error"])

            fresh()
            manifest = json.loads((agents / "manifest.json").read_text())
            manifest["policies"]["walk"]["checkpoint_sha256"] = "9" * 64
            (agents / "manifest.json").write_text(json.dumps(manifest))
            code, report = self.validate(agents)
            self.assertEqual(code, 1)
            self.assertIn("checkpoint", report["error"])

            fresh()
            (agents / "manifest.json").unlink()
            code, report = self.validate(agents)
            self.assertEqual(code, 1)
            self.assertIn("manifest", report["error"])


if __name__ == "__main__":
    unittest.main()
