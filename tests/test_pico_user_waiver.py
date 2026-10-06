"""The provisional user waiver admits exactly its one recorded package.

TEMPORARY: delete with the V12 USER WAIVER block of src/moves/pico_hybrid.py
when the twist-ratio replacement model is installed.
"""

import hashlib
import json
import unittest
from pathlib import Path

import onnxruntime as ort

from moves.pico_hybrid import (
    EXPECTED_V12_COMPLETION_ALLOWANCE_TRACKING_PROFILE,
    EXPECTED_V12_POSE_RELEASE_RECIPE_REVISION,
    EXPECTED_V12_RECIPE_REVISION,
    EXPECTED_V12_USER_WAIVER_BINDINGS,
    EXPECTED_V12_USER_WAIVER_RECORD,
    EXPECTED_V12_USER_WAIVER_REVISION,
    EXPECTED_V12_USER_WAIVER_TRACKING_PROFILE,
    PicoHybridMove,
    PicoHybridPolicyContractError,
    v12_user_waiver_summary,
)
from test_pico_hybrid import FakeSession, valid_v12_metadata

INSTALLED_POLICY = Path(__file__).resolve().parents[1] / "src/agents/pico_teleop.onnx"


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _recorded_parity(metadata):
    return {
        "semantics": (
            "torch_vs_onnxruntime_cpu_and_reference_on_final_tracking_runtime_"
            "smoke_observations_normwise_rule_v1"
        ),
        "corpus_sha256": metadata["v12_runtime_smoke_observations_sha256"],
        "samples": len(json.loads(metadata["v12_runtime_smoke_observations_json"])),
        "rule": "max_abs_error_le_atol_plus_rtol_times_max_abs_expected_per_sample_v1",
        "atol": 2e-05,
        "rtol": 1e-06,
        "maximum_absolute_expected_output": 3.041567802429199,
        "reference_evaluator_maximum_absolute_error": 4.76837158203125e-07,
        "onnxruntime_cpu_maximum_absolute_error": 9.5367431640625e-07,
        "reference_evaluator_maximum_bound_ratio": 0.02069464884698391,
        "onnxruntime_cpu_maximum_bound_ratio": 0.04262354224920273,
        "status": "pass",
    }


def waiver_metadata(record=None, parity=None):
    metadata = valid_v12_metadata()
    metadata["microban_teleop_recipe_revision"] = (
        EXPECTED_V12_POSE_RELEASE_RECIPE_REVISION
    )
    metadata.update(EXPECTED_V12_USER_WAIVER_BINDINGS)
    metadata.update(
        {
            "v12_onnx_parity_relative_tolerance": "1e-06",
            "v12_onnx_reference_max_bound_ratio": "0.2558188736438751",
            "v12_onnx_reference_max_abs_error": "3.0517578125e-05",
            "v12_onnxruntime_cpu_max_abs_error": "9.1552734375e-05",
        }
    )
    record = EXPECTED_V12_USER_WAIVER_RECORD if record is None else record
    metadata.update(
        {
            "v12_provisional_install": "true",
            "v12_user_waiver_revision": EXPECTED_V12_USER_WAIVER_REVISION,
            "v12_user_waiver_json": json.dumps(record, separators=(",", ":")),
            "v12_user_waiver_sha256": hashlib.sha256(
                _canonical(record).encode("utf-8")
            ).hexdigest(),
            "v12_user_waiver_recorded_corpus_parity_json": json.dumps(
                _recorded_parity(metadata) if parity is None else parity,
                separators=(",", ":"),
            ),
        }
    )
    return metadata


class PicoUserWaiverTest(unittest.TestCase):
    def assert_refused(self, metadata):
        session = FakeSession(metadata=metadata)
        with self.assertRaises(PicoHybridPolicyContractError):
            PicoHybridMove(session=session)
        self.assertEqual(session.run_count, 0)

    def test_the_recorded_package_is_admitted_as_provisional(self):
        metadata = waiver_metadata()
        session = FakeSession(metadata=metadata)
        move = PicoHybridMove(session=session)
        self.assertEqual(
            move._contract.v12_user_waiver_revision, EXPECTED_V12_USER_WAIVER_REVISION
        )
        # The startup self-test on the recorded corpus still runs.
        self.assertEqual(
            session.run_count,
            len(json.loads(metadata["v12_runtime_smoke_observations_json"])),
        )
        summary = v12_user_waiver_summary(move._contract.v12_user_waiver_revision)
        self.assertEqual(summary["stage_gate_status"], "provisional_user_waiver")
        self.assertEqual(len(summary["waived"]), 3)
        self.assertIsNone(v12_user_waiver_summary(None))

    def test_record_names_exactly_the_three_waived_items_and_decisions(self):
        record = EXPECTED_V12_USER_WAIVER_RECORD
        self.assertEqual(
            EXPECTED_V12_USER_WAIVER_TRACKING_PROFILE,
            f"{EXPECTED_V12_COMPLETION_ALLOWANCE_TRACKING_PROFILE}_user_waiver_"
            "mixed_forward_left_twist_v1",
        )
        twist, ratio, cap = record["waived"]
        self.assertEqual((twist["scenario"], twist["axis"]), ("mixed_forward_left", "vy_m_s"))
        self.assertEqual(twist["measured_signed_response"], -0.016997758105397224)
        self.assertEqual((ratio["sample_index"], ratio["bound_ratio"]), (60, 1.0178567171096802))
        self.assertEqual(cap["maximum_absolute_expected_output"], 219.99908447265625)
        self.assertEqual(
            [item["verbatim"] for item in record["user_decisions"]],
            [
                "苦手な動きが1つ残ったまま実機の前傾版ブランチに入れてよい。"
                "その後直す、が良いと思います",
                "「A（許容して入れる）」",
            ],
        )
        self.assertFalse(record["stage_gate_pass"])

    def test_any_other_checkpoint_gate_report_or_value_is_refused(self):
        for name in EXPECTED_V12_USER_WAIVER_BINDINGS:
            with self.subTest(binding=name):
                metadata = waiver_metadata()
                value = metadata[name]
                if len(value) == 64:
                    metadata[name] = "0" * 64
                elif name == "v12_onnxruntime_cpu_max_bound_ratio":
                    metadata[name] = "1.02"
                elif name == "v12_onnx_parity_max_abs_expected_output":
                    metadata[name] = "230.0"
                else:
                    metadata[name] = value + "_x"
                self.assert_refused(metadata)

    def test_the_status_and_profile_need_the_record_and_vice_versa(self):
        for name in ("v12_stage_gate_status", "v12_tracking_profile"):
            with self.subTest(only=name):
                metadata = valid_v12_metadata()
                metadata["microban_teleop_recipe_revision"] = (
                    EXPECTED_V12_POSE_RELEASE_RECIPE_REVISION
                )
                metadata[name] = EXPECTED_V12_USER_WAIVER_BINDINGS[name]
                self.assert_refused(metadata)
        metadata = waiver_metadata()
        metadata["v12_stage_gate_status"] = "pass"
        self.assert_refused(metadata)
        metadata = waiver_metadata()
        metadata["v12_tracking_profile"] = (
            EXPECTED_V12_COMPLETION_ALLOWANCE_TRACKING_PROFILE
        )
        self.assert_refused(metadata)
        metadata = waiver_metadata()
        metadata["microban_teleop_recipe_revision"] = EXPECTED_V12_RECIPE_REVISION
        self.assert_refused(metadata)

    def test_partial_extra_or_drifted_waiver_metadata_is_refused(self):
        cases = {
            "missing": lambda m: m.pop("v12_user_waiver_sha256"),
            "extra": lambda m: m.update({"v12_user_waiver_note": "x"}),
            "provisional_flag": lambda m: m.update({"v12_provisional_install": "false"}),
            "revision": lambda m: m.update({"v12_user_waiver_revision": "v2"}),
            "record_sha": lambda m: m.update({"v12_user_waiver_sha256": "0" * 64}),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                metadata = waiver_metadata()
                mutate(metadata)
                self.assert_refused(metadata)

    def test_a_resigned_record_naming_anything_else_is_refused(self):
        def drift(path, value):
            record = json.loads(json.dumps(EXPECTED_V12_USER_WAIVER_RECORD))
            target = record
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return record

        for path, value in (
            (("waived", 0, "scenario"), "mixed_backward_right"),
            (("waived", 0, "axis"), "vx_m_s"),
            (("waived", 0, "check"), "no_falls"),
            (("waived", 0, "measured_signed_response"), -0.05),
            (("waived", 1, "sample_index"), 59),
            (("waived", 1, "bound_ratio"), 1.05),
            (("waived", 2, "maximum_absolute_expected_output"), 300.0),
            (("checkpoint", "sha256"), "f" * 64),
            (("user_decisions", 1, "verbatim"), "A"),
            (("stage_gate_pass",), True),
            (("checkpoint", "iteration"), 14999.0),
        ):
            with self.subTest(path=path):
                self.assert_refused(waiver_metadata(record=drift(path, value)))

    def test_the_recorded_corpus_parity_must_pass_the_normal_rule(self):
        for name, value in (
            ("onnxruntime_cpu_maximum_bound_ratio", 1.01),
            ("reference_evaluator_maximum_bound_ratio", 1.5),
            ("onnxruntime_cpu_maximum_absolute_error", 1.0e-3),
            ("maximum_absolute_expected_output", 250.0),
            ("corpus_sha256", "0" * 64),
            ("samples", 15),
            ("status", "fail"),
            ("rule", "elementwise_v0"),
            ("atol", 1.0e-4),
            ("extra", 1),
        ):
            with self.subTest(field=name):
                metadata = waiver_metadata()
                parity = _recorded_parity(metadata)
                parity[name] = value
                self.assert_refused(waiver_metadata(parity=parity))
        metadata = waiver_metadata()
        parity = _recorded_parity(metadata)
        parity.pop("status")
        self.assert_refused(waiver_metadata(parity=parity))

    def test_waived_values_without_the_waiver_stay_refused(self):
        metadata = valid_v12_metadata()
        metadata["microban_teleop_recipe_revision"] = (
            EXPECTED_V12_POSE_RELEASE_RECIPE_REVISION
        )
        metadata["v12_tracking_profile"] = (
            EXPECTED_V12_COMPLETION_ALLOWANCE_TRACKING_PROFILE
        )
        metadata.update(
            {
                "v12_onnx_parity_rule": EXPECTED_V12_USER_WAIVER_BINDINGS[
                    "v12_onnx_parity_rule"
                ],
                "v12_onnx_parity_relative_tolerance": "1e-06",
                "v12_onnx_reference_max_bound_ratio": "0.2558188736438751",
                "v12_onnxruntime_cpu_max_bound_ratio": "1.0178567171096802",
                "v12_onnx_parity_max_abs_expected_output": "219.99908447265625",
            }
        )
        self.assert_refused(metadata)

    @unittest.skipUnless(INSTALLED_POLICY.is_file(), "no installed PICO policy")
    def test_installed_policy_is_this_provisional_package(self):
        session = ort.InferenceSession(
            str(INSTALLED_POLICY), providers=["CPUExecutionProvider"]
        )
        metadata = session.get_modelmeta().custom_metadata_map
        if "v12_user_waiver_revision" not in metadata:
            self.skipTest("installed policy is not the provisional package")
        move = PicoHybridMove(session=session)
        self.assertEqual(
            move._contract.v12_user_waiver_revision, EXPECTED_V12_USER_WAIVER_REVISION
        )


if __name__ == "__main__":
    unittest.main()
