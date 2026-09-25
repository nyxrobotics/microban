import tempfile
import unittest
from pathlib import Path

from tools.validate_pico_policy import (
    EXPECTED_WALK_FALLBACK_SHA256,
    WALK_FALLBACK_POLICY,
    WALK_FALLBACK_SMOKE_SAMPLE_COUNT,
    validate_walk_fallback,
)


class WalkFallbackValidatorTest(unittest.TestCase):
    def test_pinned_walk_fallback_passes_fixed_cpu_smoke(self):
        report = validate_walk_fallback()

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["policy"], str(WALK_FALLBACK_POLICY.resolve()))
        self.assertEqual(report["sha256"], EXPECTED_WALK_FALLBACK_SHA256)
        self.assertEqual(report["providers"], ["CPUExecutionProvider"])
        self.assertEqual(
            report["input"],
            {"name": "obs", "shape": [1, 63], "type": "tensor(float)"},
        )
        self.assertEqual(
            report["output"],
            {"name": "actions", "shape": [1, 18], "type": "tensor(float)"},
        )
        smoke = report["smoke"]
        self.assertEqual(smoke["status"], "pass")
        self.assertEqual(smoke["sample_count"], WALK_FALLBACK_SMOKE_SAMPLE_COUNT)
        self.assertTrue(smoke["all_outputs_finite"])
        self.assertGreater(smoke["maximum_absolute_output"], 0.0)

    def test_changed_walk_fallback_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            changed = Path(temp_dir) / "walk.onnx"
            changed.write_bytes(b"not the pinned fallback")

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                validate_walk_fallback(changed)


if __name__ == "__main__":
    unittest.main()
