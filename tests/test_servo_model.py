"""The servo model: the simulator loads the XC330-T288-T identification that training loads (mjlab_microban's
src/mjlab_microban/robot/xc330_params.json, the same bytes), and the current proxy and the self-test's speed bound
are its kt and R."""

import hashlib
import json
import math
import unittest
from pathlib import Path

from constants import PROXY_KT, PROXY_R
from policy_contract import SELF_TEST_MAX_JOINT_SPEED_RAD_S

PARAMS = Path(__file__).resolve().parents[1] / "tools" / "actuator_id" / "xc330_params.json"
PARAMS_SHA256 = "d4206f02f31ac817804ae0785fcb94849388ce3f42e8987195b7e179683468ce"


class ServoModelTest(unittest.TestCase):
    def test_same_identification_as_training(self) -> None:
        self.assertEqual(hashlib.sha256(PARAMS.read_bytes()).hexdigest(), PARAMS_SHA256)

    def test_values_follow_the_identification(self) -> None:
        params = json.loads(PARAMS.read_text())
        self.assertAlmostEqual(PROXY_KT, params["kt"], delta=5e-4)
        self.assertAlmostEqual(PROXY_R, params["R"], delta=5e-4)
        # No-load speed at a full 3S pack (12.6 V), rounded up to 0.1 rad/s.
        self.assertEqual(SELF_TEST_MAX_JOINT_SPEED_RAD_S, math.ceil(10 * 12.6 / params["kt"]) / 10)


if __name__ == "__main__":
    unittest.main()
