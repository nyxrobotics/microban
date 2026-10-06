import math
import unittest

from constants import VTHETA_MAX_MOVING, VTHETA_MAX_STATIONARY, VX_MAX, VX_MAX_BACKWARD, VY_MAX
from input.input_source import scale_velocity


class ScaleVelocityTest(unittest.TestCase):
    def test_in_range_commands_map_to_the_physical_limits(self):
        self.assertEqual(
            scale_velocity({"vx": 1.0, "vy": -1.0, "vtheta": 0.5}),
            {"vx": VX_MAX, "vy": -VY_MAX, "vtheta": 0.5 * VTHETA_MAX_MOVING},
        )
        self.assertEqual(
            scale_velocity({"vx": -0.5, "vy": 0.0, "vtheta": 0.0}),
            {"vx": -0.5 * VX_MAX_BACKWARD, "vy": 0.0, "vtheta": 0.0},
        )
        self.assertEqual(
            scale_velocity({"vx": 0.0, "vy": 0.0, "vtheta": -1.0}),
            {"vx": 0.0, "vy": 0.0, "vtheta": -VTHETA_MAX_STATIONARY},
        )

    def test_out_of_range_commands_keep_their_ratio(self):
        # Clipping vx alone would turn (2, 1, 1) into (1, 1, 1): a different
        # walking direction.  The whole command is scaled by 1/2 instead.
        scaled = scale_velocity({"vx": 2.0, "vy": 1.0, "vtheta": 1.0})
        self.assertEqual(
            scaled,
            {"vx": VX_MAX, "vy": 0.5 * VY_MAX, "vtheta": 0.5 * VTHETA_MAX_MOVING},
        )
        normalized = [
            scaled["vx"] / VX_MAX,
            scaled["vy"] / VY_MAX,
            scaled["vtheta"] / VTHETA_MAX_MOVING,
        ]
        self.assertEqual(normalized, [1.0, 0.5, 0.5])
        mirrored = scale_velocity({"vx": 2.0, "vy": -1.0, "vtheta": -1.0})
        self.assertEqual(
            mirrored,
            {"vx": VX_MAX, "vy": -0.5 * VY_MAX, "vtheta": -0.5 * VTHETA_MAX_MOVING},
        )
        backward = scale_velocity({"vx": -1.5, "vy": 0.75, "vtheta": 0.0})
        self.assertAlmostEqual(backward["vx"], -VX_MAX_BACKWARD)
        self.assertAlmostEqual(backward["vy"], 0.5 * VY_MAX)

    def test_non_finite_axes_are_zero(self):
        self.assertEqual(
            scale_velocity({"vx": math.nan, "vy": 2.0, "vtheta": math.inf}),
            {"vx": 0.0, "vy": VY_MAX, "vtheta": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
