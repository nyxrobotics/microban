"""config/home_pose.yaml is the robot's only HOME source."""

import math
import re
import tempfile
import unittest
from pathlib import Path

import constants
import home_pose

CONFIG_TEXT = home_pose.HOME_POSE_PATH.read_text(encoding="utf-8")

def _edit(pattern, replace, text=None):
    """CONFIG_TEXT with the first match of ``pattern`` rewritten by ``replace(match)``."""

    text = CONFIG_TEXT if text is None else text
    return re.sub(pattern, replace, text, count=1, flags=re.MULTILINE)


def _trunk_pitched_by_10_deg(match):
    degrees = float(match.group(1)) + 10.0
    return f"trunk_pitch_deg: {degrees!r}\ntrunk_pitch_rad: {math.radians(degrees)!r}"


def load_variant(text):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "home_pose.yaml"
        path.write_text(text, encoding="utf-8")
        return home_pose.load_home_pose(path)


class HomePoseConfigTest(unittest.TestCase):
    def test_constants_come_from_the_config(self):
        loaded = home_pose.load_home_pose()
        self.assertEqual(
            constants.NEUTRAL_POSE, {name: loaded["joint_pos_rad"][name] for name in constants.MOTOR_TO_ID}
        )
        self.assertEqual(list(constants.NEUTRAL_POSE), list(constants.MOTOR_TO_ID))
        self.assertEqual(constants.HOME_ROOT_POS_Z_M, loaded["root_pos_m"][2])
        self.assertEqual(constants.HOME_ROOT_QUAT_WXYZ, loaded["root_quat_wxyz"])

    def test_parser_matches_a_full_yaml_parser(self):
        try:
            import yaml
        except ImportError:  # pragma: no cover - PyYAML is optional on the robot
            self.skipTest("PyYAML not installed")
        self.assertEqual(home_pose.parse_home_pose_yaml(CONFIG_TEXT), yaml.safe_load(CONFIG_TEXT))

    def test_malformed_or_inconsistent_configs_are_refused(self):
        cases = {
            "tab": CONFIG_TEXT.replace("\n  head: 0.0", "\n\thead: 0.0", 1),
            "indent": CONFIG_TEXT.replace("\n  head: 0.0", "\n   head: 0.0", 1),
            "duplicate": CONFIG_TEXT.replace("\n  head: 0.0", "\n  head: 0.0\n  head: 0.0", 1),
            "bad_value": CONFIG_TEXT.replace(
                f'tag: "{home_pose.HOME_TAG}"', f"tag: {home_pose.HOME_TAG}", 1
            ),
            # The radian table's left knee 0.001 rad off its degree value.
            "rad_not_deg": _edit(
                r"^(joint_pos_rad:\n(?:  .*\n)*?  left_knee: )(\S+)$",
                lambda m: f"{m.group(1)}{float(m.group(2)) + 0.001!r}",
            ),
            # A consistent trunk pitch the root quaternion / gravity do not have.
            "pitched_trunk": _edit(
                r"^trunk_pitch_deg: (\S+)\ntrunk_pitch_rad: \S+$", _trunk_pitched_by_10_deg
            ),
            "gravity": _edit(r"^projected_gravity: .*$", lambda m: "projected_gravity: [0.0, 0.0, 1.0]"),
            "schema": CONFIG_TEXT.replace("schema_version: 2", "schema_version: 1", 1),
        }
        for case, text in cases.items():
            with self.subTest(case=case):
                self.assertNotEqual(text, CONFIG_TEXT)
                with self.assertRaises(home_pose.HomePoseConfigError):
                    load_variant(text)
        self.assertEqual(load_variant(CONFIG_TEXT)["joint_pos_rad"], home_pose.HOME_JOINT_POS_RAD)



if __name__ == "__main__":
    unittest.main()
