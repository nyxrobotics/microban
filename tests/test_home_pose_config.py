"""config/home_pose.yaml is the robot's only HOME source and reproduces today's HOME."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import constants
import home_pose
from moves import pico_hybrid, walk
from moves.walk import WalkPolicyContractError
from policy_fixtures import OLD_HOME, home_pose_stamp, walk_contract_metadata
from test_walk_contract import fake_walk

CONFIG_TEXT = home_pose.HOME_POSE_PATH.read_text(encoding="utf-8")

# The centered HOME every installed policy was trained at (2026-10-05).
CENTERED_CONTRACTS = {
    "walk_contract_version": "v3_centered_home_servo_range",
    "getup_contract_version": "v5",
    # The centered v5 exporter also published runs stamped "v4": not checked.
    "getup_checkpoint_stamp": "",
    "v12_target_frame": "robot_trunk_xyz_forward_left_up",
    "v12_home_pose_revision": (
        "centered_home_hip_plus1p198384259489_ankle_minus1p198384259489_shoulder_zero_v5"
    ),
    "v12_recipe_revision": (
        "centered_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
        "raw_prev_action_servo_range_pi_v11"
    ),
    "v12_hand_pose_release_recipe_revision": (
        "centered_home_velocity_source_staged_mask_reachable_fk_elbow_minus10_"
        "raw_prev_action_servo_range_pi_active_hand_arm_pose_release_v12"
    ),
    "v12_packager_revision": (
        "microban_teleop_v12_final_deployment_packager_v6_centered_home_servo_range"
    ),
}


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
        self.assertEqual(constants.HOME_PROJECTED_GRAVITY, loaded["projected_gravity"])
        self.assertEqual(constants.HOME_PITCH_RAD, loaded["joint_pos_rad"]["left_hip_pitch"])

    def test_centered_home_values_are_unchanged(self):
        self.assertEqual(home_pose.HOME_TAG, "centered_home")
        self.assertEqual(home_pose.HOME_ROOT_POS_M, (0.0, 0.0, 0.170554885633559))
        self.assertEqual(home_pose.HOME_ROOT_QUAT_WXYZ, (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(home_pose.HOME_PROJECTED_GRAVITY, (0.0, 0.0, -1.0))
        self.assertEqual(home_pose.HOME_TRUNK_PITCH_RAD, 0.0)
        self.assertEqual(dict(home_pose.HOME_CONTRACTS), CENTERED_CONTRACTS)
        self.assertEqual(walk.WALK_CONTRACT_VERSION, CENTERED_CONTRACTS["walk_contract_version"])
        self.assertEqual(
            pico_hybrid.EXPECTED_V12_HOME_POSE_REVISION,
            CENTERED_CONTRACTS["v12_home_pose_revision"],
        )
        self.assertEqual(
            pico_hybrid.EXPECTED_V12_RECIPE_REVISIONS,
            frozenset(
                (
                    CENTERED_CONTRACTS["v12_recipe_revision"],
                    CENTERED_CONTRACTS["v12_hand_pose_release_recipe_revision"],
                )
            ),
        )
        fk = pico_hybrid.EXPECTED_V12_HAND_TARGET_FK
        self.assertEqual(fk["revision"], "microban_robot_xml_arm_fk_reachable_box_elbow_upper_minus10_v2")
        self.assertEqual(fk["home_joint_deg"], [[0.0, 10.0, -20.0], [0.0, -10.0, -20.0]])
        self.assertEqual(fk["normalizer_abs_bound_m"], [0.063, 0.0388, 0.0605])
        self.assertEqual(
            fk["offset_aabb_max_m"][0],
            [0.06289464331528255, 0.0387512193701912, 0.060477220857479266],
        )

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
            "bad_value": CONFIG_TEXT.replace('tag: "centered_home"', "tag: centered_home", 1),
            "rad_not_deg": CONFIG_TEXT.replace(
                "  left_knee: 0.0\n  right_knee: 0.0\n  left_ankle_roll: -0.0872",
                "  left_knee: 0.001\n  right_knee: 0.0\n  left_ankle_roll: -0.0872",
                1,
            ),
            "pitched_trunk": CONFIG_TEXT.replace(
                "trunk_pitch_deg: 0.0\ntrunk_pitch_rad: 0.0",
                "trunk_pitch_deg: 10.0\ntrunk_pitch_rad: 0.17453292519943295",
                1,
            ),
            "gravity": CONFIG_TEXT.replace(
                "projected_gravity: [0.0, 0.0, -1.0]", "projected_gravity: [0.0, 0.0, 1.0]", 1
            ),
            "missing_contract": CONFIG_TEXT.replace("  walk_contract_version:", "  walk_contract:", 1),
            "schema": CONFIG_TEXT.replace("schema_version: 1", "schema_version: 2", 1),
        }
        for case, text in cases.items():
            with self.subTest(case=case):
                self.assertNotEqual(text, CONFIG_TEXT)
                with self.assertRaises(home_pose.HomePoseConfigError):
                    load_variant(text)
        self.assertEqual(load_variant(CONFIG_TEXT)["joint_pos_rad"], home_pose.HOME_JOINT_POS_RAD)

    def test_home_pose_stamp_must_be_this_home(self):
        self.assertTrue(home_pose.home_pose_stamp_matches(home_pose_stamp()))
        self.assertFalse(home_pose.home_pose_stamp_matches(home_pose_stamp(OLD_HOME)))
        self.assertFalse(home_pose.home_pose_stamp_matches(None))
        self.assertFalse(home_pose.home_pose_stamp_matches("{"))
        raised = json.loads(home_pose_stamp())
        raised["root_pos_m"][2] += 0.005
        self.assertFalse(home_pose.home_pose_stamp_matches(json.dumps(raised)))
        tilted = json.loads(home_pose_stamp())
        tilted["root_quat_wxyz"] = [math.cos(0.05), 0.0, math.sin(0.05), 0.0]
        self.assertFalse(home_pose.home_pose_stamp_matches(json.dumps(tilted)))

    def test_walk_policy_needs_the_home_stamp(self):
        fake_walk()  # the deployed contract loads
        missing = walk_contract_metadata()
        del missing["home_pose"]
        other_home = walk_contract_metadata()
        other_home["home_pose"] = home_pose_stamp(OLD_HOME)
        for case, metadata in (("missing", missing), ("other_home", other_home)):
            with self.subTest(case=case), self.assertRaises(WalkPolicyContractError):
                fake_walk(metadata)


if __name__ == "__main__":
    unittest.main()
