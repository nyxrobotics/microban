"""The crouch offsets of the PICO policy (src/crouch_feedforward.py, the
training's file) and their use in the PICO runtime."""

import math
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import crouch_feedforward as cf
from constants import NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from home_pose import HOME_TRUNK_PITCH_RAD
from moves.move import MotorCommand
from moves.pico_hybrid import PicoHybridMove
from policy_contract import crouch_table
from policy_fixtures import CROUCH_PROBES
from test_pico_hybrid import observation, pico_session

ROBOT_XML = Path(__file__).resolve().parents[1] / "src" / "model" / "mjcf" / "robot.xml"
LEG_BODIES = {
    "right": ("hip", "hip_block__configuration_right", "femur__configuration_right",
              "tibia__configuration_right", "ankle_block__configuration_left", "foot"),
    "left": ("hip_2", "hip_block__configuration_left", "femur__configuration_left",
             "tibia__configuration_left", "ankle_block__configuration_right", "foot_2"),
}
HOME_LEGS = np.array([NEUTRAL_POSE[name] for name in cf.LEG_JOINT_NAMES])


def _numbers(text: str) -> np.ndarray:
    return np.array([float(value) for value in text.split()])


def _unit(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.linalg.norm(quat)


class TrainingNumbersTest(unittest.TestCase):
    def test_the_robot_reproduces_the_training_offsets(self):
        offsets = cf.leg_offsets(crouch_table(), np.asarray(CROUCH_PROBES["foot_targets"]))
        np.testing.assert_allclose(offsets, CROUCH_PROBES["leg_offsets_rad"], rtol=0.0, atol=1e-9)
        self.assertEqual(CROUCH_PROBES["contract"], cf.CONTRACT)
        self.assertEqual(CROUCH_PROBES["joint_names"], list(cf.LEG_JOINT_NAMES))

    def test_the_leg_chains_are_robot_xml(self):
        bodies = {body.get("name"): body for body in ET.parse(ROBOT_XML).iter("body")}
        for side, names in LEG_BODIES.items():
            for name, (pos, quat) in zip(names, cf._CHAINS[side], strict=True):
                body = bodies[name]
                np.testing.assert_allclose(pos, _numbers(body.get("pos")), atol=1e-12, err_msg=name)
                np.testing.assert_allclose(_unit(quat), _unit(_numbers(body.get("quat"))), atol=1e-12)
                joint = body.find("joint")
                self.assertTrue(joint.get("name").startswith(side) and joint.get("axis") == "0 0 1")
            site = bodies[names[-1]].find("site")
            self.assertEqual(site.get("name"), f"{side}_foot")
            pos, quat = cf._SITES[side]
            np.testing.assert_allclose(pos, _numbers(site.get("pos")), atol=1e-12)
            np.testing.assert_allclose(_unit(quat), _unit(_numbers(site.get("quat"))), atol=1e-12)

    def test_both_soles_stay_while_the_trunk_goes_down(self):
        c, s = math.cos(HOME_TRUNK_PITCH_RAD), math.sin(HOME_TRUNK_PITCH_RAD)
        levelled_to_trunk = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
        rng = np.random.default_rng(0)
        d = np.column_stack((rng.uniform(-0.01, 0.01, (40, 2)), rng.uniform(0.0026, 0.02, 40)))
        offsets = cf.leg_offsets(crouch_table(), np.hstack((d, d)))
        for index, side in enumerate(("right", "left")):
            legs = slice(6 * index, 6 * index + 6)
            home_pos, home_rot, _, _ = cf.leg_fk(side, HOME_LEGS[None, legs])
            pos, rot, _, _ = cf.leg_fk(side, HOME_LEGS[legs] + offsets[:, legs])
            # Relative to the trunk the soles rise by (dx, dy, dz): in the world
            # they stay while the trunk moves by -(dx, dy, dz).
            np.testing.assert_allclose(pos, home_pos + d @ levelled_to_trunk.T, atol=5e-5)
            np.testing.assert_allclose(rot, np.broadcast_to(home_rot, rot.shape), atol=2e-3)

    def test_zero_without_a_two_foot_target(self):
        feet = np.array([
            [0.0] * 6,
            [0.01, 0.0, 0.03, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.01, 0.05],
            [0.0, 0.0, 0.016, 0.0, 0.0, 0.0],
        ])
        np.testing.assert_array_equal(cf.leg_offsets(crouch_table(), feet), np.zeros((4, 12)))


class PicoCrouchTest(unittest.TestCase):
    def test_the_legs_work_around_the_crouch_pose(self):
        raw = np.linspace(-0.2, 0.2, 18, dtype=np.float32).reshape(1, 18)
        move = PicoHybridMove(session=pico_session(raw), gyro_transform=lambda value: value)
        obs = observation()
        obs.user_input.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
        obs.user_input.foot_target = {"left": (0.004, 0.004, 0.012), "right": (0.004, 0.004, 0.012)}
        crouch = dict(zip(cf.LEG_JOINT_NAMES, cf.leg_offsets(
            crouch_table(), np.array([[0.004, 0.004, 0.012] * 2]))[0], strict=True))
        self.assertGreater(crouch["left_knee"], 0.4)
        for name in crouch:
            obs.robot_state.motor_positions[name] = NEUTRAL_POSE[name] + crouch[name] + 0.01
        values = move.build_observation(obs)
        joints = move.contract.observation_joint_names
        for index, name in enumerate(joints):
            expected = 0.01 if name in crouch else 0.0
            self.assertAlmostEqual(values[6 + index], expected, places=12, msg=name)
        command = MotorCommand()
        move.on_start(obs, command)
        move.step(obs, command)
        for index, name in enumerate(OBSERVATION_DOF_ORDER):
            if name in crouch:
                expected = NEUTRAL_POSE[name] + crouch[name] + float(raw[0, index])
                self.assertAlmostEqual(command.target_angles[name], expected, places=12, msg=name)
        # The previous action is the raw output.
        np.testing.assert_array_equal(np.asarray(move.build_observation(obs)[48:66], dtype=np.float32), raw[0])

    def test_one_foot_up_keeps_home(self):
        move = PicoHybridMove(session=pico_session(), gyro_transform=lambda value: value)
        values = move.build_observation(observation())  # the left foot up, the right at zero
        self.assertEqual(values[6:27], [0.0] * 21)


if __name__ == "__main__":
    unittest.main()
