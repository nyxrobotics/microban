import json
import io
import math
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from constants import MOTOR_TO_ID, NEUTRAL_POSE
from input.network_input import NetworkInputSource
from observer import RobotState
from pico_arm_contract import PICO_ARM_HOME_RAD
from scheduler import Scheduler

BODY_TARGET_CONTRACT = "microban_pico_offsets_v2_both_feet_stationary"
BODY_TARGET_SAFETY_MARGIN = 0.8
COMPLETE_FEET = {
    "left": [0.01, 0.0, 0.02],
    "right": [0.0, 0.0, 0.0],
}
COMPLETE_HANDS = {
    "left": [0.01, -0.02, 0.03],
    "right": [-0.01, 0.02, -0.03],
}
_DEFAULT_TARGET = object()
ARM_HOME = {side: list(values) for side, values in PICO_ARM_HOME_RAD.items()}
ARM_TARGET = {
    "left": [math.radians(12.0), math.radians(18.0), math.radians(-32.0)],
    "right": [math.radians(-12.0), math.radians(-18.0), math.radians(-32.0)],
}


def packet(
    seq,
    moves=None,
    velocity=None,
    session="test",
    policy="walk",
    foot_target=None,
    hand_target=None,
    body_target_contract=None,
    body_target_safety_margin=None,
    arm_tracking_enabled=False,
    arm_joint_target=None,
    torque_enabled=True,
    policy_enabled=True,
    torque_off_requested=False,
):
    return {
        "version": 1,
        "session_id": session,
        "seq": seq,
        "active_moves": moves or [],
        "locomotion_policy": policy,
        "velocity": velocity or {},
        "head_orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "head_yaw_front": False,
        "body_target_contract": body_target_contract,
        "body_target_safety_margin": body_target_safety_margin,
        "foot_target": foot_target,
        "hand_target": hand_target,
        "arm_tracking_enabled": arm_tracking_enabled,
        "arm_joint_target": arm_joint_target,
        "torque_enabled": torque_enabled,
        "policy_enabled": policy_enabled,
        "torque_off_requested": torque_off_requested,
    }


def pico_walk_packet(
    seq,
    velocity=None,
    session="test",
    foot_target=_DEFAULT_TARGET,
    hand_target=_DEFAULT_TARGET,
    body_target_contract=BODY_TARGET_CONTRACT,
    body_target_safety_margin=BODY_TARGET_SAFETY_MARGIN,
):
    if foot_target is _DEFAULT_TARGET:
        foot_target = COMPLETE_FEET
    if hand_target is _DEFAULT_TARGET:
        hand_target = COMPLETE_HANDS
    return packet(
        seq,
        ["walk"],
        velocity,
        session=session,
        policy="pico_teleop",
        foot_target=foot_target,
        hand_target=hand_target,
        body_target_contract=body_target_contract,
        body_target_safety_margin=body_target_safety_margin,
    )


def pico_release_packet(
    seq,
    session="test",
    foot_target=None,
    hand_target=None,
    body_target_contract=BODY_TARGET_CONTRACT,
    body_target_safety_margin=BODY_TARGET_SAFETY_MARGIN,
):
    return packet(
        seq,
        session=session,
        policy="pico_teleop",
        foot_target=foot_target,
        hand_target=hand_target,
        body_target_contract=body_target_contract,
        body_target_safety_margin=body_target_safety_margin,
    )


def arm_pico(source):
    # Repeated release snapshots are harmless and mirror the steady bridge stream.
    source._apply(pico_release_packet(0))
    source._apply(pico_release_packet(1))


def pico_arm_packet(
    seq,
    *,
    enabled,
    target=_DEFAULT_TARGET,
    session="test",
    extra_moves=None,
    foot_target=None,
    hand_target=None,
):
    if target is _DEFAULT_TARGET:
        target = ARM_TARGET if enabled else ARM_HOME
    return packet(
        seq,
        ["pico_arms", *(extra_moves or [])],
        session=session,
        policy="pico_teleop",
        foot_target=foot_target,
        hand_target=hand_target,
        body_target_contract=BODY_TARGET_CONTRACT,
        body_target_safety_margin=BODY_TARGET_SAFETY_MARGIN,
        arm_tracking_enabled=enabled,
        arm_joint_target=target,
    )


class NetworkInputTest(unittest.TestCase):
    def assert_degraded_fallback(self, state, expected_velocity):
        self.assertIn("walk", state.active_moves)
        self.assertEqual(state.velocity, expected_velocity)
        self.assertEqual(state.locomotion_policy, "walk")
        self.assertTrue(state.learned_policy_degraded)
        self.assertIsNone(state.foot_target)
        self.assertIsNone(state.hand_target)

    def test_complete_snapshots_do_not_retain_old_velocity(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))  # released snapshot arms walking
        source._apply(packet(1, ["walk", "hmd_head"], {"vx": 0.8}))
        self.assertEqual(source.read().velocity["vx"], 0.8)
        source._apply(packet(2, ["hmd_head"]))
        state = source.read()
        self.assertEqual(state.velocity, {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        self.assertNotIn("walk", state.active_moves)

    def test_getup_is_not_accepted_via_network_active_moves(self):
        source = NetworkInputSource(stale_after_s=0.5)
        with self.assertRaises(ValueError):
            source._apply(packet(0, ["getup"]))

    def test_hardware_torque_and_policy_states_round_trip(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(
            packet(
                0,
                ["walk", "hmd_head"],
                {"vx": 1.0},
                torque_enabled=False,
                policy_enabled=True,
                torque_off_requested=True,
            )
        )
        state = source.read()
        self.assertFalse(state.torque_enabled)
        self.assertFalse(state.policy_enabled)
        self.assertFalse(state.hold_last_targets)
        self.assertEqual(state.active_moves, set())
        self.assertEqual(state.velocity, {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})

        source._apply(packet(1, torque_enabled=True, policy_enabled=False))
        state = source.read()
        self.assertTrue(state.torque_enabled)
        self.assertFalse(state.policy_enabled)

        source._apply(packet(2, torque_enabled=True, policy_enabled=True))
        state = source.read()
        self.assertTrue(state.torque_enabled)
        self.assertTrue(state.policy_enabled)

    def test_false_torque_snapshot_without_b_holds_previous_goal(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0, torque_enabled=True, policy_enabled=False))
        source._apply(packet(1, torque_enabled=True, policy_enabled=True))
        self.assertTrue(source.read().torque_enabled)

        # The PC can briefly report its own torque state as false while the
        # authenticated link is recovering. Only a separate B event may make
        # the robot's physical torque state false.
        interrupted = packet(
            2,
            ["walk", "hmd_head"],
            {"vx": 1.0},
            torque_enabled=False,
            policy_enabled=False,
        )
        del interrupted["torque_off_requested"]  # old bridge: no explicit B bit
        source._apply(interrupted)
        held = source.read()
        self.assertIsNone(held.torque_enabled)
        self.assertTrue(held.hold_last_targets)
        self.assertEqual(held.active_moves, set())
        self.assertEqual(held.velocity, {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        self.assertIsNone(held.arm_joint_target)

        source._apply(
            packet(
                3,
                torque_enabled=False,
                policy_enabled=False,
                torque_off_requested=True,
            )
        )
        stopped = source.read()
        self.assertIs(stopped.torque_enabled, False)
        self.assertFalse(stopped.hold_last_targets)

    def test_only_explicit_b_can_cut_torque_after_timeout(self):
        source = NetworkInputSource(stale_after_s=0.05)
        source._apply(packet(0, torque_enabled=True, policy_enabled=False))
        source._last_recv_s -= 1.0
        held = source.read()
        self.assertIsNone(held.torque_enabled)
        self.assertTrue(held.hold_last_targets)

        source._apply(
            packet(
                1,
                torque_enabled=False,
                policy_enabled=False,
                torque_off_requested=True,
            )
        )
        stopped = source.read()
        self.assertIs(stopped.torque_enabled, False)
        self.assertFalse(stopped.hold_last_targets)

    def test_non_boolean_torque_or_policy_is_rejected(self):
        source = NetworkInputSource(stale_after_s=0.5)
        good = packet(0, torque_enabled=True, policy_enabled=False)
        source._apply(good)
        bad = dict(packet(1))
        bad["torque_enabled"] = "yes"
        with self.assertRaises(TypeError):
            source._apply(bad)
        bad = dict(packet(2))
        bad["policy_enabled"] = 1
        with self.assertRaises(TypeError):
            source._apply(bad)
        bad = dict(packet(3))
        bad["torque_off_requested"] = "yes"
        with self.assertRaises(TypeError):
            source._apply(bad)
        # The rejected packet must not have overwritten the prior good state.
        state = source.read()
        self.assertTrue(state.torque_enabled)
        self.assertFalse(state.policy_enabled)

    def test_stale_timeout_holds_torque_and_last_goal(self):
        source = NetworkInputSource(stale_after_s=0.05)
        source._apply(packet(0, torque_enabled=True, policy_enabled=False))
        self.assertTrue(source.read().torque_enabled)
        source._last_recv_s -= 0.1
        state = source.read()
        self.assertIsNone(state.torque_enabled)
        self.assertTrue(state.hold_last_targets)
        self.assertEqual(state.active_moves, set())
        self.assertEqual(state.velocity, {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})

    def test_timeout_before_first_packet_cannot_enable_torque(self):
        source = NetworkInputSource(stale_after_s=0.05)
        state = source.read()
        self.assertIsNone(state.torque_enabled)
        self.assertTrue(state.hold_last_targets)
        self.assertEqual(state.active_moves, set())

    def test_new_session_cannot_start_with_trigger_held(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0, ["walk"], {"vx": 1.0}, session="new"))
        state = source.read()
        self.assertNotIn("walk", state.active_moves)
        self.assertEqual(state.velocity["vx"], 0.0)
        source._apply(packet(1, [], session="new"))
        source._apply(packet(2, ["walk"], {"vx": 1.0}, session="new"))
        self.assertIn("walk", source.read().active_moves)

    def test_out_of_order_snapshot_is_ignored(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(5, [], session="same"))
        source._apply(packet(6, ["hmd_head"], {"vy": 0.5}, session="same"))
        source._apply(packet(4, [], {"vy": -1.0}, session="same"))
        self.assertEqual(source.read().velocity["vy"], 0.5)

    def test_retired_session_cannot_take_control_back(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(10, [], session="old"))
        source._apply(packet(20, [], session="new"))
        source._apply(packet(21, ["walk"], {"vx": 0.5}, session="new"))
        self.assertEqual(source.read().velocity["vx"], 0.5)

        source._apply(packet(11, [], {"vx": -1.0}, session="old"))
        self.assertEqual(source.read().velocity["vx"], 0.5)

    def test_timeout_neutralizes_and_disarms(self):
        source = NetworkInputSource(stale_after_s=0.05)
        source._apply(packet(0))
        source._apply(packet(1, ["walk"], {"vx": 1.0}))
        source._last_recv_s -= 0.1
        stale = source.read()
        self.assertEqual(stale.active_moves, set())
        self.assertTrue(stale.hold_last_targets)
        source._apply(packet(2, ["walk"], {"vx": 1.0}))
        self.assertNotIn("walk", source.read().active_moves)

    def test_robot_safety_inhibit_requires_release_after_recovery(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))
        source._apply(packet(1, ["walk"], {"vx": 1.0}))
        self.assertIn("walk", source.read().active_moves)

        source.set_motion_inhibited(True)
        source._apply(packet(2, ["walk"], {"vx": 1.0}))
        inhibited = source.read()
        self.assertNotIn("walk", inhibited.active_moves)
        self.assertEqual(inhibited.velocity, {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})

        source.set_motion_inhibited(False)
        source._apply(packet(3, ["walk"], {"vx": 1.0}))
        self.assertNotIn("walk", source.read().active_moves)
        source._apply(packet(4))
        source._apply(packet(5, ["walk"], {"vx": 1.0}))
        self.assertIn("walk", source.read().active_moves)

    def test_right_trigger_arm_channel_is_independent_and_release_homes(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(pico_arm_packet(0, enabled=False))
        released = source.read()
        self.assertIn("pico_arms", released.active_moves)
        self.assertFalse(released.arm_tracking_enabled)
        self.assertEqual(released.arm_joint_target, PICO_ARM_HOME_RAD)

        source._apply(pico_arm_packet(1, enabled=True, target=ARM_TARGET))
        held = source.read()
        self.assertIn("pico_arms", held.active_moves)
        self.assertTrue(held.arm_tracking_enabled)
        self.assertEqual(held.arm_joint_target["left"], tuple(ARM_TARGET["left"]))
        self.assertNotIn("walk", held.active_moves)

        source._apply(pico_arm_packet(2, enabled=False))
        released_again = source.read()
        self.assertIn("pico_arms", released_again.active_moves)
        self.assertFalse(released_again.arm_tracking_enabled)
        self.assertEqual(released_again.arm_joint_target, PICO_ARM_HOME_RAD)

    def test_release_requires_authenticated_home_target(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(pico_arm_packet(0, enabled=False))
        wrong_home = {side: list(values) for side, values in ARM_HOME.items()}
        wrong_home["left"][0] = 0.01
        source._apply(pico_arm_packet(1, enabled=False, target=wrong_home))

        rejected = source.read()
        self.assertNotIn("pico_arms", rejected.active_moves)
        self.assertFalse(rejected.arm_tracking_enabled)
        self.assertIsNone(rejected.arm_joint_target)

    def test_new_session_cannot_start_with_right_trigger_held(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(
            pico_arm_packet(
                10, enabled=True, target=ARM_TARGET, session="new-arm-session"
            )
        )
        self.assertNotIn("pico_arms", source.read().active_moves)

        source._apply(
            pico_arm_packet(11, enabled=False, session="new-arm-session")
        )
        source._apply(
            pico_arm_packet(
                12, enabled=True, target=ARM_TARGET, session="new-arm-session"
            )
        )
        self.assertTrue(source.read().arm_tracking_enabled)

    def test_bad_arm_target_clears_and_disarms_complete_overlay(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(pico_arm_packet(0, enabled=False))
        source._apply(pico_arm_packet(1, enabled=True, target=ARM_TARGET))
        self.assertTrue(source.read().arm_tracking_enabled)

        malformed = {"left": list(ARM_TARGET["left"]), "right": [0.0, 0.0]}
        source._apply(pico_arm_packet(2, enabled=True, target=malformed))
        cleared = source.read()
        self.assertNotIn("pico_arms", cleared.active_moves)
        self.assertFalse(cleared.arm_tracking_enabled)
        self.assertIsNone(cleared.arm_joint_target)

        # A still-held trigger cannot reuse the old or new target until a
        # deliberate valid release snapshot arrives.
        source._apply(pico_arm_packet(3, enabled=True, target=ARM_TARGET))
        self.assertNotIn("pico_arms", source.read().active_moves)
        source._apply(pico_arm_packet(4, enabled=False))
        source._apply(pico_arm_packet(5, enabled=True, target=ARM_TARGET))
        self.assertTrue(source.read().arm_tracking_enabled)

    def test_arm_target_bounds_are_robot_authenticated(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(pico_arm_packet(0, enabled=False))
        outside = {side: list(values) for side, values in ARM_TARGET.items()}
        outside["left"][0] = math.radians(100.0) + 1.0e-9
        source._apply(pico_arm_packet(1, enabled=True, target=outside))
        rejected = source.read()
        self.assertNotIn("pico_arms", rejected.active_moves)
        self.assertFalse(rejected.arm_tracking_enabled)
        self.assertIsNone(rejected.arm_joint_target)

    def test_body_policy_degradation_does_not_drop_valid_direct_arms(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(pico_arm_packet(0, enabled=False))
        source._apply(
            pico_arm_packet(
                1,
                enabled=True,
                target=ARM_TARGET,
                extra_moves=["walk"],
                # Missing learned-policy hands/feet intentionally degrades only
                # locomotion to the proven actor.
            )
        )
        state = source.read()
        self.assertTrue(state.learned_policy_degraded)
        self.assertEqual(state.locomotion_policy, "walk")
        self.assertIn("walk", state.active_moves)
        self.assertIn("pico_arms", state.active_moves)
        self.assertTrue(state.arm_tracking_enabled)
        self.assertEqual(state.arm_joint_target["right"], tuple(ARM_TARGET["right"]))

    def test_timeout_and_safety_inhibit_clear_direct_arm_target(self):
        source = NetworkInputSource(stale_after_s=0.05)
        source._apply(pico_arm_packet(0, enabled=False))
        source._apply(pico_arm_packet(1, enabled=True, target=ARM_TARGET))
        source.set_motion_inhibited(True)
        inhibited = source.read()
        self.assertNotIn("pico_arms", inhibited.active_moves)
        self.assertFalse(inhibited.arm_tracking_enabled)
        self.assertIsNone(inhibited.arm_joint_target)

        source.set_motion_inhibited(False)
        source._apply(pico_arm_packet(2, enabled=False))
        source._apply(pico_arm_packet(3, enabled=True, target=ARM_TARGET))
        source._last_recv_s -= 0.1
        timed_out = source.read()
        self.assertNotIn("pico_arms", timed_out.active_moves)
        self.assertFalse(timed_out.arm_tracking_enabled)
        self.assertIsNone(timed_out.arm_joint_target)

    def test_wire_policy_change_never_drops_held_deadman_or_joystick(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))
        source._apply(packet(1, ["walk"], {"vx": 0.5}))
        self.assertIn("walk", source.read().active_moves)

        # The receiver carries a generic wire-policy field for legacy/custom
        # clients. The current PICO bridge always sends pico_teleop and has no
        # policy button. A wire change must still not manufacture a stop.
        source._apply(pico_walk_packet(2, {"vx": 0.5}))
        state = source.read()
        self.assertIn("walk", state.active_moves)
        self.assertEqual(state.velocity["vx"], 0.5)
        self.assertEqual(state.locomotion_policy, "pico_teleop")
        self.assertFalse(state.learned_policy_degraded)

        # Release is the activation boundary; the next press may use the new policy.
        source._apply(pico_release_packet(3))
        self.assertEqual(source.read().locomotion_policy, "pico_teleop")
        source._apply(pico_walk_packet(4, {"vx": 0.5}))
        state = source.read()
        self.assertIn("walk", state.active_moves)
        self.assertEqual(state.locomotion_policy, "pico_teleop")

    def test_unknown_policy_is_rejected(self):
        source = NetworkInputSource()
        with self.assertRaises(ValueError):
            source._apply(packet(0, policy="unknown"))

    def test_missing_foot_targets_fall_back_without_losing_joystick(self):
        incomplete_targets = (
            None,
            {"left": COMPLETE_FEET["left"]},
            {"right": COMPLETE_FEET["right"]},
            {"left": None, "right": COMPLETE_FEET["right"]},
        )
        for incomplete in incomplete_targets:
            with self.subTest(foot_target=incomplete):
                source = NetworkInputSource(stale_after_s=0.5)
                source._apply(pico_release_packet(0))

                source._apply(
                    pico_walk_packet(
                        1,
                        {"vx": 0.8},
                        foot_target=incomplete,
                    )
                )
                fallback = source.read()
                self.assertIn("walk", fallback.active_moves)
                self.assertEqual(fallback.velocity["vx"], 0.8)
                self.assertEqual(fallback.locomotion_policy, "walk")
                self.assertTrue(fallback.learned_policy_degraded)
                self.assertTrue(source._walk_armed)
                self.assertIsNone(fallback.foot_target)
                self.assertIsNone(fallback.hand_target)

                # Receiver recovery also does not drop the command. The selector
                # independently keeps legacy latched until trigger release.
                source._apply(pico_walk_packet(2, {"vx": 0.8}))
                recovered = source.read()
                self.assertIn("walk", recovered.active_moves)
                self.assertEqual(recovered.velocity["vx"], 0.8)
                self.assertEqual(recovered.locomotion_policy, "pico_teleop")
                self.assertFalse(recovered.learned_policy_degraded)

    def test_pico_contract_accepts_exact_80_percent_boundaries(self):
        source = NetworkInputSource(stale_after_s=0.5)
        arm_pico(source)
        feet = {
            "left": [-0.024, 0.024, 0.01],
            "right": [0.0, 0.0, 0.0],
        }
        hands = {
            "left": [-0.064, 0.064, -0.064],
            "right": [0.064, -0.064, 0.064],
        }

        source._apply(
            pico_walk_packet(
                2,
                {"vx": 0.4},
                foot_target=feet,
                hand_target=hands,
            )
        )
        accepted = source.read()

        self.assertIn("walk", accepted.active_moves)
        self.assertEqual(accepted.velocity["vx"], 0.4)
        self.assertEqual(accepted.foot_target["left"], tuple(feet["left"]))
        self.assertEqual(accepted.hand_target["right"], tuple(hands["right"]))

    def test_pico_contract_accepts_only_stationary_narrow_both_feet(self):
        source = NetworkInputSource(stale_after_s=0.5)
        arm_pico(source)
        boundary = {
            "left": [-0.008, 0.008, 0.016],
            "right": [0.008, -0.008, 0.016],
        }
        source._apply(pico_walk_packet(2, foot_target=boundary))
        accepted = source.read()
        self.assertIn("walk", accepted.active_moves)
        self.assertEqual(accepted.foot_target["left"], tuple(boundary["left"]))

        invalid_cases = (
            (
                "too_wide",
                {**boundary, "left": [-0.0080001, 0.0, 0.01]},
                {},
            ),
            ("moving", boundary, {"vx": 0.01}),
            ("turning", boundary, {"vtheta": -0.01}),
        )
        for case, feet, velocity in invalid_cases:
            with self.subTest(case=case):
                source = NetworkInputSource(stale_after_s=0.5)
                arm_pico(source)
                source._apply(pico_walk_packet(2, velocity, foot_target=feet))
                fallback = source.read()
                self.assert_degraded_fallback(
                    fallback,
                    {
                        "vx": velocity.get("vx", 0.0),
                        "vy": velocity.get("vy", 0.0),
                        "vtheta": velocity.get("vtheta", 0.0),
                    },
                )
                self.assertTrue(source._walk_armed)

    def test_pico_contract_rejects_unprojected_support_floor_band(self):
        invalid_feet = (
            {
                "left": [0.001, 0.0, 0.0],
                "right": [0.0, 0.0, 0.0],
            },
            {
                "left": [0.0, 0.0, 0.0025],
                "right": [0.0, 0.0, 0.0],
            },
            {
                "left": [-0.001, 0.001, 0.0025],
                "right": [0.0, 0.0, 0.0],
            },
        )
        for feet in invalid_feet:
            with self.subTest(feet=feet):
                source = NetworkInputSource(stale_after_s=0.5)
                arm_pico(source)
                source._apply(pico_walk_packet(2, foot_target=feet))
                fallback = source.read()
                self.assert_degraded_fallback(
                    fallback,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertTrue(source._walk_armed)

    def test_pico_contract_rejects_epsilon_outside_80_percent_bounds(self):
        invalid_targets = (
            (
                "foot_xy",
                {"left": [0.024000001, 0.0, 0.02], "right": [0.0, 0.0, 0.02]},
                COMPLETE_HANDS,
            ),
            (
                "foot_z",
                {"left": [0.0, 0.0, 0.040000001], "right": [0.0, 0.0, 0.02]},
                COMPLETE_HANDS,
            ),
            (
                "hand",
                COMPLETE_FEET,
                {
                    "left": [-0.064000001, 0.0, 0.0],
                    "right": [0.0, 0.0, 0.0],
                },
            ),
        )
        for name, feet, hands in invalid_targets:
            with self.subTest(name=name):
                source = NetworkInputSource(stale_after_s=0.5)
                arm_pico(source)
                source._apply(pico_walk_packet(2, {"vx": 0.8}))
                self.assertIn("walk", source.read().active_moves)

                source._apply(
                    pico_walk_packet(
                        3,
                        {"vx": 0.8},
                        foot_target=feet,
                        hand_target=hands,
                    )
                )
                fallback = source.read()
                self.assert_degraded_fallback(
                    fallback,
                    {"vx": 0.8, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertTrue(source._walk_armed)

    def test_bad_pico_metadata_falls_back_without_stopping_current_walk(self):
        mutations = (
            ("missing_contract", lambda value: value.pop("body_target_contract")),
            (
                "legacy_contract",
                lambda value: value.update(
                    body_target_contract="microban_pico_offsets_v1"
                ),
            ),
            (
                "missing_margin",
                lambda value: value.pop("body_target_safety_margin"),
            ),
            (
                "wrong_margin",
                lambda value: value.update(body_target_safety_margin=0.81),
            ),
            (
                "boolean_margin",
                lambda value: value.update(body_target_safety_margin=True),
            ),
            (
                "string_margin",
                lambda value: value.update(body_target_safety_margin="0.8"),
            ),
            (
                "huge_integer_margin",
                lambda value: value.update(body_target_safety_margin=10**1000),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                source = NetworkInputSource(stale_after_s=0.5)
                arm_pico(source)
                source._apply(pico_walk_packet(2, {"vx": 0.8}))
                self.assertIn("walk", source.read().active_moves)
                invalid = pico_walk_packet(3, {"vx": 0.8})
                mutate(invalid)

                source._apply(invalid)
                fallback = source.read()

                self.assert_degraded_fallback(
                    fallback,
                    {"vx": 0.8, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertTrue(source._walk_armed)

    def test_pico_contract_requires_complete_paired_hands(self):
        incomplete_hands = (
            None,
            {"left": COMPLETE_HANDS["left"]},
            {"right": COMPLETE_HANDS["right"]},
            {"left": None, "right": COMPLETE_HANDS["right"]},
            {"left": COMPLETE_HANDS["left"], "right": None},
            {
                "left": COMPLETE_HANDS["left"],
                "right": COMPLETE_HANDS["right"],
                "other": [0.0, 0.0, 0.0],
            },
        )
        for hands in incomplete_hands:
            with self.subTest(hand_target=hands):
                source = NetworkInputSource(stale_after_s=0.5)
                arm_pico(source)
                source._apply(pico_walk_packet(2, {"vx": 0.8}))
                source._apply(pico_walk_packet(3, {"vx": 0.8}, hand_target=hands))
                fallback = source.read()
                self.assert_degraded_fallback(
                    fallback,
                    {"vx": 0.8, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertTrue(source._walk_armed)

    def test_invalid_pico_release_remains_a_release_and_arms_next_press(self):
        invalid_releases = (
            (
                "malformed_target",
                pico_release_packet(3, foot_target={"left": [0.0, 0.0]}),
            ),
            (
                "non_null_targets",
                pico_release_packet(
                    3,
                    foot_target=COMPLETE_FEET,
                    hand_target=COMPLETE_HANDS,
                ),
            ),
            ("missing_metadata", packet(3, policy="pico_teleop")),
        )
        for name, invalid_release in invalid_releases:
            with self.subTest(name=name):
                source = NetworkInputSource(stale_after_s=0.5)
                arm_pico(source)
                source._apply(pico_walk_packet(2, {"vx": 0.8}))
                self.assertIn("walk", source.read().active_moves)

                # It is still a released deadman snapshot: targets are discarded,
                # the policy is downgraded, and the following press may walk.
                source._apply(invalid_release)
                released = source.read()
                self.assertNotIn("walk", released.active_moves)
                self.assertEqual(
                    released.velocity,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertEqual(released.locomotion_policy, "walk")
                self.assertTrue(released.learned_policy_degraded)
                self.assertIsNone(released.foot_target)
                self.assertIsNone(released.hand_target)
                self.assertTrue(source._walk_armed)

                source._apply(pico_walk_packet(4, {"vx": 0.8}))
                self.assertIn("walk", source.read().active_moves)

    def test_pico_target_components_require_json_numbers(self):
        invalid_components = (True, "0.01")
        for component in invalid_components:
            with self.subTest(component=component):
                source = NetworkInputSource(stale_after_s=0.5)
                arm_pico(source)
                source._apply(pico_walk_packet(2, {"vx": 0.8}))
                invalid = pico_walk_packet(
                    3,
                    {"vx": 0.8},
                    hand_target={
                        "left": [component, 0.0, 0.0],
                        "right": [0.0, 0.0, 0.0],
                    },
                )

                source._apply(invalid)
                fallback = source.read()

                self.assert_degraded_fallback(
                    fallback,
                    {"vx": 0.8, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertTrue(source._walk_armed)

    def test_invalid_held_pico_policy_change_clears_targets(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))
        source._apply(packet(1, ["walk"], {"vx": 0.8}))
        self.assertIn("walk", source.read().active_moves)

        source._apply(
            pico_walk_packet(
                2,
                {"vx": 0.8},
                body_target_contract="wrong",
            )
        )
        fallback = source.read()

        self.assert_degraded_fallback(
            fallback,
            {"vx": 0.8, "vy": 0.0, "vtheta": 0.0},
        )
        self.assertTrue(source._walk_armed)

    def test_tracker_recovery_never_interrupts_held_joystick(self):
        source = NetworkInputSource(stale_after_s=0.5)
        arm_pico(source)
        source._apply(pico_walk_packet(2, {"vx": 0.8}))
        self.assertIn("walk", source.read().active_moves)

        source._apply(
            pico_walk_packet(
                3,
                {"vx": 0.8},
                body_target_contract="wrong",
            )
        )
        degraded = source.read()
        self.assert_degraded_fallback(
            degraded,
            {"vx": 0.8, "vy": 0.0, "vtheta": 0.0},
        )
        source._apply(pico_walk_packet(4, {"vx": 0.8}))
        recovered = source.read()
        self.assertIn("walk", recovered.active_moves)
        self.assertEqual(recovered.velocity["vx"], 0.8)
        self.assertEqual(recovered.locomotion_policy, "pico_teleop")
        self.assertFalse(recovered.learned_policy_degraded)

        # Release/press remains the selector's policy-retry boundary.
        source._apply(pico_release_packet(5))
        source._apply(pico_walk_packet(6, {"vx": 0.8}))
        self.assertIn("walk", source.read().active_moves)

    def test_walk_policy_does_not_require_pico_body_target_contract(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))
        source._apply(
            packet(
                1,
                ["walk"],
                {"vx": 0.7},
                foot_target={"left": [9.0, 9.0, 9.0]},
                hand_target=None,
            )
        )
        accepted = source.read()
        self.assertIn("walk", accepted.active_moves)
        self.assertEqual(accepted.velocity["vx"], 0.7)

    def test_camera_diagnostics_never_gate_trigger_or_joystick(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))
        visual_fault = packet(1, ["walk"], {"vx": 0.6, "vtheta": -0.4})
        visual_fault.update(
            camera_fresh=False,
            camera_configuration_valid=False,
            stereo_fov_valid=False,
            eye_baseline_valid=False,
        )

        source._apply(visual_fault)
        state = source.read()

        self.assertIn("walk", state.active_moves)
        self.assertEqual(state.velocity["vx"], 0.6)
        self.assertEqual(state.velocity["vtheta"], -0.4)
        self.assertFalse(state.learned_policy_degraded)

    def test_non_finite_value_is_rejected(self):
        source = NetworkInputSource()
        bad = packet(0, [], {"vx": float("nan")})
        with self.assertRaises(ValueError):
            source._apply(bad)

    def test_protocol_metadata_is_required(self):
        source = NetworkInputSource()
        for missing in ("version", "session_id", "seq"):
            bad = packet(0)
            del bad[missing]
            with (
                self.subTest(missing=missing),
                self.assertRaises((TypeError, ValueError)),
            ):
                source._apply(bad)

    def test_watchdog_cannot_be_disabled_with_non_finite_timeout(self):
        for timeout in (float("nan"), float("inf"), 0.0, 1.0):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                NetworkInputSource(stale_after_s=timeout)


def _free_udp_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


class ClockPingTest(unittest.TestCase):
    def setUp(self):
        self.robot_port = _free_udp_port()
        self.source = NetworkInputSource(port=self.robot_port, stale_after_s=0.5)
        self.source.start()
        self.probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.probe.settimeout(1.0)
        self.addCleanup(self.probe.close)
        self.addCleanup(self.source.stop)

    def _send(self, payload: dict) -> None:
        self.probe.sendto(
            json.dumps(payload).encode("utf-8"),
            ("127.0.0.1", self.robot_port),
        )

    def test_ping_is_echoed_as_pong_with_the_same_nonce(self):
        self._send({"type": "clock_ping", "nonce": "abc-123"})
        data, _ = self.probe.recvfrom(4096)
        self.assertEqual(json.loads(data), {"type": "clock_pong", "nonce": "abc-123"})

    def test_ping_never_touches_user_input_state(self):
        self.source._apply(packet(0))  # release walk so a later hold could arm it
        before = self.source.read()
        self._send({"type": "clock_ping", "nonce": "n"})
        self.probe.recvfrom(4096)
        after = self.source.read()
        self.assertEqual(before.active_moves, after.active_moves)
        self.assertEqual(before.velocity, after.velocity)

    def test_malformed_ping_is_silently_ignored(self):
        for bad in (
            {"type": "clock_ping"},
            {"type": "clock_ping", "nonce": 1},
            {"type": "clock_ping", "nonce": "x" * 65},
        ):
            with self.subTest(bad=bad):
                self._send(bad)
                with self.assertRaises((TimeoutError, socket.timeout)):
                    self.probe.settimeout(0.2)
                    self.probe.recvfrom(4096)
                self.probe.settimeout(1.0)

    def test_pong_carries_the_latest_head_telemetry_once_set(self):
        self.source.set_head_telemetry(
            head=0.1,
            neck_roll=-0.2,
            neck_pitch=0.3,
            trunk_roll=0.01,
            trunk_pitch=-0.02,
        )
        self._send({"type": "clock_ping", "nonce": "with-telemetry"})
        data, _ = self.probe.recvfrom(4096)
        self.assertEqual(
            json.loads(data),
            {
                "type": "clock_pong",
                "nonce": "with-telemetry",
                "head": 0.1,
                "neck_roll": -0.2,
                "neck_pitch": 0.3,
                "trunk_roll": 0.01,
                "trunk_pitch": -0.02,
            },
        )

    def test_non_finite_telemetry_is_ignored_and_pong_stays_bare(self):
        self.source.set_head_telemetry(
            head=float("nan"),
            neck_roll=0.0,
            neck_pitch=0.0,
            trunk_roll=0.0,
            trunk_pitch=0.0,
        )
        self._send({"type": "clock_ping", "nonce": "nan-rejected"})
        data, _ = self.probe.recvfrom(4096)
        self.assertEqual(
            json.loads(data), {"type": "clock_pong", "nonce": "nan-rejected"}
        )


class NetworkDisconnectSchedulerTest(unittest.TestCase):
    def test_timeout_holds_physical_goal_until_explicit_b(self):
        source = NetworkInputSource(stale_after_s=0.05)
        source._apply(packet(0, torque_enabled=True, policy_enabled=False))

        class Controller:
            def __init__(self):
                self.tick = 0
                self.events = []

            def sync_write_goal_position(self, ids, positions):
                self.events.append(("goal", self.tick, list(positions)))

            def sync_write_kp(self, ids, values):
                self.events.append(("kp", self.tick, list(values)))

            def sync_write_torque_enable(self, ids, enabled):
                self.events.append(("torque", self.tick, all(enabled)))

            def shutdown(self):
                self.events.append(("shutdown", self.tick, None))

        controller = Controller()
        with tempfile.TemporaryDirectory() as temp_dir:
            stop_path = Path(temp_dir) / "stop"

            class Observer:
                reads = 0

                def read_state(self, _dt):
                    self.reads += 1
                    controller.tick = self.reads
                    if self.reads == 2:
                        source._last_recv_s -= 1.0  # longer than watchdog
                    elif self.reads == 3:
                        source._apply(
                            packet(
                                1,
                                torque_enabled=False,
                                policy_enabled=False,
                                torque_off_requested=True,
                            )
                        )
                        stop_path.touch()
                    return RobotState(
                        gyro=[0.0, 0.0, 0.0],
                        quat=[1.0, 0.0, 0.0, 0.0],
                        body_quat=[1.0, 0.0, 0.0, 0.0],
                        projected_gravity=[0.0, 0.0, -1.0],
                        motor_positions={
                            name: neutral + 0.4
                            for name, neutral in NEUTRAL_POSE.items()
                        },
                        motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
                    )

            scheduler = Scheduler(
                controller=controller,
                stop_flag_path=str(stop_path),
                input_source=source,
                moves={},
                hardware_power_control=True,
                serial_hold_on_error=True,
            )
            scheduler.observer = Observer()
            with (
                patch.object(source, "start"),
                patch.object(source, "stop"),
                patch("scheduler.time.sleep"),
                redirect_stdout(io.StringIO()),
            ):
                scheduler.run()

        self.assertIn(("torque", 1, True), controller.events)
        self.assertFalse(
            any(kind == "torque" and tick == 2 and not value
                for kind, tick, value in controller.events),
            "a UDP timeout must not write torque OFF",
        )
        self.assertFalse(
            any(kind == "goal" and tick == 2
                for kind, tick, _value in controller.events),
            "a UDP timeout must not advance the neutral or policy goal",
        )
        self.assertIn(("torque", 3, False), controller.events)
        shutdown_index = next(
            index for index, event in enumerate(controller.events)
            if event[0] == "shutdown"
        )
        b_off_index = controller.events.index(("torque", 3, False))
        self.assertLess(b_off_index, shutdown_index)


if __name__ == "__main__":
    unittest.main()
