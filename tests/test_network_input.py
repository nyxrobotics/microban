import unittest

from input.network_input import NetworkInputSource

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
        self.assertEqual(source.read().active_moves, set())
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

    def test_policy_button_change_never_drops_held_deadman_or_joystick(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))
        source._apply(packet(1, ["walk"], {"vx": 0.5}))
        self.assertIn("walk", source.read().active_moves)

        # The receiver carries the requested button state, while the policy selector
        # keeps the current child for this activation. It must not manufacture a stop.
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


if __name__ == "__main__":
    unittest.main()
