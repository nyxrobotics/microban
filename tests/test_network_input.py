import unittest

from input.network_input import NetworkInputSource


BODY_TARGET_CONTRACT = "microban_pico_offsets_v1"
BODY_TARGET_SAFETY_MARGIN = 0.8
COMPLETE_FEET = {
    "left": [0.01, 0.0, 0.02],
    "right": [-0.01, 0.0, 0.02],
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
    # Changing joint owner forces one released packet; the second release arms it.
    source._apply(pico_release_packet(0))
    source._apply(pico_release_packet(1))


class NetworkInputTest(unittest.TestCase):
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

    def test_policy_change_requires_released_deadman_and_an_extra_release(self):
        source = NetworkInputSource(stale_after_s=0.5)
        source._apply(packet(0))
        source._apply(packet(1, ["walk"], {"vx": 0.5}))
        self.assertIn("walk", source.read().active_moves)

        # A malicious/inconsistent sender cannot hot-swap the joint owner while
        # the trigger is active. It forces a stop and retains the old policy.
        source._apply(packet(2, ["walk"], {"vx": 0.5}, policy="pico_teleop"))
        state = source.read()
        self.assertNotIn("walk", state.active_moves)
        self.assertEqual(state.locomotion_policy, "walk")

        # A released packet accepts the mode but does not arm it. A second
        # released snapshot and a later trigger press are required.
        source._apply(pico_release_packet(3))
        self.assertEqual(source.read().locomotion_policy, "pico_teleop")
        source._apply(pico_walk_packet(4, {"vx": 0.5}))
        self.assertNotIn("walk", source.read().active_moves)
        source._apply(pico_release_packet(5))
        source._apply(pico_walk_packet(6, {"vx": 0.5}))
        state = source.read()
        self.assertIn("walk", state.active_moves)
        self.assertEqual(state.locomotion_policy, "pico_teleop")

    def test_unknown_policy_is_rejected(self):
        source = NetworkInputSource()
        with self.assertRaises(ValueError):
            source._apply(packet(0, policy="unknown"))

    def test_pico_walk_requires_both_foot_targets_and_release_to_rearm(self):
        incomplete_targets = (
            None,
            {"left": COMPLETE_FEET["left"]},
            {"right": COMPLETE_FEET["right"]},
            {"left": None, "right": COMPLETE_FEET["right"]},
        )
        for incomplete in incomplete_targets:
            with self.subTest(foot_target=incomplete):
                source = NetworkInputSource(stale_after_s=0.5)
                # Mode changes require one extra released snapshot before arming.
                source._apply(pico_release_packet(0))
                source._apply(pico_release_packet(1))

                source._apply(
                    pico_walk_packet(
                        2,
                        {"vx": 0.8},
                        foot_target=incomplete,
                    )
                )
                refused = source.read()
                self.assertNotIn("walk", refused.active_moves)
                self.assertEqual(
                    refused.velocity,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertFalse(source._walk_armed)
                self.assertIsNone(refused.foot_target)
                self.assertIsNone(refused.hand_target)

                # Supplying targets while still held cannot bypass the latch.
                source._apply(pico_walk_packet(3, {"vx": 0.8}))
                self.assertNotIn("walk", source.read().active_moves)

                source._apply(pico_release_packet(4))
                source._apply(pico_walk_packet(5, {"vx": 0.8}))
                accepted = source.read()
                self.assertIn("walk", accepted.active_moves)
                self.assertEqual(accepted.velocity["vx"], 0.8)

    def test_pico_contract_accepts_exact_80_percent_boundaries(self):
        source = NetworkInputSource(stale_after_s=0.5)
        arm_pico(source)
        feet = {
            "left": [-0.024, 0.024, 0.0],
            # Match the bridge's actual 0.05 * 0.8 float, as well as the
            # mathematical 0.04 boundary, without admitting a real epsilon.
            "right": [0.024, -0.024, 0.05 * BODY_TARGET_SAFETY_MARGIN],
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
                refused = source.read()
                self.assertNotIn("walk", refused.active_moves)
                self.assertEqual(
                    refused.velocity,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertIsNone(refused.foot_target)
                self.assertIsNone(refused.hand_target)
                self.assertFalse(source._walk_armed)

    def test_pico_contract_missing_or_wrong_metadata_stops_current_walk(self):
        mutations = (
            ("missing_contract", lambda value: value.pop("body_target_contract")),
            ("wrong_contract", lambda value: value.update(body_target_contract="v2")),
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
                refused = source.read()

                self.assertNotIn("walk", refused.active_moves)
                self.assertEqual(
                    refused.velocity,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertIsNone(refused.foot_target)
                self.assertIsNone(refused.hand_target)
                self.assertFalse(source._walk_armed)

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
                source._apply(
                    pico_walk_packet(3, {"vx": 0.8}, hand_target=hands)
                )
                refused = source.read()
                self.assertNotIn("walk", refused.active_moves)
                self.assertEqual(
                    refused.velocity,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertIsNone(refused.foot_target)
                self.assertIsNone(refused.hand_target)
                self.assertFalse(source._walk_armed)

    def test_invalid_pico_release_packet_fails_closed_and_cannot_rearm(self):
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

                # Invalid release packets are accepted sequence snapshots, so
                # they must replace the active command with a fail-closed state.
                source._apply(invalid_release)
                refused = source.read()
                self.assertNotIn("walk", refused.active_moves)
                self.assertEqual(
                    refused.velocity,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertIsNone(refused.foot_target)
                self.assertIsNone(refused.hand_target)
                self.assertFalse(source._walk_armed)

                source._apply(pico_walk_packet(4, {"vx": 0.8}))
                self.assertNotIn("walk", source.read().active_moves)
                source._apply(pico_release_packet(5))
                source._apply(pico_walk_packet(6, {"vx": 0.8}))
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
                refused = source.read()

                self.assertNotIn("walk", refused.active_moves)
                self.assertEqual(
                    refused.velocity,
                    {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
                )
                self.assertIsNone(refused.foot_target)
                self.assertIsNone(refused.hand_target)
                self.assertFalse(source._walk_armed)

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
        refused = source.read()

        self.assertEqual(refused.locomotion_policy, "walk")
        self.assertNotIn("walk", refused.active_moves)
        self.assertEqual(
            refused.velocity,
            {"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
        )
        self.assertIsNone(refused.foot_target)
        self.assertIsNone(refused.hand_target)
        self.assertFalse(source._walk_armed)

    def test_invalid_pico_contract_requires_release_to_rearm(self):
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
        self.assertNotIn("walk", source.read().active_moves)
        source._apply(pico_walk_packet(4, {"vx": 0.8}))
        self.assertNotIn("walk", source.read().active_moves)

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
            with self.subTest(missing=missing), self.assertRaises((TypeError, ValueError)):
                source._apply(bad)

    def test_watchdog_cannot_be_disabled_with_non_finite_timeout(self):
        for timeout in (float("nan"), float("inf"), 0.0, 1.0):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                NetworkInputSource(stale_after_s=timeout)


if __name__ == "__main__":
    unittest.main()
