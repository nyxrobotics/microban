import unittest

from input.network_input import NetworkInputSource


def packet(
    seq,
    moves=None,
    velocity=None,
    session="test",
    policy="walk",
    foot_target=None,
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
        "foot_target": foot_target,
        "hand_target": None,
    }


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
        complete_feet = {
            "left": [0.0, 0.0, 0.0],
            "right": [0.0, 0.0, 0.0],
        }
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
        source._apply(packet(3, policy="pico_teleop"))
        self.assertEqual(source.read().locomotion_policy, "pico_teleop")
        source._apply(
            packet(
                4,
                ["walk"],
                {"vx": 0.5},
                policy="pico_teleop",
                foot_target=complete_feet,
            )
        )
        self.assertNotIn("walk", source.read().active_moves)
        source._apply(packet(5, policy="pico_teleop"))
        source._apply(
            packet(
                6,
                ["walk"],
                {"vx": 0.5},
                policy="pico_teleop",
                foot_target=complete_feet,
            )
        )
        state = source.read()
        self.assertIn("walk", state.active_moves)
        self.assertEqual(state.locomotion_policy, "pico_teleop")

    def test_unknown_policy_is_rejected(self):
        source = NetworkInputSource()
        with self.assertRaises(ValueError):
            source._apply(packet(0, policy="unknown"))

    def test_pico_walk_requires_both_foot_targets_and_release_to_rearm(self):
        complete = {
            "left": [0.01, 0.0, 0.02],
            "right": [-0.01, 0.0, 0.02],
        }
        incomplete_targets = (
            None,
            {"left": complete["left"]},
            {"right": complete["right"]},
            {"left": None, "right": complete["right"]},
        )
        for incomplete in incomplete_targets:
            with self.subTest(foot_target=incomplete):
                source = NetworkInputSource(stale_after_s=0.5)
                # Mode changes require one extra released snapshot before arming.
                source._apply(packet(0, policy="pico_teleop"))
                source._apply(packet(1, policy="pico_teleop"))

                source._apply(
                    packet(
                        2,
                        ["walk"],
                        {"vx": 0.8},
                        policy="pico_teleop",
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

                # Supplying targets while still held cannot bypass the latch.
                source._apply(
                    packet(
                        3,
                        ["walk"],
                        {"vx": 0.8},
                        policy="pico_teleop",
                        foot_target=complete,
                    )
                )
                self.assertNotIn("walk", source.read().active_moves)

                source._apply(packet(4, policy="pico_teleop"))
                source._apply(
                    packet(
                        5,
                        ["walk"],
                        {"vx": 0.8},
                        policy="pico_teleop",
                        foot_target=complete,
                    )
                )
                accepted = source.read()
                self.assertIn("walk", accepted.active_moves)
                self.assertEqual(accepted.velocity["vx"], 0.8)

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
