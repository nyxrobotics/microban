"""Whole-body tracking: contract, reference wire, stop sequence and policy selection.

The observation itself is checked against the training in
test_wholebody_observation.py.
"""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from constants import (
    HOME_PROJECTED_GRAVITY,
    HOME_ROOT_QUAT_WXYZ,
    HOME_TRUNK_PITCH_RAD,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
)
from input.input_source import UserInput
from input.wholebody_reference import (
    ACTIVE,
    CALIBRATED,
    HELD,
    RESTART,
    STALE,
    ReferenceDecodeError,
    ReferenceReceiver,
    decode_frame,
    encode_frame,
)
from moves.move import MotorCommand, Move, MoveState
from moves.hmd_head import HmdHeadTrackingMove
from moves.move import WholebodyClaim
from moves.pico_arms import PicoArmTrackingMove
from moves.policy_selector import PolicySelectableWalkMove
from moves.wholebody import (
    HANDOFF_TICKS,
    MAX_HELD_FLAG_S,
    MAX_SEQUENCE_GAP,
    MAX_STALE_FLAG_S,
    RETURN_S,
    SETTLE_S,
    STOP_HOLD_S,
    Phase,
    WholebodyMove,
    quat_from_yaw,
    quat_mul,
)
from observer import Observation, RobotState
from policy_contract import PolicyContractError, PolicySelfTestError
from policy_fixtures import JOINT_NAMES, home_pose_stamp, csv
import wholebody_contract as wc

ROBOT_JOINTS = wc.OBSERVATION_JOINT_NAMES


def home_frame() -> list[float]:
    """The observation of the robot standing still at HOME under a HOME reference."""
    frame = np.zeros(wc.OBSERVATION_WIDTH)
    terms = wc.term_slices()
    frame[terms["ref_joint_pos"]] = [NEUTRAL_POSE[n] for n in OBSERVATION_DOF_ORDER]
    frame[terms["ref_anchor_ori_b"]] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    frame[terms["projected_gravity"]] = HOME_PROJECTED_GRAVITY
    frame[terms["ref_projected_gravity_b"]] = HOME_PROJECTED_GRAVITY
    frame[terms["ref_anchor_height"]] = 0.17
    return frame.tolist()


def metadata(**overrides) -> dict[str, str]:
    histories = [[home_frame()] * wc.HISTORY_LENGTH for _ in range(8)]
    values = {
        "microban_policy_contract": wc.CONTRACT,
        "microban_policy_kind": wc.KIND,
        "microban_recipe": wc.RECIPE,
        "home_pose": home_pose_stamp(),
        "joint_names": ",".join(JOINT_NAMES),
        "default_joint_pos": csv(NEUTRAL_POSE[n] for n in JOINT_NAMES),
        "action_joint_names": ",".join(OBSERVATION_DOF_ORDER),
        "action_scale": "1.0",
        "action_clip_lower": csv([-math.pi] * 18),
        "action_clip_upper": csv([math.pi] * 18),
        "observation_schema_json": json.dumps([list(t) for t in wc.OBSERVATION_SCHEMA]),
        "observation_joint_names": ",".join(wc.OBSERVATION_JOINT_NAMES),
        "reference_joint_names": ",".join(OBSERVATION_DOF_ORDER),
        "neck_limits_rad_json": json.dumps({"head": [-1.48, 1.48], "neck_roll": [-0.4, 0.4],
                                            "neck_pitch": [-1.48, 0.4]}),
        "neck_slew_rates_rad_s_json": json.dumps({"head": 2.5, "neck_roll": 2.5, "neck_pitch": 2.5}),
        "control_hz": "50.0",
        "checkpoint_filename": "model_2000.pt",
        "checkpoint_iteration": "2000",
        "checkpoint_sha256": "a" * 64,
        "benchmark_report_sha256": "b" * 64,
        "eval_only_benchmark_report_sha256": "c" * 64,
        "self_test_histories_json": json.dumps(histories),
        "self_test_actions_json": json.dumps([[0.01] * 18] * 8),
        **wc.FIXED_FIELDS,
    }
    values.update(overrides)
    return {k: v for k, v in values.items() if v is not None}


class Session:
    """Enough of an onnxruntime session; run() answers ``action`` and records its input."""

    def __init__(self, meta, action=0.01, shape=(1, 10, 118)):
        self.meta = meta
        self.action = action
        self.shape = shape
        self.inputs = []

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.meta)

    def get_inputs(self):
        return [SimpleNamespace(name="obs_history", shape=list(self.shape), type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name="actions", shape=[1, 18], type="tensor(float)")]

    def run(self, _outputs, feeds):
        self.inputs.append(feeds["obs_history"])
        value = self.action(len(self.inputs)) if callable(self.action) else self.action
        return [np.full((1, 18), value, dtype=np.float32)]


class ContractTest(unittest.TestCase):
    def test_a_contract_file_passes_its_self_test(self):
        session = Session(metadata())
        contract = wc.parse_wholebody_policy(session)
        self.assertEqual(wc.run_self_test(session, contract), 8)
        self.assertEqual(contract.neck_limits_rad["neck_pitch"], (-1.48, 0.4))

    def test_refusals(self):
        cases = {
            "another contract": {"microban_policy_contract": "microban-policy-1"},
            "another HOME": {"home_pose": home_pose_stamp({**NEUTRAL_POSE, "left_knee": 0.3})},
            "another layout": {"observation_schema_json": json.dumps([["ref_joint_pos", 18]])},
            "another reference timing": {"reference_time": "now"},
            "no yaw alignment": {"reference_yaw": "none"},
            "a neck limit without HOME": {"neck_limits_rad_json": json.dumps(
                {"head": [0.1, 1.0], "neck_roll": [-0.4, 0.4], "neck_pitch": [-1.48, 0.4]})},
            "a short history": {"self_test_histories_json": json.dumps([[home_frame()] * 9] * 8)},
            "an impossible row": {"self_test_histories_json": json.dumps(
                [[[0.0] * wc.OBSERVATION_WIDTH] * 10] * 8)},
            "too few rows": {"self_test_histories_json": json.dumps([[home_frame()] * 10] * 7),
                             "self_test_actions_json": json.dumps([[0.0] * 18] * 7)},
            "the PICO input name": {},
        }
        for name, override in cases.items():
            with self.subTest(name):
                session = Session(metadata(**override))
                if name == "the PICO input name":
                    session.get_inputs = lambda: [SimpleNamespace(name="obs", shape=[1, 10, 118])]
                with self.assertRaises(PolicyContractError):
                    wc.parse_wholebody_policy(session)

    def test_self_test_catches_another_network(self):
        session = Session(metadata(), action=0.02)
        with self.assertRaises(PolicySelfTestError):
            wc.run_self_test(session, wc.parse_wholebody_policy(session))

    def test_a_dry_run_runs_only_when_allowed(self):
        meta = metadata(dry_run_not_deployable="true")
        with mock.patch.dict(os.environ, {"MICROBAN_ALLOW_DRYRUN_POLICY": ""}):
            with self.assertRaises(PolicyContractError):
                wc.parse_wholebody_policy(Session(meta))
        with mock.patch.dict(os.environ, {"MICROBAN_ALLOW_DRYRUN_POLICY": "1"}):
            self.assertTrue(wc.parse_wholebody_policy(Session(meta)).dry_run)

    def test_the_manifest_binds_the_installed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            agents = Path(directory)
            (agents / wc.POLICY_FILE).write_bytes(b"onnx bytes")
            manifest = {"contract": wc.CONTRACT, "home_tag": wc.HOME_TAG, "dry_run": False,
                        "file": wc.POLICY_FILE, "sha256": "0" * 64, "checkpoint_sha256": "a" * 64}
            (agents / wc.MANIFEST_FILE).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(PolicyContractError, "not the file"):
                wc.load_wholebody_policy(agents / wc.POLICY_FILE)
            (agents / wc.MANIFEST_FILE).write_text(json.dumps({**manifest, "home_tag": "other"}))
            with self.assertRaisesRegex(PolicyContractError, "HOME"):
                wc.load_wholebody_policy(agents / wc.POLICY_FILE)


ENGAGED = CALIBRATED | ACTIVE


def frame_arrays(*, pitch=HOME_TRUNK_PITCH_RAD, height=0.17, joint_offset=0.0, yaw=0.0, vel=0.0, neck=(0.0, 0.0, 0.0)):
    quat = quat_mul(quat_from_yaw(yaw), [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    joints = np.array([NEUTRAL_POSE[n] for n in ROBOT_JOINTS]) + joint_offset
    joints[:3] = neck  # head, neck_roll, neck_pitch (HOME is 0)
    others = np.zeros((21, 3))
    return {
        "joint_pos": joints, "joint_vel": np.full(21, vel),
        "body_pos_w": np.vstack([[0.0, 0.0, height], others]),
        "body_quat_w": np.vstack([quat, np.tile([1.0, 0.0, 0.0, 0.0], (21, 1))]),
        "body_lin_vel_w": np.vstack([[vel, 0.0, 0.0], others]),
        "body_ang_vel_w": np.vstack([[0.0, 0.0, vel], others]),
    }


class WireTest(unittest.TestCase):
    def test_round_trip(self):
        data = encode_frame(7, 12.5, frame_arrays(pitch=0.2, vel=0.3), flags=ENGAGED | STALE)
        self.assertEqual(len(data), 24 + 4 * 328)  # the page's 1336 bytes
        frame = decode_frame(data)
        self.assertEqual((frame.seq, frame.time_s, frame.engaged), (7, 12.5, True))
        self.assertAlmostEqual(float(frame.anchor_pos[2]), 0.17, places=6)
        self.assertAlmostEqual(float(frame.anchor_lin_vel[0]), 0.3, places=6)
        for flags in (0, CALIBRATED, ACTIVE):
            self.assertFalse(decode_frame(encode_frame(1, 0.0, frame_arrays(), flags=flags)).engaged)

    def test_only_the_control_packets_sender_is_heard(self):
        with tempfile.TemporaryDirectory() as directory:
            allowlist = Path(directory) / "microban-pico-runtime"
            allowlist.write_text("MICROBAN_NETWORK_ALLOWED_IP=192.168.1.20\n")
            receiver = ReferenceReceiver(port=1, allowed_remote_file=str(allowlist))
            self.assertTrue(receiver._sender_allowed("192.168.1.20"))
            self.assertFalse(receiver._sender_allowed("192.168.1.21"))
            allowlist.unlink()
            receiver._allowed_checked_s = -math.inf
            self.assertFalse(receiver._sender_allowed("192.168.1.20"))

    def test_the_sequence_number_wraps_around(self):
        receiver = ReferenceReceiver(port=1, clock=lambda: 0.0)
        for k, seq in enumerate((0xFFFFFFFE, 0xFFFFFFFF, 0, 1)):
            self.assertTrue(receiver.accept(encode_frame(seq, 5.0 + 0.02 * k, frame_arrays(), ENGAGED), 0.02 * k))
        self.assertEqual((receiver.read().lost, receiver.read().stream), (0, 0))

    def test_malformed_datagrams_are_refused(self):
        good = encode_frame(1, 0.0, frame_arrays())
        bad_quat = frame_arrays()
        bad_quat["body_quat_w"][0] = [2.0, 0.0, 0.0, 0.0]
        nan = frame_arrays()
        nan["joint_vel"][3] = np.nan
        two_bodies = {key: value[:2] if key.startswith("body") else value for key, value in frame_arrays().items()}
        for name, data in {
            "short": good[:20], "magic": b"XXXX" + good[4:], "length": good + b"\0\0\0\0",
            "body count": encode_frame(1, 0.0, two_bodies),
            "version": good[:4] + (9).to_bytes(2, "little") + good[6:],
            "quaternion": encode_frame(1, 0.0, bad_quat), "nan": encode_frame(1, 0.0, nan),
        }.items():
            with self.subTest(name), self.assertRaises(ReferenceDecodeError):
                decode_frame(data)

    def test_receiver_reports_gaps_restarts_late_frames_and_activations(self):
        receiver = ReferenceReceiver(port=1, clock=lambda: 0.0)

        def send(seq, sent, received, flags=ENGAGED):
            return receiver.accept(encode_frame(seq, sent, frame_arrays(), flags), received_s=received)

        send(0, 1000.00, 0.01, flags=CALIBRATED)
        send(1, 1000.02, 0.03)
        send(2, 1000.04, 0.05)
        self.assertEqual((receiver.read().lost, receiver.read().activation), (0, 1))
        send(9, 1000.18, 0.19)
        self.assertEqual(receiver.read().lost, 6)
        self.assertFalse(send(8, 1000.16, 0.2))  # reordered: dropped
        send(10, 1000.20, 0.51)
        self.assertAlmostEqual(receiver.read().latency_excess_s, 0.3, places=6)
        send(11, 1000.22, 0.52, flags=CALIBRATED)  # the operator stopped on the PC
        send(12, 1000.24, 0.53)  # and started again
        self.assertEqual(receiver.read().activation, 2)
        send(13, 1000.26, 0.54, flags=ENGAGED | RESTART)
        self.assertEqual(receiver.read().activation, 3)
        # The PC restarted: the sequence begins again, sent later.
        self.assertTrue(send(0, 1010.0, 10.0, flags=CALIBRATED))
        self.assertEqual(receiver.read().stream, 1)
        for k, flags in enumerate((ENGAGED | STALE, ENGAGED | STALE, ENGAGED | STALE | HELD)):
            send(1 + k, 1010.02 * (k + 1), 10.02 * (k + 1), flags)
        snapshot = receiver.read()
        self.assertAlmostEqual(snapshot.stale_flag_s, 1010.02 * 3 - 1010.02, places=6)
        self.assertEqual(snapshot.held_flag_s, 0.0)


class FakeReceiver:
    def __init__(self):
        self.receiver = ReferenceReceiver(port=1, clock=lambda: self.time)
        self.time = 0.0
        self.seq = 0
        self.sent_offset = 1000.0  # the PC's clock

    def send(self, **kwargs):
        flags = kwargs.pop("flags", ENGAGED)
        frame = encode_frame(self.seq, self.time + self.sent_offset, frame_arrays(**kwargs), flags)
        self.receiver.accept(frame, self.time)
        self.seq += 1

    def read(self):
        return self.receiver.read()

    def now(self):
        return self.time


def tick_obs(t, *, gravity=None, quat=None, moves=("walk",), balance_only=False, policy="wholebody"):
    quat = list(HOME_ROOT_QUAT_WXYZ) if quat is None else quat
    return Observation(
        robot_state=RobotState(
            time_s=t, gyro=[0.0, 0.0, 0.0], body_quat=quat,
            projected_gravity=list(HOME_PROJECTED_GRAVITY) if gravity is None else gravity,
            motor_positions=dict(NEUTRAL_POSE), motor_velocities={n: 0.0 for n in NEUTRAL_POSE},
        ),
        user_input=UserInput(active_moves=set(moves), locomotion_policy=policy, balance_only=balance_only),
    )


def started_move(action=0.01):
    ref = FakeReceiver()
    session = Session(metadata(), action=action)
    move = WholebodyMove(ref, session=session)
    ref.send()
    move.state = MoveState.STARTING
    move.on_start(tick_obs(0.0), MotorCommand())
    return move, ref, session


def run(move, ref, start, end, send=True, **frame):
    """Ticks from ``start`` to ``end`` (s), the PC sending a frame each tick when ``send``."""
    command = MotorCommand()
    for k in range(round((end - start) / 0.02)):
        t = start + 0.02 * (k + 1)
        ref.time = t
        if send:
            ref.send(**frame)
        command = MotorCommand()
        move.step(tick_obs(t), command)
        if move.handoff_ready:
            break
    return command, t


class MoveTest(unittest.TestCase):
    def test_targets_are_home_plus_raw_and_the_neck_follows_the_reference(self):
        move, ref, _ = started_move(action=0.25)
        command, _ = run(move, ref, 0.0, 0.1)
        for name in OBSERVATION_DOF_ORDER:
            self.assertAlmostEqual(command.target_angles[name], max(-math.pi, min(math.pi, NEUTRAL_POSE[name] + 0.25)))
        for name in ("head", "neck_roll", "neck_pitch"):
            self.assertAlmostEqual(command.target_angles[name], NEUTRAL_POSE[name])

    def test_the_neck_follows_the_references_neck_and_returns_to_home_with_it(self):
        move, ref, _ = started_move()
        rate = 2.5 * 0.02
        neck_target = lambda command: [command.target_angles[n] for n in ("head", "neck_roll", "neck_pitch")]
        command, t = run(move, ref, 0.0, 0.1, neck=(0.6, 0.9, -0.3))
        # Slewed at 2.5 rad/s from the measured neck (HOME) toward the reference.
        np.testing.assert_allclose(neck_target(command), [5 * rate, 5 * rate, -5 * rate], atol=1e-9)
        command, t = run(move, ref, t, 1.5, neck=(0.6, 0.9, -0.3))
        # Clamped to the contract's limits (neck_roll +-0.4).
        np.testing.assert_allclose(neck_target(command), [0.6, 0.4, -0.3], atol=1e-9)
        # A falling tick keeps the neck's last target.
        command = MotorCommand()
        move.step(tick_obs(t + 0.01, gravity=[0.9, 0.0, -0.3]), command)
        np.testing.assert_allclose(neck_target(command), [0.6, 0.4, -0.3], atol=1e-9)
        # The PC stops: the neck holds, then returns to HOME with the reference.
        command, t = run(move, ref, t, t + 0.3, flags=CALIBRATED, neck=(0.0, 0.0, 0.0))
        self.assertEqual(move.phase, Phase.HOLD)
        np.testing.assert_allclose(neck_target(command), [0.6, 0.4, -0.3], atol=1e-9)
        command, t = run(move, ref, t, t + STOP_HOLD_S + RETURN_S, flags=CALIBRATED)
        np.testing.assert_allclose(neck_target(command), [0.0, 0.0, 0.0], atol=1e-9)

    def test_settle_then_track(self):
        move, ref, session = started_move()
        run(move, ref, 0.0, SETTLE_S - 0.1, vel=0.5)
        self.assertEqual(move.phase, Phase.SETTLE)
        # Standing still while settling: the reference velocities read 0.
        self.assertEqual(float(np.abs(session.inputs[-1][0, -1, 18:36]).max()), 0.0)
        run(move, ref, SETTLE_S - 0.1, SETTLE_S + 0.2, vel=0.5)
        self.assertEqual(move.phase, Phase.TRACK)
        self.assertAlmostEqual(float(session.inputs[-1][0, -1, 18]), 0.5, places=6)

    def stop_and_hand_off(self, move, ref, t, **frame):
        """Ticks until the hand-off; the hold and the return last their times."""
        changes = [(t, move.phase)]
        while not move.handoff_ready and t < changes[0][0] + 5.0:
            _, t = run(move, ref, t, t + 0.02, **frame)
            if move.phase != changes[-1][1]:
                changes.append((t, move.phase))
        self.assertTrue(move.handoff_ready)
        self.assertEqual([phase for _, phase in changes], [Phase.HOLD, Phase.RETURN, Phase.STAND])
        self.assertAlmostEqual(changes[2][0] - changes[1][0], RETURN_S, delta=0.03)
        self.assertLessEqual(t - changes[2][0], (HANDOFF_TICKS + 1) * 0.02)
        return t

    def test_a_silent_sender_stops_tracking_and_returns_to_home(self):
        move, ref, session = started_move()
        _, t = run(move, ref, 0.0, 2.0, joint_offset=0.2)
        _, t = run(move, ref, t, t + 0.08, send=False)
        self.assertEqual(move.phase, Phase.TRACK)
        _, t = run(move, ref, t, t + 0.06, send=False)
        self.assertEqual(move.phase, Phase.HOLD)
        self.assertIn("no reference frame", move.stop_reason)
        # The hold keeps the last pose standing still.
        held = session.inputs[-1][0, -1]
        self.assertAlmostEqual(float(held[0]), NEUTRAL_POSE[OBSERVATION_DOF_ORDER[0]] + 0.2, places=5)
        self.assertEqual(float(np.abs(held[18:36]).max()), 0.0)
        self.stop_and_hand_off(move, ref, t, send=False)
        # Back at HOME: the reference is HOME.
        self.assertAlmostEqual(float(session.inputs[-1][0, -1, 0]), NEUTRAL_POSE[OBSERVATION_DOF_ORDER[0]], places=5)

    def test_every_stop_reason(self):
        def restart(ref):
            ref.seq = 0
            ref.sent_offset += 60.0

        for name, before, frame, ticks in (
            ("lost frames", lambda ref: setattr(ref, "seq", ref.seq + MAX_SEQUENCE_GAP + 1), {}, 1),
            ("restarted PC", restart, {}, 1),
            ("the PC stopped", lambda ref: None, {"flags": CALIBRATED}, 1),
            ("not calibrated", lambda ref: None, {"flags": ACTIVE}, 1),
            ("diverged attitude", lambda ref: None, {"pitch": 2.2}, 1),
            # The PC's held pose is followed until the line.
            ("STALE too long", lambda ref: None, {"flags": ENGAGED | STALE, "joint_offset": 0.1},
             round(MAX_STALE_FLAG_S / 0.02) + 2),
            ("HELD too long", lambda ref: None, {"flags": ENGAGED | HELD, "joint_offset": 0.1},
             round(MAX_HELD_FLAG_S / 0.02) + 2),
        ):
            with self.subTest(name):
                move, ref, session = started_move()
                _, t = run(move, ref, 0.0, 1.5, joint_offset=0.1)
                before(ref)
                if ticks > 1:
                    _, t = run(move, ref, t, t + 0.02 * (ticks - 2), **frame)
                    self.assertEqual(move.phase, Phase.TRACK)
                _, t = run(move, ref, t, t + 0.04 if ticks > 1 else t + 0.02, **frame)
                self.assertEqual(move.phase, Phase.HOLD, move.stop_reason)
                # Never the stale or diverged frame: the last one followed.
                self.assertAlmostEqual(float(session.inputs[-1][0, -1, 0]),
                                       NEUTRAL_POSE[OBSERVATION_DOF_ORDER[0]] + 0.1, places=5)
                gravity_z = float(session.inputs[-1][0, -1, wc.term_slices()["ref_projected_gravity_b"]][2])
                self.assertLess(gravity_z, -0.9)

    def test_a_falling_trunk_holds_the_measured_pose(self):
        move, ref, session = started_move(action=0.5)
        run(move, ref, 0.0, 0.2)
        calls = len(session.inputs)
        command = MotorCommand()
        move.step(tick_obs(0.22, gravity=[0.9, 0.0, -0.3]), command)
        self.assertEqual(len(session.inputs), calls)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(command.target_angles[name], NEUTRAL_POSE[name])

    def test_a_restart_from_the_pc_aligns_again_and_rebuilds_the_history(self):
        move, ref, session = started_move()
        _, t = run(move, ref, 0.0, 1.5, yaw=0.0)
        self.assertEqual(move.phase, Phase.TRACK)
        _, t = run(move, ref, t, t + 0.02, yaw=0.8, flags=ENGAGED | RESTART, joint_offset=0.3)
        self.assertEqual(move.phase, Phase.SETTLE)
        history = session.inputs[-1][0]
        self.assertTrue(np.array_equal(history, np.repeat(history[-1:], 10, axis=0)))
        orientation = history[-1, wc.term_slices()["ref_anchor_ori_b"]]
        np.testing.assert_allclose(orientation, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)
        _, t = run(move, ref, t, t + 0.1, yaw=0.8, joint_offset=0.3)
        self.assertEqual(move.phase, Phase.SETTLE)

    def test_a_stopped_activation_is_not_started_again(self):
        move, ref, _ = started_move()
        _, t = run(move, ref, 0.0, 1.0, flags=ENGAGED)
        _, t = run(move, ref, t, t + 0.02, flags=CALIBRATED)  # the PC stops
        self.stop_and_hand_off(move, ref, t, flags=CALIBRATED)
        move.state = MoveState.INACTIVE
        self.assertIn("not tracking", move.start_refusal())
        ref.send()  # started again on the PC
        self.assertIsNone(move.start_refusal())
        move, ref, _ = started_move()
        move.refuse("test")
        self.assertIn("waiting for the PC", move.start_refusal())

    def test_no_start_without_a_fresh_reference(self):
        ref = FakeReceiver()
        move = WholebodyMove(ref, session=Session(metadata()))
        with self.assertRaises(RuntimeError):
            move.on_start(tick_obs(0.0), MotorCommand())
        ref.send()
        ref.time = 0.5
        with self.assertRaisesRegex(RuntimeError, "no reference frame"):
            move.on_start(tick_obs(0.5), MotorCommand())

    def test_the_reference_heading_is_turned_to_the_imus(self):
        ref = FakeReceiver()
        session = Session(metadata())
        move = WholebodyMove(ref, session=session)
        ref.send(yaw=1.0)
        robot = list(quat_mul(quat_from_yaw(-2.0), HOME_ROOT_QUAT_WXYZ))
        move.state = MoveState.STARTING
        move.on_start(tick_obs(0.0, quat=robot), MotorCommand())
        ref.time = 0.02
        ref.send(yaw=1.0)
        move.step(tick_obs(0.02, quat=robot), MotorCommand())
        orientation = session.inputs[-1][0, -1, wc.term_slices()["ref_anchor_ori_b"]]
        np.testing.assert_allclose(orientation, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)


class OverlayYieldTest(unittest.TestCase):
    """While whole-body tracking runs, the arm overlay and the HMD head move write nothing."""

    def test_arms_and_neck_yield_then_resume_from_the_measured_pose(self):
        claim = WholebodyClaim(active=True)
        arms = PicoArmTrackingMove(wholebody_claim=claim)
        head = HmdHeadTrackingMove(wholebody_claim=claim)
        obs = tick_obs(0.0, moves=("walk", "pico_arms", "hmd_head"))
        obs.user_input.head_orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.8}
        for move in (arms, head):
            move.state = MoveState.ACTIVE
            command = MotorCommand(target_angles={})
            move.step(obs, command)
            self.assertEqual(command.target_angles, {})
        claim.active = False
        command = MotorCommand(target_angles={})
        head.step(tick_obs(0.02, moves=("walk", "hmd_head")), command)
        self.assertAlmostEqual(command.target_angles["head"], NEUTRAL_POSE["head"], places=6)  # from measured, slewed


class StandingMove(Move):
    def __init__(self):
        super().__init__()
        self.starts = 0

    def can_balance(self, user_input):
        return True

    def on_start(self, obs, command):
        self.starts += 1
        self.state = MoveState.ACTIVE

    def step(self, obs, command):
        command.target_angles["probe"] = 1.0


class SelectorTest(unittest.TestCase):
    """The current PICO bridge sends pico_teleop; the PC's ACTIVE flag starts tracking."""

    def selector(self):
        ref = FakeReceiver()
        move = WholebodyMove(ref, session=Session(metadata()))
        standing = StandingMove()
        selector = PolicySelectableWalkMove(pico_move=standing, wholebody_move=move)
        selector.preload()
        return selector, move, ref, standing

    def tick(self, selector, ref, t, *, policy="pico_teleop", send=True, **frame):
        ref.time = t
        if send:
            ref.send(**frame)
        obs = tick_obs(t, balance_only=True, policy=policy)
        command = MotorCommand()
        if selector.state == MoveState.INACTIVE:
            selector.state = MoveState.STARTING
        if selector.state == MoveState.STARTING:
            selector.on_start(obs, command)
        else:
            selector.step(obs, command)
        return command

    def run_until(self, selector, ref, t, end, until=None, **kwargs):
        while t < end and not (until and until()):
            t += 0.02
            self.tick(selector, ref, t, **kwargs)
        return t

    def test_the_pc_starts_and_stops_tracking(self):
        selector, move, ref, standing = self.selector()
        self.tick(selector, ref, 0.0, flags=CALIBRATED)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.tick(selector, ref, 0.02)
        self.assertEqual(selector.effective_policy, "wholebody")
        t = self.run_until(selector, ref, 0.02, 2.0)
        self.assertEqual(move.phase, Phase.TRACK)
        # The PC stops: hold, HOME, then the standing policy.
        t = self.run_until(selector, ref, t, 7.0, lambda: selector.effective_policy != "wholebody", flags=CALIBRATED)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertGreater(t, 2.0 + STOP_HOLD_S + RETURN_S)
        self.assertIn("not tracking", selector.fallback_reason)
        # Started again on the PC: tracking again.
        self.tick(selector, ref, t + 0.02)
        self.assertEqual(selector.effective_policy, "wholebody")

    def test_a_stale_stream_stands_until_the_pc_starts_again(self):
        selector, move, ref, standing = self.selector()
        t = 0.0
        self.tick(selector, ref, t)
        self.assertEqual(selector.effective_policy, "wholebody")
        t = self.run_until(selector, ref, t, 1.0)
        t = self.run_until(selector, ref, t, 6.0, lambda: selector.effective_policy != "wholebody", send=False)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertIn("no reference frame", selector.fallback_reason)
        # The stream is back, but this activation was stopped: still standing.
        t = self.run_until(selector, ref, t, t + 0.2)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.tick(selector, ref, t + 0.02, flags=ENGAGED | RESTART)
        self.assertEqual(selector.effective_policy, "wholebody")

    def test_walk_on_the_wire_stops_tracking(self):
        selector, move, ref, standing = self.selector()
        self.tick(selector, ref, 0.0, policy="wholebody")
        t = self.run_until(selector, ref, 0.0, 1.0, policy="wholebody")
        self.tick(selector, ref, t + 0.02, policy="walk")
        self.assertEqual(move.stop_reason, "walk requested")

    def test_policy_off_and_on_does_not_resume_the_activation(self):
        selector, move, ref, standing = self.selector()
        t = self.run_until(selector, ref, -0.02, 1.0)
        self.assertEqual(selector.effective_policy, "wholebody")
        selector.state = MoveState.INACTIVE  # the scheduler's gate: R3 off
        self.tick(selector, ref, t + 0.02)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertIn("waiting for the PC", move.start_refusal())
        self.tick(selector, ref, t + 0.04, flags=ENGAGED | RESTART)
        self.assertEqual(selector.effective_policy, "wholebody")

    def test_no_reference_stands(self):
        selector, move, ref, standing = self.selector()
        self.tick(selector, ref, 1.0, send=False, policy="wholebody")
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertIn("no reference frame", selector.fallback_reason)
        self.assertTrue(selector.can_balance(UserInput(locomotion_policy="wholebody", balance_only=True)))

    def test_an_inference_failure_stands_and_the_arms_and_head_return_to_the_overlays(self):
        claim = WholebodyClaim()
        ref = FakeReceiver()
        move = WholebodyMove(ref, session=Session(metadata()))
        selector = PolicySelectableWalkMove(pico_move=StandingMove(), wholebody_move=move, wholebody_claim=claim)
        selector.preload()
        self.tick(selector, ref, 0.0)
        self.assertTrue(claim.active)
        move._session.action = lambda _: math.nan
        self.tick(selector, ref, 0.02)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertIn("whole-body inference", selector.fallback_reason)
        self.assertFalse(claim.active)
        self.tick(selector, ref, 0.04)
        self.assertEqual(selector.effective_policy, "pico_teleop")


if __name__ == "__main__":
    unittest.main()
