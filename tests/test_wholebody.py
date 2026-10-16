"""Whole-body policy: contract, reference wire, its own standing reference, the join, the stop sequence and policy
selection.

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
    HOME_ROOT_POS_Z_M,
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
    LATE,
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
    COAST_S,
    DT,
    ENTER_S,
    FALL_LAG_S,
    HEADING_STEP_RAD,
    JOIN_S,
    MAX_HELD_FLAG_S,
    MAX_SEQUENCE_GAP,
    MAX_STALE_FLAG_S,
    RETURN_S,
    STILL_TICKS,
    STOP_HOLD_S,
    Phase,
    Reference,
    WholebodyMove,
    coast_reference,
    quat_conj,
    quat_from_yaw,
    quat_mul,
    rotvec_from_quat,
    yaw_of,
)
from observer import Observation, RobotState
from policy_contract import PolicyContractError, PolicySelfTestError
from policy_fixtures import JOINT_NAMES, home_pose_stamp, csv
from test_getup_balance import Harness, sec
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


def frame_arrays(*, pitch=HOME_TRUNK_PITCH_RAD, height=0.17, joint_offset=0.0, yaw=0.0, vel=0.0, turn=0.0,
                 neck=(0.0, 0.0, 0.0)):
    quat = quat_mul(quat_from_yaw(yaw), [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    joints = np.array([NEUTRAL_POSE[n] for n in ROBOT_JOINTS]) + joint_offset
    joints[:3] = neck  # head, neck_roll, neck_pitch (HOME is 0)
    others = np.zeros((21, 3))
    return {
        "joint_pos": joints, "joint_vel": np.full(21, vel),
        "body_pos_w": np.vstack([[0.0, 0.0, height], others]),
        "body_quat_w": np.vstack([quat, np.tile([1.0, 0.0, 0.0, 0.0], (21, 1))]),
        "body_lin_vel_w": np.vstack([[vel, 0.0, 0.0], others]),
        "body_ang_vel_w": np.vstack([[0.0, 0.0, vel + turn], others]),
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

    def test_a_pc_that_jumps_out_of_held_is_not_heard(self):
        """Version 1 is the PC that sends the held pose, then the live one in one frame: no frame of it is used."""
        receiver = ReferenceReceiver(port=1, clock=lambda: 0.0)
        old = lambda seq, flags, **frame: encode_frame(seq, 0.02 * seq, frame_arrays(**frame), flags)[:4] + (
            1).to_bytes(2, "little") + encode_frame(seq, 0.02 * seq, frame_arrays(**frame), flags)[6:]
        with mock.patch("builtins.print") as said:
            for seq in range(3):
                self.assertFalse(receiver.accept(old(seq, ENGAGED | HELD), 0.02 * seq))
            self.assertFalse(receiver.accept(old(3, ENGAGED, joint_offset=1.0), 0.06))  # HELD cleared: the jump
        snapshot = receiver.read()
        self.assertIsNone(snapshot.frame)
        self.assertEqual((snapshot.activation, receiver.rejected), (0, 4))
        self.assertEqual(said.call_count, 1)
        self.assertIn("unsupported version 1", said.call_args.args[0])
        self.assertTrue(receiver.accept(encode_frame(4, 0.08, frame_arrays(), ENGAGED), 0.08))

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


def pitched(pitch, yaw=0.0):
    """(quat, projected gravity) of a trunk at heading ``yaw`` pitched ``pitch`` about y."""
    quat = list(quat_mul(quat_from_yaw(yaw), [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]))
    return quat, [math.sin(pitch), 0.0, -math.cos(pitch)]


def tick_obs(t, *, gravity=None, quat=None, moves=("walk",), balance_only=True, policy="pico_teleop",
             joint_speed=0.0, positions=None):
    quat = list(HOME_ROOT_QUAT_WXYZ) if quat is None else quat
    return Observation(
        robot_state=RobotState(
            time_s=t, gyro=[0.0, 0.0, 0.0], body_quat=quat,
            projected_gravity=list(HOME_PROJECTED_GRAVITY) if gravity is None else gravity,
            motor_positions=dict(NEUTRAL_POSE) if positions is None else positions,
            motor_velocities={n: joint_speed for n in NEUTRAL_POSE},
        ),
        user_input=UserInput(active_moves=set(moves), locomotion_policy=policy, balance_only=balance_only),
    )


def started_move(action=0.01, obs=None):
    ref = FakeReceiver()
    session = Session(metadata(), action=action)
    move = WholebodyMove(ref, session=session)
    move.state = MoveState.STARTING
    move.on_start(obs or tick_obs(0.0), MotorCommand())
    return move, ref, session


def run(move, ref, start, end, send=True, until=None, obs=None, **frame):
    """Ticks from ``start`` to ``end`` (s), the PC sending a frame each tick when ``send``; ``obs(t)`` the robot."""
    command, t = MotorCommand(), start
    for k in range(round((end - start) / DT)):
        t = start + DT * (k + 1)
        ref.time = t
        if send:
            ref.send(**frame)
        command = MotorCommand()
        move.step(obs(t) if obs else tick_obs(t), command)
        if until is not None and until():
            break
    return command, t


def terms(session, index=-1):
    """The newest frame's terms of the ``index``-th policy input."""
    frame = session.inputs[index][0, -1]
    return {name: frame[s] for name, s in wc.term_slices().items()}


HOME_18 = np.array([NEUTRAL_POSE[n] for n in OBSERVATION_DOF_ORDER])


def to_track(move, ref, t=0.0, **frame):
    """Ticks (the PC sending ``frame``) until the move tracks."""
    _, t = run(move, ref, t, t + 4.0, until=lambda: move.phase == Phase.TRACK, **frame)
    assert move.phase == Phase.TRACK, move.phase
    return t


class MoveTest(unittest.TestCase):
    def test_r3_stands_on_its_own_home_reference_without_a_pc(self):
        move, ref, session = started_move()
        _, t = run(move, ref, 0.0, ENTER_S - 2 * DT, send=False)
        self.assertEqual(move.phase, Phase.ENTER)
        _, t = run(move, ref, t, t + 0.1, send=False)
        self.assertEqual(move.phase, Phase.STAND)
        first = session.inputs[0][0]
        self.assertTrue(np.array_equal(first, np.repeat(first[-1:], wc.HISTORY_LENGTH, axis=0)))
        self.assertEqual(float(np.abs(first[-1, wc.term_slices()["actions"]]).max()), 0.0)
        now = terms(session)
        np.testing.assert_allclose(now["ref_joint_pos"], HOME_18, atol=1e-6)
        np.testing.assert_allclose(now["ref_anchor_ori_b"], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(now["ref_projected_gravity_b"], HOME_PROJECTED_GRAVITY, atol=1e-6)
        self.assertAlmostEqual(float(now["ref_anchor_height"][0]), HOME_ROOT_POS_Z_M, places=6)
        for name in ("ref_joint_vel", "ref_anchor_lin_vel_b", "ref_anchor_ang_vel_b"):
            self.assertEqual(float(np.abs(now[name]).max()), 0.0, name)

    def test_enter_goes_smoothly_from_the_measured_pose_to_home(self):
        quat, gravity = pitched(HOME_TRUNK_PITCH_RAD + math.radians(8.0))
        bent = {n: v + 0.3 for n, v in NEUTRAL_POSE.items()}
        obs = lambda t: tick_obs(t, quat=quat, gravity=gravity, positions=bent)  # noqa: E731
        move, ref, session = started_move(obs=obs(0.0))
        refs = []
        for k in range(round(ENTER_S / DT) + 2):
            run(move, ref, k * DT, (k + 1) * DT, send=False, obs=obs)
            current = move._current
            refs.append((current.joint_pos.copy(), current.joint_vel.copy(), current.quat.copy(), current.ang_vel.copy()))
        # Starts at the measured pose, ends at HOME (one tick ahead), no step over the blend's slope.
        self.assertLess(float(np.abs(refs[0][0] - [bent[n] for n in wc.OBSERVATION_JOINT_NAMES]).max()), 0.01)
        np.testing.assert_allclose(refs[-1][0][3:], [NEUTRAL_POSE[n] for n in wc.OBSERVATION_JOINT_NAMES][3:])
        steps = np.abs(np.diff([r[0] for r in refs], axis=0)).max()
        self.assertLessEqual(steps, 0.3 * 1.5 * DT / ENTER_S + 1e-9)
        # The velocities are the poses' own (central differences).
        for k in range(1, len(refs) - 2):
            np.testing.assert_allclose(refs[k][1], (refs[k + 1][0] - refs[k - 1][0]) / (2 * DT), atol=0.02)
            turn = rotvec_from_quat(quat_mul(refs[k + 1][2], quat_conj(refs[k - 1][2]))) / (2 * DT)
            np.testing.assert_allclose(refs[k][3], turn, atol=0.01)

    def test_the_stand_heading_follows_the_imu_but_the_reference_does_not_turn(self):
        move, ref, session = started_move()
        _, t = run(move, ref, 0.0, ENTER_S + 0.1, send=False)
        turned = list(quat_mul(quat_from_yaw(0.5), HOME_ROOT_QUAT_WXYZ))
        _, t = run(move, ref, t, t + 10 * DT, send=False, obs=lambda t: tick_obs(t, quat=turned))
        self.assertAlmostEqual(yaw_of(move._current.quat), 10 * HEADING_STEP_RAD, places=9)
        self.assertEqual(float(np.abs(terms(session)["ref_anchor_ang_vel_b"]).max()), 0.0)
        _, t = run(move, ref, t, t + 20 * DT, send=False, obs=lambda t: tick_obs(t, quat=turned))
        self.assertAlmostEqual(yaw_of(move._current.quat), 0.5, places=9)
        np.testing.assert_allclose(terms(session)["ref_anchor_ori_b"], [1, 0, 0, 1, 0, 0], atol=1e-6)
        # Tilted more than HEADING_FREEZE_TILT_DEG from HOME: the heading stays.
        quat, gravity = pitched(HOME_TRUNK_PITCH_RAD + math.radians(40.0), yaw=-0.5)
        run(move, ref, t, t + 10 * DT, send=False, obs=lambda t: tick_obs(t, quat=quat, gravity=gravity))
        self.assertAlmostEqual(yaw_of(move._current.quat), 0.5, places=9)
        # The IMU's heading drifting 0.5 deg/s: the policy sees no heading difference.
        move, ref, session = started_move()
        drift = lambda t: tick_obs(t, quat=list(quat_mul(quat_from_yaw(math.radians(0.5) * t), HOME_ROOT_QUAT_WXYZ)))  # noqa: E731
        run(move, ref, 0.0, 10.0, send=False, obs=drift)
        np.testing.assert_allclose(terms(session)["ref_anchor_ori_b"], [1, 0, 0, 1, 0, 0], atol=1e-6)

    def test_the_join_waits_for_the_robot_and_the_operator_to_stand_still(self):
        tilted = pitched(HOME_TRUNK_PITCH_RAD + math.radians(13.0))
        for name, robot, frame in (
            ("legs moving", {"joint_speed": 1.2}, {}),
            ("trunk off HOME", {"quat": tilted[0], "gravity": tilted[1]}, {}),
            ("operator moving", {}, {"vel": 0.1}),
            ("operator turning", {}, {"turn": 0.4}),
            ("operator bent", {}, {"pitch": math.radians(40.0)}),
            ("operator crouching", {}, {"height": HOME_ROOT_POS_Z_M - 0.03}),
            ("PC's STALE", {}, {"flags": ENGAGED | STALE}),
            ("PC's HELD", {}, {"flags": ENGAGED | HELD}),
            ("PC's LATE", {}, {"flags": ENGAGED | LATE}),
            ("PC not started", {}, {"flags": CALIBRATED}),
        ):
            with self.subTest(name):
                move, ref, _ = started_move()
                run(move, ref, 0.0, 3.0, obs=lambda t: tick_obs(t, **robot), **frame)
                self.assertEqual(move.phase, Phase.STAND)
        with self.subTest("walk asked for"):
            move, ref, _ = started_move()
            move.joins_allowed = False
            run(move, ref, 0.0, 3.0)
            self.assertEqual(move.phase, Phase.STAND)
            move.joins_allowed = True
            _, t = run(move, ref, 3.0, 3.0 + DT)
            self.assertEqual(move.phase, Phase.JOIN)
        with self.subTest("all still: 25 ticks of standing still"):
            move, ref, _ = started_move()
            _, t = run(move, ref, 0.0, 3.0, until=lambda: move.phase == Phase.STAND)
            stand = t
            _, t = run(move, ref, t, t + 1.0, until=lambda: move.phase == Phase.JOIN)
            self.assertEqual(move.phase, Phase.JOIN)
            self.assertEqual(round((t - stand) / DT), STILL_TICKS - 1)

    def test_the_join_keeps_the_history_and_blends_into_the_pcs_reference(self):
        move, ref, session = started_move(action=0.02)
        frame = {"yaw": 1.0, "joint_offset": 0.1, "height": 0.165}
        _, t = run(move, ref, 0.0, 3.0, until=lambda: move.phase == Phase.JOIN, **frame)
        joined = len(session.inputs) - 1
        before, at = session.inputs[joined - 1][0], session.inputs[joined][0]
        np.testing.assert_array_equal(at[:-1], before[1:])  # not rebuilt
        self.assertAlmostEqual(float(at[-1, wc.term_slices()["actions"]][0]), 0.02, places=6)
        _, t = run(move, ref, t, t + JOIN_S, until=lambda: move.phase == Phase.TRACK, **frame)
        refs = [i[0, -1, wc.term_slices()["ref_joint_pos"]] for i in session.inputs[joined:]]
        self.assertLessEqual(float(np.abs(np.diff(refs, axis=0)).max()), 0.1 * 1.875 * DT / JOIN_S + 1e-6)
        now = terms(session)
        np.testing.assert_allclose(now["ref_joint_pos"], HOME_18 + 0.1, atol=1e-6)
        # Turned to the stand's heading (the robot's): the PC's yaw is gone.
        np.testing.assert_allclose(now["ref_anchor_ori_b"], [1, 0, 0, 1, 0, 0], atol=1e-6)
        self.assertAlmostEqual(float(now["ref_anchor_height"][0]), 0.165, places=6)

    def test_the_join_velocities_are_the_blends_own(self):
        move, ref, _ = started_move()
        frame = {"yaw": 0.6, "pitch": HOME_TRUNK_PITCH_RAD + 0.3, "joint_offset": 0.2, "height": 0.16}
        _, t = run(move, ref, 0.0, 3.0, until=lambda: move.phase == Phase.JOIN, **frame)
        refs = []
        while move.phase == Phase.JOIN:
            _, t = run(move, ref, t, t + DT, **frame)
            c = move._current
            refs.append((c.joint_pos.copy(), c.joint_vel.copy(), c.quat.copy(), c.ang_vel.copy(), c.height,
                         c.lin_vel.copy()))
        for k in range(1, len(refs) - 2):
            np.testing.assert_allclose(refs[k][1], (refs[k + 1][0] - refs[k - 1][0]) / (2 * DT), atol=0.01)
            turn = rotvec_from_quat(quat_mul(refs[k + 1][2], quat_conj(refs[k - 1][2]))) / (2 * DT)
            np.testing.assert_allclose(refs[k][3], turn, atol=0.01)
            self.assertAlmostEqual(refs[k][5][2], (refs[k + 1][4] - refs[k - 1][4]) / (2 * DT), delta=0.002)

    def test_every_stop_coasts_holds_returns_and_stands(self):
        def restart(ref):
            ref.seq = 0
            ref.sent_offset += 60.0

        for name, before, frame, ticks in (
            ("lost frames", lambda ref: setattr(ref, "seq", ref.seq + MAX_SEQUENCE_GAP + 1), {}, 1),
            ("restarted PC", restart, {}, 1),
            ("the PC stopped", lambda ref: None, {"flags": CALIBRATED}, 1),
            ("not calibrated", lambda ref: None, {"flags": ACTIVE}, 1),
            ("a new start", lambda ref: ref.send(flags=ENGAGED | RESTART), {}, 1),
            ("STALE too long", lambda ref: None, {"flags": ENGAGED | STALE}, round(MAX_STALE_FLAG_S / DT) + 2),
            ("HELD too long", lambda ref: None, {"flags": ENGAGED | HELD}, round(MAX_HELD_FLAG_S / DT) + 2),
        ):
            with self.subTest(name):
                move, ref, session = started_move()
                t = to_track(move, ref)  # joined on a still operator, then moving
                _, t = run(move, ref, t, t + 0.2, vel=0.5)
                before(ref)
                if ticks > 1:
                    _, t = run(move, ref, t, t + DT * (ticks - 2), vel=0.5, **frame)
                    self.assertEqual(move.phase, Phase.TRACK)
                    _, t = run(move, ref, t, t + DT, vel=0.5, **frame)
                last = move._current
                _, t = run(move, ref, t, t + DT, vel=0.5, **frame)
                self.assertEqual(move.phase, Phase.COAST, move.stop_reason)
                stopped = t
                phases, refs = [], []
                while t < stopped + COAST_S + STOP_HOLD_S + RETURN_S:
                    _, t = run(move, ref, t, t + DT, vel=0.5, **frame)
                    phases.append(move.phase)
                    refs.append(move._current.joint_pos.copy())
                changes = [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]
                self.assertEqual(changes, [Phase.COAST, Phase.HOLD, Phase.RETURN, Phase.STAND])
                hold = phases.index(Phase.HOLD)
                self.assertAlmostEqual(hold * DT, COAST_S - 2 * DT, delta=DT + 1e-9)
                self.assertAlmostEqual(phases.index(Phase.RETURN) * DT, STOP_HOLD_S - DT, delta=DT + 1e-9)
                # The coast goes on at the last speed (0.5 rad/s), fading to zero: 0.05 rad further.
                np.testing.assert_allclose(refs[hold][3:] - last.joint_pos[3:], 0.5 * COAST_S / 2, atol=1e-6)
                self.assertLessEqual(float(np.abs(np.diff(refs, axis=0)).max()), 0.5 * DT + 1e-6 + 0.1 * 1.5 * DT / RETURN_S)
                np.testing.assert_allclose(refs[-1][3:], [NEUTRAL_POSE[n] for n in wc.OBSERVATION_JOINT_NAMES][3:])
                # Standing again; that start is not joined again.
                self.assertEqual(move.phase, Phase.STAND)

    def test_a_joined_start_is_not_joined_again_and_the_next_one_is(self):
        move, ref, _ = started_move()
        t = to_track(move, ref)
        _, t = run(move, ref, t, t + 0.2, send=False)  # stopped: no frame
        _, t = run(move, ref, t, t + 4.0, flags=ENGAGED)  # the frames are back, the same start
        self.assertEqual(move.phase, Phase.STAND)
        _, t = run(move, ref, t, t + DT, flags=CALIBRATED)  # stopped on the PC
        _, t = run(move, ref, t, t + 1.0, until=lambda: move.phase == Phase.JOIN, flags=ENGAGED)  # started again
        self.assertEqual(move.phase, Phase.JOIN)

    def test_a_trunk_off_the_references_attitude_enters_again(self):
        for name, setup, flags in (
            ("tracking", lambda move, ref, t: t, ENGAGED),
            ("holding", lambda move, ref, t: run(move, ref, t, t + 0.3, flags=CALIBRATED)[1], CALIBRATED),
        ):
            with self.subTest(name):
                move, ref, session = started_move()
                t = to_track(move, ref)
                t = setup(move, ref, t)
                if name == "holding":
                    self.assertEqual(move.phase, Phase.HOLD)
                    move._hold_ref = move._hold_ref.__class__(  # the held reference bent past 90 deg
                        move._hold_ref.joint_pos, move._hold_ref.joint_vel, pitched(2.2)[0],
                        np.zeros(3), np.zeros(3), move._hold_ref.height)
                    _, t = run(move, ref, t, t + DT, flags=CALIBRATED)
                else:
                    _, t = run(move, ref, t, t + DT, pitch=2.2)
                self.assertEqual(move.phase, Phase.ENTER)
                history = session.inputs[-1][0]
                self.assertTrue(np.array_equal(history, np.repeat(history[-1:], wc.HISTORY_LENGTH, axis=0)))
                self.assertEqual(float(np.abs(history[-1, wc.term_slices()["actions"]]).max()), 0.0)
                _, t = run(move, ref, t, t + 4.0, flags=flags)
                self.assertEqual(move.phase, Phase.STAND)  # that start is done

    def test_the_neck_is_the_pcs_while_following_and_home_otherwise(self):
        move, ref, _ = started_move()
        neck = lambda command: [command.target_angles[n] for n in ("head", "neck_roll", "neck_pitch")]  # noqa: E731
        command, t = run(move, ref, 0.0, 1.0, neck=(0.6, 0.9, -0.3))
        np.testing.assert_allclose(neck(command), [0.0, 0.0, 0.0], atol=1e-9)
        self.assertFalse(move.drives_neck)
        _, t = run(move, ref, t, 3.0, until=lambda: move.phase == Phase.JOIN, neck=(0.6, 0.9, -0.3))
        self.assertTrue(move.drives_neck)
        command, t = run(move, ref, t, t + 1.5, neck=(0.6, 0.9, -0.3))
        # Slewed at 2.5 rad/s toward the reference's neck, clamped to the contract's limits (neck_roll +-0.4).
        np.testing.assert_allclose(neck(command), [0.6, 0.4, -0.3], atol=1e-9)
        _, t = run(move, ref, t, t + DT, flags=CALIBRATED)
        self.assertFalse(move.drives_neck)
        command, t = run(move, ref, t, t + STOP_HOLD_S + RETURN_S + 0.1, flags=CALIBRATED)
        np.testing.assert_allclose(neck(command), [0.0, 0.0, 0.0], atol=1e-9)

    def test_a_falling_trunk_holds_the_measured_pose(self):
        move, ref, session = started_move(action=0.5)
        run(move, ref, 0.0, 0.2)
        calls = len(session.inputs)
        command = MotorCommand()
        move.step(tick_obs(0.22, gravity=[0.9, 0.0, -0.3]), command)
        self.assertEqual(len(session.inputs), calls)
        for name in OBSERVATION_DOF_ORDER:
            self.assertEqual(command.target_angles[name], NEUTRAL_POSE[name])

    def test_a_safety_hold_enters_again_from_the_measured_pose(self):
        move, ref, session = started_move()
        t = to_track(move, ref)
        move.on_safety_resume(tick_obs(t))
        _, t = run(move, ref, t, t + DT)
        self.assertEqual(move.phase, Phase.ENTER)
        history = session.inputs[-1][0]
        self.assertTrue(np.array_equal(history, np.repeat(history[-1:], wc.HISTORY_LENGTH, axis=0)))
        _, t = run(move, ref, t, t + 4.0)
        self.assertEqual(move.phase, Phase.STAND)

    def test_the_coast_does_not_rise(self):
        last = Reference(np.zeros(21), np.zeros(21), [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.8], np.zeros(3), 0.15)
        heights = [coast_reference(last, tau).height for tau in np.linspace(0.0, COAST_S, 11)]
        self.assertLessEqual(max(heights), HOME_ROOT_POS_Z_M + 1e-12)
        high = Reference(np.zeros(21), np.zeros(21), [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.8], np.zeros(3), 0.2)
        self.assertEqual(coast_reference(high, COAST_S).height, 0.2)


class OverlayYieldTest(unittest.TestCase):
    """The arm overlay yields while the whole-body policy runs; the HMD head move only while it follows the PC."""

    def test_arms_yield_to_the_policy_and_the_neck_to_the_pcs_reference(self):
        claim = WholebodyClaim(active=True)
        arms = PicoArmTrackingMove(wholebody_claim=claim)
        head = HmdHeadTrackingMove(wholebody_claim=claim)
        obs = tick_obs(0.0, moves=("walk", "pico_arms", "hmd_head"))
        obs.user_input.head_orientation = {"roll": 0.0, "pitch": 0.0, "yaw": 0.8}
        for move in (arms, head):
            command = MotorCommand(target_angles={})
            move.state = MoveState.STARTING
            move.on_start(obs, command)
            self.assertEqual(move.state, MoveState.ACTIVE)
            if move is arms:
                self.assertEqual(command.target_angles, {})  # started while the policy drives the arms
            command = MotorCommand(target_angles={})
            move.step(obs, command)
            if move is arms:
                self.assertEqual(command.target_angles, {})
            else:
                self.assertIn("head", command.target_angles)  # standing on its own reference: the HMD's
        claim.neck = True
        for start in (False, True):
            command = MotorCommand(target_angles={})
            if start:
                head = HmdHeadTrackingMove(wholebody_claim=claim)
                head.on_start(obs, command)
            else:
                head.step(obs, command)
            self.assertEqual(command.target_angles, {})
        claim.active = claim.neck = False
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
    """With the whole-body policy installed it stands the robot from R3 on; the PICO policy is not started."""

    def selector(self, claim=None):
        ref = FakeReceiver()
        move = WholebodyMove(ref, session=Session(metadata()))
        standing = StandingMove()
        selector = PolicySelectableWalkMove(pico_move=standing, wholebody_move=move,
                                            wholebody_claim=claim or WholebodyClaim())
        selector.preload()
        return selector, move, ref, standing

    def tick(self, selector, ref, t, *, policy="pico_teleop", send=True, **frame):
        ref.time = t
        if send:
            ref.send(**frame)
        obs = tick_obs(t, policy=policy)
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
            t += DT
            self.tick(selector, ref, t, **kwargs)
        return t

    def test_the_policy_stands_tracks_and_stands_again_without_pico(self):
        claim = WholebodyClaim()
        selector, move, ref, standing = self.selector(claim)
        self.tick(selector, ref, 0.0, flags=CALIBRATED)
        self.assertEqual((selector.effective_policy, move.phase), ("wholebody", Phase.ENTER))
        self.assertTrue(claim.active)
        self.assertFalse(claim.neck)
        t = self.run_until(selector, ref, 0.0, 3.0, flags=CALIBRATED)
        self.assertEqual(move.phase, Phase.STAND)
        t = self.run_until(selector, ref, t, t + 3.0, lambda: move.phase == Phase.TRACK)
        self.assertTrue(claim.neck)
        for name, kwargs in (("stopped on the PC", {"flags": CALIBRATED}), ("silent PC", {"send": False}),
                             ("walk on the wire", {"policy": "walk"})):
            t = self.run_until(selector, ref, t, t + 4.0, lambda: move.phase == Phase.STAND, **kwargs)
            self.assertEqual(move.phase, Phase.STAND, name)
            self.assertFalse(claim.neck)
            self.tick(selector, ref, t + DT, flags=CALIBRATED)
            t = self.run_until(selector, ref, t + DT, t + 3.0, lambda: move.phase == Phase.TRACK)
            self.assertEqual(move.phase, Phase.TRACK, name)
        self.assertEqual(standing.starts, 0)
        self.assertEqual(selector.effective_policy, "wholebody")

    def test_walk_on_the_wire_stops_tracking_and_does_not_join(self):
        selector, move, ref, standing = self.selector()
        self.tick(selector, ref, 0.0)
        t = self.run_until(selector, ref, 0.0, 4.0, lambda: move.phase == Phase.TRACK)
        self.tick(selector, ref, t + DT, policy="walk")
        self.assertEqual(move.stop_reason, "walk requested")
        self.tick(selector, ref, t + 2 * DT, policy="walk", flags=CALIBRATED)
        t = self.run_until(selector, ref, t + 2 * DT, t + 5.0, policy="walk")
        self.assertEqual(move.phase, Phase.STAND)
        self.assertEqual(standing.starts, 0)

    def test_it_balances_for_the_get_up_hand_back(self):
        selector, *_ = self.selector()
        for policy in ("pico_teleop", "wholebody", "walk"):
            self.assertTrue(selector.can_balance(UserInput(locomotion_policy=policy, balance_only=True)))

    def test_policy_off_and_on_enters_again_and_does_not_rejoin(self):
        selector, move, ref, standing = self.selector()
        self.tick(selector, ref, 0.0)
        t = self.run_until(selector, ref, 0.0, 4.0, lambda: move.phase == Phase.TRACK)
        for _ in range(2):  # the scheduler's gate (R3 off, B) resets the moves
            move.state = selector.state = MoveState.INACTIVE
            self.tick(selector, ref, t + DT)
            self.assertEqual((selector.effective_policy, move.phase), ("wholebody", Phase.ENTER))
            t = self.run_until(selector, ref, t + DT, t + 4.0)
            self.assertEqual(move.phase, Phase.STAND)
        self.tick(selector, ref, t + DT, flags=ENGAGED | RESTART)
        self.run_until(selector, ref, t + DT, t + 2.0, lambda: move.phase == Phase.JOIN)
        self.assertEqual(move.phase, Phase.JOIN)
        self.assertEqual(standing.starts, 0)

    def test_a_safety_hold_that_drops_walk_stops_and_starts_the_policy_again(self):
        """The network input drops ``walk`` while the scheduler holds for the IMU: on_stop, then on_start."""
        selector, move, ref, standing = self.selector()
        self.tick(selector, ref, 0.0)
        t = self.run_until(selector, ref, 0.0, 4.0, lambda: move.phase == Phase.TRACK)
        selector.on_safety_resume(tick_obs(t))
        selector.state = MoveState.STOPPING
        while selector.state != MoveState.INACTIVE:
            t += DT
            selector.on_stop(tick_obs(t, moves=()), MotorCommand())
        self.tick(selector, ref, t + DT)
        self.assertEqual((selector.effective_policy, move.phase), ("wholebody", Phase.ENTER))
        self.run_until(selector, ref, t + DT, t + 4.0)
        self.assertEqual(move.phase, Phase.STAND)  # the start it tracked is not joined again
        self.assertEqual(standing.starts, 0)

    def test_an_inference_failure_stands_on_pico_until_the_next_start(self):
        claim = WholebodyClaim()
        selector, move, ref, standing = self.selector(claim)
        self.tick(selector, ref, 0.0)
        t = self.run_until(selector, ref, 0.0, 4.0, lambda: move.phase == Phase.TRACK)
        self.assertTrue(claim.active and claim.neck)
        move._session.action = lambda _: math.nan
        self.tick(selector, ref, t + DT)
        self.assertEqual(selector.effective_policy, "pico_teleop")
        self.assertIn("whole-body inference", selector.fallback_reason)
        self.assertFalse(claim.active or claim.neck)
        self.tick(selector, ref, t + 2 * DT)
        self.assertEqual((selector.effective_policy, standing.starts), ("pico_teleop", 1))
        move._session.action = 0.01
        selector.state = MoveState.INACTIVE  # the next start (R3, a get-up's hand-back)
        self.tick(selector, ref, t + 3 * DT)
        self.assertEqual(selector.effective_policy, "wholebody")

    def test_without_the_whole_body_policy_pico_stands(self):
        standing = StandingMove()
        selector = PolicySelectableWalkMove(pico_move=standing)
        selector.preload()
        ref = FakeReceiver()
        self.tick(selector, ref, 0.0, send=False, policy="wholebody")
        self.assertEqual((selector.effective_policy, standing.starts), ("pico_teleop", 1))


def pitch_gravity(pitch):
    """The projected gravity of a trunk pitched ``pitch`` about y (HOME_TRUNK_PITCH_RAD: HOME's)."""
    return [math.sin(pitch), 0.0, -math.cos(pitch)]


BOW_START = 3.0  # tracking from 2.5 s: entered at R3 (1 s), stood still 0.5 s, joined (1 s)


def bow(t, deep=math.radians(75.0), start=BOW_START, height=0.17):
    """(trunk pitch (rad), trunk height): HOME, down to ``deep`` in 1 s, held 0.5 s, back up in 1 s."""
    home = HOME_TRUNK_PITCH_RAD
    phase = min(max(t - start, 0.0), 2.5)
    fraction = min(phase, 1.0) if phase < 1.5 else max(0.0, 2.5 - phase)
    return home + (deep - home) * fraction, 0.17 + (height - 0.17) * fraction


class FallWhileFollowingTest(unittest.TestCase):
    """The scheduler, the selector and the move: a reference bent past the fall angle is followed."""

    def run_scheduler(self, ref_pose, robot_pitch, seconds=7.0, stop_s=None):
        ref = FakeReceiver()
        session = Session(metadata())
        move = WholebodyMove(ref, session=session)
        standing = StandingMove()
        selector = PolicySelectableWalkMove(pico_move=standing, wholebody_move=move)
        selector.preload()
        h = Harness(selector)
        trace = []

        def script(tick, _inhibited):
            t = tick / 50.0
            ref.time = t
            pitch, height = ref_pose(t)
            ref.send(pitch=pitch, height=height, flags=CALIBRATED if stop_s and t >= stop_s else ENGAGED)
            trace.append((t, move.phase, selector.effective_policy, len(session.inputs)))
            return pitch_gravity(robot_pitch(t)), UserInput(
                active_moves={"walk"}, locomotion_policy="pico_teleop", balance_only=True
            )

        rows = h.run(script, sec(seconds))
        self.assertEqual(standing.starts, 0)
        return h, rows, trace

    def test_following_a_deep_bow_is_not_a_fall(self):
        # The robot follows 0.2 s late, past 60 deg for about 1 s.
        h, rows, trace = self.run_scheduler(bow, lambda t: bow(t - 0.2)[0])
        self.assertFalse(any(row["override"] for row in rows))
        self.assertEqual(h.getup.starts, 0)
        tracking = [(t, phase, policy, runs) for t, phase, policy, runs in trace if t >= BOW_START - 0.4]
        self.assertTrue(all(phase == Phase.TRACK and policy == "wholebody" for _, phase, policy, _ in tracking))
        # The policy ran every tick, the bow included (no measured-pose hold).
        runs = [runs for *_, runs in tracking]
        self.assertEqual(runs, list(range(runs[0], runs[0] + len(runs))))

    def test_still_a_fall(self):
        upright = lambda t: (HOME_TRUNK_PITCH_RAD, 0.17)  # noqa: E731
        for name, ref_pose, robot_pitch, fell_at in (
            # The reference never bends: the robot falling past 60 deg at 4 s.
            ("reference upright", upright, lambda t: math.radians(80.0) if t >= 4.0 else upright(t)[0], 4.0),
            # The robot stays bent after the reference came back above 60 deg.
            ("robot stays bent", bow, lambda t: bow(t - 0.2)[0] if t < BOW_START + 1.5 else math.radians(75.0),
             BOW_START + 1.5 + 15.0 / 65.0 + FALL_LAG_S),
        ):
            with self.subTest(name):
                h, rows, trace = self.run_scheduler(ref_pose, robot_pitch)
                first = next(row["tick"] for row in rows if row["override"]) / 50.0
                # Counted from the first fallen tick, 15 ticks (0.3 s) of debounce.
                self.assertAlmostEqual(first, fell_at + 0.3, delta=0.05)

    def test_a_stop_in_mid_bow_is_not_a_fall_but_one_on_the_floor_is(self):
        stop = BOW_START + 1.2
        for name, height, fell in (("bow", 0.14, False), ("on the floor", 0.06, True)):
            with self.subTest(name):
                pose = lambda t: bow(t, height=height)  # noqa: E731
                h, rows, trace = self.run_scheduler(pose, lambda t: bow(t - 0.2)[0], stop_s=stop)
                overrides = [row["tick"] / 50.0 for row in rows if row["override"]]
                if fell:
                    # The exemption ends at the stop: 15 ticks of debounce, the stop's tick and one of the input.
                    self.assertAlmostEqual(overrides[0], stop + 0.3 + DT, delta=0.03)
                else:
                    self.assertEqual(overrides, [])
                    self.assertEqual(h.getup.starts, 0)

    def test_a_robot_off_the_references_attitude_is_not_following_it(self):
        move, ref, _ = started_move()
        t = to_track(move, ref)
        _, t = run(move, ref, t, t + 1.0, pitch=math.radians(75.0))
        bent = pitch_gravity(math.radians(75.0))
        self.assertTrue(move.follows_fallen_reference(t, bent))
        # Gravity's z differs by more than the training's anchor_ori line.
        self.assertFalse(move.follows_fallen_reference(t, pitch_gravity(math.radians(140.0))))
        self.assertFalse(move.follows_fallen_reference(t + FALL_LAG_S + DT, bent))
        # Standing on its own reference (the measured attitude entered): never.
        move.on_safety_resume(tick_obs(t))
        bent_quat, _ = pitched(math.radians(75.0))
        move.step(tick_obs(t + DT, quat=bent_quat, gravity=bent), MotorCommand())
        self.assertFalse(move.follows_fallen_reference(t + DT, bent))


if __name__ == "__main__":
    unittest.main()
