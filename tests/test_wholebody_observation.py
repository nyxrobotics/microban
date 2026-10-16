"""The whole-body observation is the training's, value for value.

tests/fixtures/wholebody_observation.npz was recorded by
tools/wholebody_observation_fixture.py in the training environment itself
(mjlab_microban wholebody-tracking, play mode, sensor delays 0): per tick the
robot state, the reference and the actor's 118 values, for three pieces of
the PICO recordings run by a checkpoint's policy.  Here the robot runtime
rebuilds the values from the same state, with the reference passed through
the UDP wire format, and the IMU and the PC each in their own yaw.
"""

import math
import unittest
from pathlib import Path

import numpy as np

from imu_reader import quat_apply_inverse as observer_gravity
from input.input_source import UserInput
from input.wholebody_reference import ACTIVE, CALIBRATED, ReferenceSnapshot, decode_frame, encode_frame
from moves.move import MotorCommand, MoveState
from moves.wholebody import (
    ROBOT_JOINT_ORDER,
    ObservationHistory,
    Phase,
    Reference,
    WholebodyMove,
    build_frame,
    quat_from_yaw,
    quat_mul,
    yaw_alignment,
    yaw_of,
)
from observer import Observation, RobotState
from wholebody_contract import HISTORY_LENGTH, OBSERVATION_WIDTH, WholebodyContract

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wholebody_observation.npz"
# float32 state in, float32 values out: the training computes in float32.
ATOL = 2.0e-5


def load_fixture():
    data = np.load(FIXTURE)
    assert tuple(data["joint_names"]) == ROBOT_JOINT_ORDER
    return data


def wire_frame(data, piece: int, tick: int, pc_yaw: float, seq: int):
    """The reference of one tick as the PC would send it, in the PC's own yaw."""
    turn = quat_from_yaw(pc_yaw)

    def rotate(v):
        return quat_mul(quat_mul(turn, np.r_[0.0, v]), [turn[0], -turn[1], -turn[2], -turn[3]])[1:]

    quat = quat_mul(turn, data["ref_quat"][piece, tick])
    others = 21  # body 0 is the trunk; the other bodies check that only it is read
    arrays = {
        "joint_pos": data["ref_joint_pos"][piece, tick],
        "joint_vel": data["ref_joint_vel"][piece, tick],
        "body_pos_w": np.vstack([[0.3, -0.2, data["ref_height"][piece, tick, 0]], np.full((others, 3), 9.0)]),
        "body_quat_w": np.vstack([quat, np.tile([1.0, 0.0, 0.0, 0.0], (others, 1))]),
        "body_lin_vel_w": np.vstack([rotate(data["ref_lin_vel"][piece, tick]), np.full((others, 3), 5.0)]),
        "body_ang_vel_w": np.vstack([rotate(data["ref_ang_vel"][piece, tick]), np.full((others, 3), 5.0)]),
    }
    return decode_frame(encode_frame(seq, 0.02 * tick, arrays, flags=CALIBRATED | ACTIVE))


def robot_state(data, piece: int, tick: int, imu_yaw: float) -> RobotState:
    quat = quat_mul(quat_from_yaw(imu_yaw), data["robot_quat"][piece, tick].astype(np.float64))
    return RobotState(
        time_s=0.02 * tick,
        gyro=[float(v) for v in data["gyro"][piece, tick]],
        body_quat=[float(v) for v in quat],
        projected_gravity=[float(v) for v in observer_gravity(list(quat), [0.0, 0.0, -1.0])],
        motor_positions={n: float(v) for n, v in zip(ROBOT_JOINT_ORDER, data["joint_pos"][piece, tick])},
        motor_velocities={n: float(v) for n, v in zip(ROBOT_JOINT_ORDER, data["joint_vel"][piece, tick])},
    )


class FrameTest(unittest.TestCase):
    def test_every_frame_is_the_trainings(self):
        data = load_fixture()
        worst = 0.0
        for pc_yaw, imu_yaw in ((0.0, 0.0), (0.7, -1.3), (-2.9, 2.2)):
            # The turn from the PC's world into the IMU's (the move finds it at the start).
            offset = quat_from_yaw(imu_yaw - pc_yaw)
            for piece in range(data["frame"].shape[0]):
                for tick in range(data["frame"].shape[1]):
                    state = robot_state(data, piece, tick, imu_yaw)
                    frame = build_frame(
                        Reference.aligned(wire_frame(data, piece, tick, pc_yaw, tick), offset),
                        robot_quat=state.body_quat,
                        gyro=state.gyro,
                        projected_gravity=state.projected_gravity,
                        joint_pos=state.motor_positions,
                        joint_vel=state.motor_velocities,
                        last_action=data["last_action"][piece, tick],
                    )
                    error = float(np.max(np.abs(frame.astype(np.float32) - data["frame"][piece, tick])))
                    worst = max(worst, error)
                    self.assertLess(error, ATOL, f"piece {piece} tick {tick} yaw {pc_yaw}/{imu_yaw}")
        print(f"worst |robot - training| over {data['frame'].size * 3} values: {worst:.2e}")

    def test_alignment_gives_the_reference_the_robots_heading(self):
        data = load_fixture()
        for pc_yaw, imu_yaw in ((0.0, 0.0), (0.7, -1.3), (-2.9, 2.2)):
            robot = robot_state(data, 1, 70, imu_yaw).body_quat
            reference = wire_frame(data, 1, 70, pc_yaw, 0)
            aligned = Reference.aligned(reference, yaw_alignment(robot, reference.anchor_quat))
            self.assertAlmostEqual(math.remainder(yaw_of(aligned.quat) - yaw_of(robot), math.tau), 0.0, places=9)
            # Only the heading turns: the tilt and height stay the reference's.
            self.assertAlmostEqual(aligned.quat[1] ** 2 + aligned.quat[2] ** 2,
                                   reference.anchor_quat[1] ** 2 + reference.anchor_quat[2] ** 2, places=9)
            self.assertEqual(aligned.height, reference.anchor_pos[2])

    def test_history_fills_with_the_first_frame_then_shifts(self):
        history = ObservationHistory()
        frames = np.arange(13 * OBSERVATION_WIDTH, dtype=np.float32).reshape(13, OBSERVATION_WIDTH)
        out = history.push(frames[0])
        self.assertEqual(out.shape, (1, HISTORY_LENGTH, OBSERVATION_WIDTH))
        self.assertTrue(np.array_equal(out[0], np.repeat(frames[:1], HISTORY_LENGTH, axis=0)))
        for k in range(1, 13):
            out = history.push(frames[k])
        self.assertTrue(np.array_equal(out[0], frames[3:13]))
        history.reset()
        self.assertTrue(np.array_equal(history.push(frames[5])[0], np.repeat(frames[5:6], HISTORY_LENGTH, axis=0)))


def fake_contract() -> WholebodyContract:
    """The contract fields the move reads (the training's HMD neck limits and slew rate)."""
    return WholebodyContract(
        recipe="test", checkpoint_filename="model_0.pt", checkpoint_iteration=0,
        checkpoint_sha256="0" * 64, benchmark_report_sha256="0" * 64,
        eval_only_benchmark_report_sha256="0" * 64, dry_run=False,
        neck_limits_rad={"head": (-1.48, 1.48), "neck_roll": (-0.4, 0.4), "neck_pitch": (-1.48, 0.4)},
        neck_slew_rates_rad_s={"head": 2.5, "neck_roll": 2.5, "neck_pitch": 2.5},
        self_test_histories=np.zeros((0, 1, HISTORY_LENGTH, OBSERVATION_WIDTH), dtype=np.float32),
        self_test_actions=np.zeros((0, 18)),
    )


class _Receiver:
    def __init__(self):
        self.frame = None
        self.time = 0.0

    def read(self):
        return ReferenceSnapshot(frame=self.frame, received_s=self.time, stream=0, lost=0,
                                 latency_excess_s=0.0, activation=1, stale_flag_s=0.0, held_flag_s=0.0)

    def now(self):
        return self.time


class _Session:
    """Answers 0 until the move tracks, then the training policy's actions in turn; keeps the tracked inputs."""

    def __init__(self, actions):
        self.actions = actions
        self.move = None
        self.inputs = []
        self.tracked = 0

    def run(self, _outputs, feeds):
        if self.move.phase != Phase.TRACK:
            return [np.zeros((1, 18), dtype=np.float32)]
        self.inputs.append(feeds["obs_history"].copy())
        self.tracked += 1
        return [self.actions[self.tracked - 1][None].astype(np.float32)]


class MoveFeedsTrainingHistoryTest(unittest.TestCase):
    def test_the_move_gives_the_policy_the_trainings_histories(self):
        """The move enters, stands and joins the PC's first frame (sent still),
        then gives the policy the training's frames: the newest row from the
        first tracked tick, the whole history from the tenth."""
        data = load_fixture()
        ticks = data["frame"].shape[1] - 1
        for piece in range(data["frame"].shape[0]):
            for pc_yaw, imu_yaw in ((0.0, 0.0), (1.1, -0.4)):
                receiver = _Receiver()
                session = _Session(data["last_action"][piece])
                move = WholebodyMove(receiver, session=session, contract=fake_contract())
                session.move = move
                # As the training benchmark starts: the robot faces the
                # reference's heading (its trunk heading is the reference's).
                first = wire_frame(data, piece, 0, pc_yaw, 0)
                zero = np.zeros(3)
                receiver.frame = first.__class__(**{
                    **first.__dict__,
                    "anchor_quat": quat_mul(quat_from_yaw(pc_yaw), data["robot_quat"][piece, 0].astype(np.float64)),
                    "joint_vel": np.zeros_like(first.joint_vel), "anchor_lin_vel": zero, "anchor_ang_vel": zero,
                })

                def observe(tick, t):
                    state = robot_state(data, piece, tick, imu_yaw)
                    state.time_s = t
                    return Observation(robot_state=state, user_input=UserInput())

                move.state = MoveState.STARTING
                move.on_start(observe(0, 0.0), MotorCommand())
                t = 0.0
                while move.phase != Phase.TRACK and t < 5.0:
                    t += 0.02
                    receiver.time = t
                    move.step(observe(0, t), MotorCommand())
                self.assertEqual(move.phase, Phase.TRACK)
                session.inputs.clear()
                for tick in range(ticks):
                    t += 0.02
                    receiver.frame = wire_frame(data, piece, tick, pc_yaw, tick + 1)
                    receiver.time = t
                    move.step(observe(tick, t), MotorCommand())
                self.assertEqual(move.phase, Phase.TRACK)
                frames = data["frame"][piece]
                for tick, given in enumerate(session.inputs):
                    rows = slice(None) if tick >= HISTORY_LENGTH - 1 else slice(-1, None)
                    window = np.stack([frames[k] for k in range(tick - HISTORY_LENGTH + 1, tick + 1)])
                    error = float(np.max(np.abs(given[0][rows] - window[rows])))
                    self.assertLess(error, ATOL, f"piece {piece} tick {tick}")


if __name__ == "__main__":
    unittest.main()
