#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Send a recorded whole-body motion as the PC's reference stream (for tests).

    PYTHONPATH=src uv run --with h5py python tools/wholebody_reference_sender.py \\
        <motion.h5> --clip picoteleopit_rec05 [--start-s 0] [--seconds 30] [--lead-in-s 3] \\
        [--host 127.0.0.1] [--port 5556] [--rate-hz 50]

Reads a microban_wholebody_motion dataset (mjlab_microban
wholebody/data/hdf5.py), plays one window at real time from ``--start-s``,
interpolated to ``--rate-hz`` as the training's MotionLib does (linear, the
quaternions slerped), and sends each frame in the wire format of
input/wholebody_reference.py as the PC does: CALIBRATED with the first frame
held during ``--lead-in-s``, then CALIBRATED and ACTIVE (RESTART on the first
active frame), the operator standing still at the first frame for
``--hold-s`` (as the training's benchmark starts a piece) before the motion
plays, then CALIBRATED again for a second.
Only the PC's live retargeting replaces it.  ``--control-port`` also stands in
for the PICO bridge in a simulation without a headset: control packets with
A (torque) from the start and R3 (policy) after a second, triggers released.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from input.wholebody_reference import ACTIVE, CALIBRATED, RESTART, encode_frame  # noqa: E402

FRAME_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
ROBOT_JOINTS = (
    "head", "neck_roll", "neck_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_elbow", "right_hip_yaw", "right_hip_roll",
    "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_elbow", "left_hip_yaw", "left_hip_roll",
    "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll",
)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(omega)
    safe = np.abs(sin) > 1e-8
    w0 = np.where(safe, np.sin((1.0 - t) * omega) / np.where(safe, sin, 1.0), 1.0 - t)
    w1 = np.where(safe, np.sin(t * omega) / np.where(safe, sin, 1.0), t)
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


class MotionReplay:
    """One window of a dataset, sampled at any time like MotionLib.get_frames."""

    def __init__(self, path: str | Path, clip: str) -> None:
        import h5py

        with h5py.File(path, "r") as h5:
            def strings(key):
                return [v.decode() if isinstance(v, bytes) else str(v) for v in h5[key][()]]

            names = strings("clip_names")
            ids = np.asarray(h5["source_clip_ids"])
            windows = [i for i, source in enumerate(ids) if names[source].startswith(clip)]
            if not windows:
                raise SystemExit(f"no window of {clip!r}; clips: {sorted(set(names))[:20]} ...")
            window = windows[0]
            self.name = names[ids[window]]
            start = int(h5["clip_starts"][window])
            self.length = int(h5["clip_lengths"][window])
            self.fps = float(h5["clip_fps"][window])
            joints = strings("joint_names")
            columns = [joints.index(name) for name in ROBOT_JOINTS]
            bodies = strings("body_names")
            if bodies[0] != "trunk":
                order = [bodies.index("trunk")] + [i for i in range(len(bodies)) if bodies[i] != "trunk"]
            else:
                order = list(range(len(bodies)))
            self.arrays = {}
            for key in FRAME_KEYS:
                data = np.asarray(h5[key][start : start + self.length], dtype=np.float64)
                self.arrays[key] = data[:, columns] if key.startswith("joint") else data[:, order]
        self.duration_s = (self.length - 1) / self.fps

    def sample(self, time_s: float, still: bool = False) -> dict[str, np.ndarray]:
        frame_f = min(max(time_s, 0.0) * self.fps, self.length - 1)
        i0 = int(frame_f)
        i1 = min(i0 + 1, self.length - 1)
        alpha = frame_f - i0
        out = {}
        for key, data in self.arrays.items():
            if key == "body_quat_w":
                out[key] = _slerp(data[i0], data[i1], alpha)
            else:
                out[key] = data[i0] + alpha * (data[i1] - data[i0])
        if still:
            for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                out[key] = np.zeros_like(out[key])
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--clip", required=True, help="Clip name (prefix) in the dataset")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--lead-in-s", type=float, default=3.0, help="CALIBRATED only, before ACTIVE")
    parser.add_argument("--hold-s", type=float, default=1.2, help="Standing still at the first frame after ACTIVE")
    parser.add_argument("--control-port", type=int, default=None, help="Also send A and R3 control packets here")
    args = parser.parse_args()
    replay = MotionReplay(args.motion, args.clip)
    end_s = replay.duration_s if args.seconds is None else min(replay.duration_s, args.start_s + args.seconds)
    print(f"{replay.name}: {replay.duration_s:.1f} s at {replay.fps:g} fps; sending {args.start_s:.1f}-{end_s:.1f} s")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1.0 / args.rate_hz
    began = time.perf_counter()
    seq = 0
    active = False
    while True:
        elapsed = time.perf_counter() - began
        motion_s = args.start_s + max(0.0, elapsed - args.lead_in_s - args.hold_s)
        if motion_s > end_s + 1.0:
            break
        if args.lead_in_s <= elapsed and motion_s <= end_s:
            flags = CALIBRATED | ACTIVE | (0 if active else RESTART)
            active = True
        else:
            flags = CALIBRATED
        frame = replay.sample(min(motion_s, end_s), still=elapsed < args.lead_in_s + args.hold_s)
        sock.sendto(encode_frame(seq, time.time(), frame, flags), (args.host, args.port))
        if args.control_port is not None:
            control = {"version": 1, "session_id": "wholebody-sender", "seq": seq, "active_moves": [],
                       "locomotion_policy": "pico_teleop", "torque_enabled": True, "torque_off_requested": False,
                       "policy_enabled": elapsed > 1.0}
            sock.sendto(json.dumps(control).encode(), (args.host, args.control_port))
        seq += 1
        time.sleep(max(0.0, began + seq * period - time.perf_counter()))
    print(f"sent {seq} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
