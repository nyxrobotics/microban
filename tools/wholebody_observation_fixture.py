#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Record the training's whole-body observation for tests/test_wholebody_observation.py.

Runs in the training repository's environment (mjlab_microban, CPU), never on
the robot:

    cd <mjlab_microban> && .venv/bin/python <microban>/tools/wholebody_observation_fixture.py \\
        --checkpoint logs/rsl_rl/microban_wholebody/<run>/model_N.pt \\
        --motion-file <data>/pico_eval/eval/motion.h5 \\
        --output <microban>/tests/fixtures/wholebody_observation.npz

The task's own environment (play mode: no noise; the sensor delays set to 0,
since on the robot they are the hardware's own) runs pieces of the motion as
the benchmark does (from HOME under a held reference, the robot facing the
reference's heading) with the checkpoint's policy.  Each tick it records the
robot state the robot runtime reads (trunk quaternion, gyro, joint angles and
velocities, previous raw action), the reference as the PC sends it (all 21
joints, trunk quaternion, velocities and height) and the actor's observation
(the newest of its 10 rows).  It checks that the actor history is the newest
frames, every row the first frame after the reset, so the fixture stores only
the newest rows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, mdp
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from tensordict import TensorDict

from mjlab_microban.policy_contract import ACTION_JOINT_NAMES
from mjlab_microban.scripts.wholebody_benchmark import benchmark_env_cfg
from mjlab_microban.wholebody import benchmark
from mjlab_microban.wholebody.env_cfg import ACTOR_GROUP, TASK_ID
from mjlab_microban.wholebody.runner import WholebodyOnPolicyRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pieces", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    cfg = benchmark_env_cfg(args.motion_file, 10.0, args.pieces, seed=0)
    for term in cfg.observations[ACTOR_GROUP].terms.values():
        term.delay_min_lag = 0
        term.delay_max_lag = 0
    step_dt = cfg.decimation * cfg.sim.mujoco.timestep
    pieces = benchmark.load_pieces(
        args.motion_file, ACTION_JOINT_NAMES, 10.0, step_dt, benchmark.servo_no_load_speed_rad_s()
    )
    # Different kinds of motion: the longest piece of each of the first kinds.
    by_kind: dict[str, benchmark.Piece] = {}
    for piece in sorted(pieces, key=lambda p: -p.steps):
        by_kind.setdefault(piece.kind, piece)
    pieces = list(by_kind.values())[: args.pieces]
    cfg.scene.num_envs = len(pieces)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        runner = WholebodyOnPolicyRunner(RslRlVecEnvWrapper(env), asdict(load_rl_cfg(TASK_ID)), device="cpu")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")
        n = len(pieces)
        ids = torch.arange(n)
        settle = benchmark.settle_steps(env.step_dt)
        env.reset()
        command = env.command_manager.get_term("motion")
        robot = command.robot
        command.hold(torch.ones(n, dtype=torch.bool))
        command.reset_to_motion(
            ids, torch.tensor([p.window_id for p in pieces]),
            torch.tensor([p.start_s for p in pieces], dtype=command.motion_times.dtype),
        )
        benchmark._place_at_home(env, command)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.command_manager.compute(dt=0.0)
        env.event_manager.apply(mode="step", dt=env.step_dt)
        env.sim.sense()
        env.observation_manager.reset(ids)
        obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[n])

        keys = ("robot_quat", "gyro", "joint_pos", "joint_vel", "last_action", "ref_joint_pos", "ref_joint_vel",
                "ref_quat", "ref_lin_vel", "ref_ang_vel", "ref_height", "frame")
        record = {key: [] for key in keys}
        for step in range(args.steps):
            history = obs[ACTOR_GROUP].detach().cpu().numpy()
            frames = history[:, -1]
            if step == 0:
                assert np.array_equal(history, np.repeat(frames[:, None], history.shape[1], axis=1))
            else:
                previous = np.stack(record["frame"], axis=1)  # (n, step, 118)
                padded = np.concatenate(
                    [np.repeat(previous[:, :1], history.shape[1], axis=1), previous, frames[:, None]], axis=1
                )
                assert np.array_equal(history, padded[:, -history.shape[1]:]), "history is not the newest frames"
            cached = command._cached_frames
            values = {
                "robot_quat": robot.data.root_link_quat_w,
                "gyro": mdp.builtin_sensor(env, "robot/imu_ang_vel"),
                "joint_pos": robot.data.joint_pos_biased,
                "joint_vel": robot.data.joint_vel,
                "last_action": env.action_manager.action,
                "ref_joint_pos": command.all_joint_pos,
                "ref_joint_vel": cached["joint_vel"],
                "ref_quat": command.anchor_quat_w,
                "ref_lin_vel": command.anchor_lin_vel_w,
                "ref_ang_vel": command.anchor_ang_vel_w,
                "ref_height": command.anchor_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3],
            }
            for key, value in values.items():
                record[key].append(value.detach().cpu().numpy().astype(np.float32))
            record["frame"].append(frames.astype(np.float32))
            if step == settle:
                command.hold(torch.zeros(n, dtype=torch.bool))
            with torch.inference_mode():
                actions = policy(obs)
            raw, _, _, _, _ = env.step(actions)
            env._manual_reset_pending.zero_()
            obs = TensorDict(raw, batch_size=[n])
        out = {key: np.stack(value, axis=1) for key, value in record.items()}  # (n, steps, ...)
        out["joint_names"] = np.asarray(robot.joint_names)
        out["clip_names"] = np.asarray([p.clip_name for p in pieces])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **out)
        print(f"{n} pieces x {args.steps} ticks -> {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
