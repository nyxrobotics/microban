# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Local simulation entry point — never deployed to the robot.

Usage:
    uv run --group sim src/sim/sim_main.py --hz 50
    make sim 
"""

import argparse

from input.gamepad_input import GamepadInputSource
from input.network_input import NetworkInputSource
from moves.getup import GetupMove
from moves.hmd_head import HmdHeadTrackingMove
from moves.pico_arms import PicoArmTrackingMove
from moves.policy_selector import PolicySelectableWalkMove
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from scheduler import Scheduler
from sim.mujoco_controller import MuJoCoController
from sim.mujoco_input import MuJoCoInputSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Run microban scheduler in MuJoCo simulation.")
    parser.add_argument("--hz", type=float, default=50.0, metavar="FREQ", help="Scheduler frequency in Hz (default: 50)")
    parser.add_argument("--delay-act", type=int, default=2, metavar="STEPS", help="Actuation delay in simulator steps (1 step = 0.005 s)")
    parser.add_argument("--delay-pos", type=int, default=0, metavar="TICKS", help="Motor position read delay in scheduler ticks (1 tick = 20 ms at 50 Hz)")
    parser.add_argument("--delay-vel", type=int, default=1, metavar="TICKS", help="Motor velocity read delay in ticks")
    parser.add_argument("--delay-gyro", type=int, default=3, metavar="TICKS", help="Gyro read delay in ticks")
    parser.add_argument("--delay-quat", type=int, default=4, metavar="TICKS", help="Quaternion (projected gravity) read delay in ticks")
    parser.add_argument("--trunk-com-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"), help="CoM offset on trunk body in meters (body frame)")
    parser.add_argument("--input", choices=("keyboard", "network", "gamepad"), default="keyboard", help="Control input (default: keyboard)")
    parser.add_argument("--network-port", type=int, default=5555, help="UDP port used with --input network")
    args = parser.parse_args()

    # "walk" active from launch: standing purely on the neutral-pose position hold
    # (no active move) isn't a stable equilibrium and drifts into an overcurrent trip
    # within a couple of seconds — pre-existing, unrelated to the moves registered
    # here, confirmed by bisecting against this file's pre-getup-integration version.
    if args.input == "network":
        input_source = NetworkInputSource(port=args.network_port)
        key_callback = None
        reset_source = None
    elif args.input == "gamepad":
        # Real B/A/R3 get-up test controls (see GamepadInputSource), so that
        # exact flow can be rehearsed here before ever touching real hardware.
        # A raw /dev/input/js* reader, independent of the MuJoCo viewer, so
        # (unlike keyboard) it needs no key_callback/reset_source.
        input_source = GamepadInputSource(button_moves={"X": "walk"})
        key_callback = None
        reset_source = None
    else:
        input_source = MuJoCoInputSource(
            move_keys={"h": "head", "s": "squat", "v": "walk", "g": "getup"},
            initial_active_moves={"walk"},
        )
        key_callback = input_source.key_callback
        reset_source = input_source
    controller = MuJoCoController(
        mjcf_path="src/model/mjcf/scene.xml",
        key_callback=key_callback,
        reset_source=reset_source,
        delay_act_steps=args.delay_act,
        delay_pos_ticks=args.delay_pos,
        delay_vel_ticks=args.delay_vel,
        delay_gyro_ticks=args.delay_gyro,
        delay_quat_ticks=args.delay_quat,
        trunk_com_offset=tuple(args.trunk_com_offset),
    )
    if isinstance(input_source, MuJoCoInputSource):
        input_source.set_viewer_opt(controller.viewer_opt)

    scheduler = Scheduler(
        frequency_hz=args.hz,
        controller=controller,
        input_source=input_source,
        moves={
            "head": RotateHeadMove(),
            "squat": SquatMove(),
            "walk": PolicySelectableWalkMove(controller=controller),
            # Match production ordering: direct controller arms overwrite only
            # the six arm joints emitted by the locomotion actor.
            "pico_arms": PicoArmTrackingMove(controller=controller),
            "hmd_head": HmdHeadTrackingMove(),
            "getup": GetupMove(controller=controller),
        },
    )
    for move in scheduler.registered_moves.values():
        move.preload()
    # The overcurrent safety's cheap proxy estimate (position-error based) exists to
    # avoid an extra bus read on real hardware; in sim that cost doesn't apply, and the
    # proxy overestimates enough during a fall/recovery's large corrective motions to
    # trip the safety before GetupMove gets a chance to run. Use the real (already
    # current-limited by the BAM actuator model) simulated current instead.
    scheduler.observer.observe_current = True
    scheduler.run()


if __name__ == "__main__":
    main()
