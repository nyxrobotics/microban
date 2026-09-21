# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Record a bam identification trajectory against a real motor, for a
pendulum testbench: a rod of known mass driven by the motor, with a known
point mass at a known radius (see tools/actuator_id/README.md).

Generalizes bam.dynamixel.record (which only supports XL330, hardcoded
ID=1) to any motor registered in motors.py, so adding a new motor type is
just adding an entry there - this script doesn't change.

Usage:
    uv run python3 tools/actuator_id/record.py \\
        --motor xc330 --id 1 --port /dev/ttyUSB0 \\
        --mass 0.3 --arm-mass 0.05 --length 0.1 \\
        --trajectory sin_time_square --logdir tools/actuator_id/logs
"""

import argparse
import datetime
import json
import time

from bam.trajectory import trajectories
from motors import MOTORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", required=True, choices=sorted(MOTORS), help="Motor type, see motors.py")
    parser.add_argument("--id", type=int, default=1, help="Dynamixel ID of the motor under test")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--mass", type=float, required=True, help="Point mass at the tip [kg]")
    parser.add_argument("--arm-mass", type=float, required=True, help="Mass of the arm/rod itself [kg]")
    parser.add_argument("--length", type=float, required=True, help="Pivot-to-mass distance [m]")
    parser.add_argument("--trajectory", default="sin_time_square", choices=sorted(trajectories))
    parser.add_argument("--kp", type=int, default=400, help="Firmware position P gain used while recording")
    parser.add_argument("--vin", type=float, default=12.0, help="Bench supply voltage (logged metadata only - actual voltage is read per-sample)")
    parser.add_argument("--logdir", required=True)
    args = parser.parse_args()

    spec = MOTORS[args.motor]
    trajectory = trajectories[args.trajectory]
    motor_id = args.id

    controller = spec.controller(serial_port=args.port, baudrate=spec.default_baudrate, timeout=0.1)

    print(f"Homing to trajectory start and enabling torque (kp={args.kp}) ...")
    start = time.time()
    torque_enable = False
    while time.time() - start < 1.0:
        goal_position, torque_enable = trajectory(0)
        if torque_enable:
            controller.write_goal_position(motor_id, goal_position)
        controller.write_torque_enable(motor_id, torque_enable)
        controller.write_position_p_gain(motor_id, args.kp)

    print(f"Recording '{args.trajectory}' ({trajectory.duration:.1f}s) ...")
    data = {
        "mass": args.mass,
        "arm_mass": args.arm_mass,
        "length": args.length,
        "kp": args.kp,
        "vin": args.vin,
        "motor": args.motor,
        "trajectory": args.trajectory,
        "entries": [],
    }

    start = time.time()
    while time.time() - start < trajectory.duration:
        t = time.time() - start
        goal_position, new_torque_enable = trajectory(t)
        if new_torque_enable != torque_enable:
            controller.write_torque_enable(motor_id, new_torque_enable)
            torque_enable = new_torque_enable
            time.sleep(0.001)
        if torque_enable:
            controller.write_goal_position(motor_id, goal_position)
            time.sleep(0.001)

        t0 = time.time() - start
        entry = {
            "position": controller.read_present_position(motor_id)[0],
            "speed": spec.velocity_of_raw(controller.read_present_velocity(motor_id)[0]),
            "load": spec.pwm_duty_of_raw(controller.read_present_pwm(motor_id)[0]),
            "input_volts": controller.read_present_input_voltage(motor_id)[0] / 10.0,
            "temp": controller.read_present_temperature(motor_id)[0],
        }
        t1 = time.time() - start

        entry["timestamp"] = (t0 + t1) / 2.0
        entry["goal_position"] = goal_position
        entry["torque_enable"] = torque_enable
        data["entries"].append(entry)

    print("Returning to zero ...")
    goal_position = data["entries"][-1]["position"]
    return_dt = 0.01
    max_step = return_dt * 1.0
    while abs(goal_position) > 0:
        goal_position = max(0.0, goal_position - max_step) if goal_position > 0 else min(0.0, goal_position + max_step)
        controller.write_goal_position(motor_id, goal_position)
        time.sleep(return_dt)
    controller.write_torque_enable(motor_id, False)

    date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
    filename = f"{args.logdir}/{date}.json"
    json.dump(data, open(filename, "w"))
    print(f"Saved {filename} ({len(data['entries'])} entries)")

    max_abs_load = max(abs(e["load"]) for e in data["entries"])
    print(f"Peak |load| (PWM duty): {max_abs_load:.2f} (1.0 = fully saturated - re-check your mass/length if this is often near 1.0)")


if __name__ == "__main__":
    main()
