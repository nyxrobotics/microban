# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Registry of motors this repo can run bam (Better Actuator Models)
identification against - add an entry here to support a new motor type.

Each entry needs:
  - controller: a rustypot PyController class that can talk to the motor
    (Xl330PyController also works for XC330 - same X-series control table,
    already the pattern this repo uses in src/robot_controller.py).
  - bam_actuator: a zero-arg factory returning a bam Actuator instance
    (subclass of bam.actuator.VoltageControlledActuator/CurrentControlledActuator).
    See xc330_actuator.py for how to add a motor bam doesn't ship a
    definition for.
  - velocity_of_raw / pwm_duty_of_raw: convert the controller's raw
    present_velocity/present_pwm register reads into rad/s and a duty
    cycle in [-1, 1]. These come from the motor's control table (velocity
    unit, PWM limit), not from bam - see each motor's ROBOTIS e-manual
    page.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from rustypot import Xl330PyController
from bam.testbench import Pendulum

from xc330_actuator import XC330Actuator


@dataclass
class MotorSpec:
    controller: type
    bam_actuator: Callable[[], object]
    velocity_of_raw: Callable[[float], float]
    pwm_duty_of_raw: Callable[[float], float]
    default_baudrate: int = 1_000_000


def _velocity_of_raw(rev_per_min_per_lsb: float) -> Callable[[float], float]:
    return lambda raw: float(raw) * rev_per_min_per_lsb * (2.0 * np.pi / 60.0)


def _pwm_duty_of_raw(pwm_limit: int) -> Callable[[float], float]:
    def convert(raw: float) -> float:
        x = float(raw)
        if x > 2**15 - 1:
            x -= 2**16
        return float(np.clip(x / pwm_limit, -1.0, 1.0))
    return convert


MOTORS: dict[str, MotorSpec] = {
    # XC330-T288-T: the 21 servos actually on this robot (see src/constants.py).
    # Control table constants (velocity unit, PWM limit) from the ROBOTIS
    # e-manual - same values as XL330, since both share the X-series
    # control table and (very likely) firmware generation.
    "xc330": MotorSpec(
        controller=Xl330PyController,
        bam_actuator=lambda: XC330Actuator(Pendulum),
        velocity_of_raw=_velocity_of_raw(0.229),
        pwm_duty_of_raw=_pwm_duty_of_raw(885),
    ),
}
