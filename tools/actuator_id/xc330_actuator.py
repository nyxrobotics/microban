# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""bam actuator definition for the XC330-T288-T, which bam (Rhoban's
better-actuator-models) doesn't ship - the closest built-in is XL330Actuator
(bam/dynamixel/actuator.py), same encoder resolution and PWM limit, so this
copies its structure. kt/R start from the ROBOTIS datasheet torque constant
and the XL330 m6 fit; record.py/fit.py identify them on a pendulum rig.
"""

import numpy as np
from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Testbench

XC330_ENCODER_COUNTS_PER_REV = 4096  # ROBOTIS e-manual: 4096 pulse/rev, same as XL330
XC330_KP_DIVISOR = 256  # ASSUMED same as XL330 (same X-series control table/firmware gen) - unverified
XC330_PWM_LIMIT = 885  # ROBOTIS e-manual control table default, same value as XL330


class XC330Actuator(VoltageControlledActuator):
    """Represents a Dynamixel XC330-T288-T actuator."""

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=12.0,  # bench supply voltage used for the identification
            kp=400,
            error_gain=(XC330_ENCODER_COUNTS_PER_REV / (2 * np.pi))
            / (XC330_KP_DIVISOR * XC330_PWM_LIMIT),
            max_pwm=1.0,
            max_current=0.91,  # ROBOTIS e-manual: 910 mA firmware current limit
        )

    def initialize(self):
        # Seed: Robotis datasheet, 1.150 Nm/A @ 11.1V
        self.model.kt = Parameter(1.150, 0.5, 2.5)
        # Seed: the XL330 m6 fit; the fit lands near 10 Ohm, inside the bounds.
        self.model.R = Parameter(2.811, 1.5, 12.0)
        self.model.armature = Parameter(0.005, 0.0001, 0.05)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value
