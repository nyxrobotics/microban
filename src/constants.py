# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import numpy as np

MOTOR_TO_ID = {
    "left_hip_yaw": 11,
    "left_hip_roll": 12,
    "left_hip_pitch": 13,
    "left_knee": 14,
    "left_ankle_pitch": 15,
    "left_ankle_roll": 16,
    "right_hip_yaw": 21,
    "right_hip_roll": 22,
    "right_hip_pitch": 23,
    "right_knee": 24,
    "right_ankle_pitch": 25,
    "right_ankle_roll": 26,
    "left_shoulder_pitch": 31,
    "left_shoulder_roll": 32,
    "left_elbow": 33,
    "right_shoulder_pitch": 41,
    "right_shoulder_roll": 42,
    "right_elbow": 43,
    "head": 51,
    "neck_roll": 52,
    "neck_pitch": 53,
}

ID_TO_MOTOR = {v: k for k, v in MOTOR_TO_ID.items()}

NEUTRAL_POSE = {
    "left_hip_yaw": float(np.deg2rad(0.0)),
    "left_hip_roll": float(np.deg2rad(5.0)),
    "left_hip_pitch": float(np.deg2rad(-10.0)),
    "left_knee": float(np.deg2rad(0.0)),
    "left_ankle_pitch": float(np.deg2rad(0.0)),
    "left_ankle_roll": float(np.deg2rad(-5.0)),
    "right_hip_yaw": float(np.deg2rad(0.0)),
    "right_hip_roll": float(np.deg2rad(-5.0)),
    "right_hip_pitch": float(np.deg2rad(-10.0)),
    "right_knee": float(np.deg2rad(0.0)),
    "right_ankle_pitch": float(np.deg2rad(0.0)),
    "right_ankle_roll": float(np.deg2rad(5.0)),
    "left_shoulder_pitch": float(np.deg2rad(10.0)),
    "left_shoulder_roll": float(np.deg2rad(10.0)),
    "left_elbow": float(np.deg2rad(-20.0)),
    "right_shoulder_pitch": float(np.deg2rad(10.0)),
    "right_shoulder_roll": float(np.deg2rad(-10.0)),
    "right_elbow": float(np.deg2rad(-20.0)),
    "head": float(np.deg2rad(0.0)),
    "neck_roll": float(np.deg2rad(0.0)),
    "neck_pitch": float(np.deg2rad(0.0)),
}

MOTOR_SIGN = {
    "left_hip_yaw": -1.0,
    "left_hip_roll": 1.0,
    "left_hip_pitch": -1.0,
    "left_knee": 1.0,
    "left_ankle_pitch": 1.0,
    "left_ankle_roll": -1.0,
    "right_hip_yaw": -1.0,
    "right_hip_roll": 1.0,
    "right_hip_pitch": 1.0,
    "right_knee": -1.0,
    "right_ankle_pitch": -1.0,
    "right_ankle_roll": -1.0,
    "left_shoulder_pitch": 1.0,
    "left_shoulder_roll": -1.0,
    "left_elbow": 1.0,
    "right_shoulder_pitch": -1.0,
    "right_shoulder_roll": -1.0,
    "right_elbow": -1.0,
    "head": 1.0,
    # neck_roll / neck_pitch signs are CAD-inferred (2026-09-20), not yet measured on real
    # hardware: on both servos, the idler-horn/idle-cap hardware sits at the low end of the
    # joint's rotation axis (x for roll, y for pitch), so the driven/output face was taken to
    # be the high end, matching the +X / +Y directions already used as the joint axes in the
    # URDF. Combined with DYNAMIXEL's positive-direction convention (CCW viewed from the horn
    # side), that gives +1.0 for both. Verify with a small test motion before trusting this for
    # full-range moves.
    "neck_roll": 1.0,
    "neck_pitch": 1.0,
}

# Position P Gain (Dynamixel register value)
KP_DEFAULT: int = 400        # ~0.886 Nm/rad in MuJoCo
KP_RL: int = 125             # ~0.277 Nm/rad in MuJoCo
KP_GAIN_PRM: float = 0.0022  # Nm/rad per register unit (for Xl330). NOT updated for XC330-T288-T: likely scales with the ~2.85x higher torque constant (see PROXY_KT) but the exact derivation (register-to-PWM/current scaling) is not confident enough here to rescale blindly. Needs review.

# BAM motor model (bam package). All 21 servos are now XC330-T288-T (previously
# XL330-M288-T), and the battery is now 3S (previously 2S). The bam project
# (github.com/Rhoban/bam) has no published identification for XC330-T288-T, so
# tools/actuator_id/ (this repo) adds an XC330 actuator definition and records a
# pendulum-rig identification against a real XC330-T288-T - see
# tools/actuator_id/README (usage) and xc330_params.json (raw fit output).
# PROXY_KT / PROXY_R below are that fit's result (2026-09-21: m6 model, 30 logs
# across sin_time_square/lift_and_drop/up_and_down, score 0.064 rad). BAM_MAX_CURRENT
# is still the Robotis datasheet current limit, not something bam fits.
BAM_VIN: float = 11.1        # 3S nominal (3x3.7V); matches XC330-T288-T's own rated voltage
BAM_VIN_MIN: float = 9.0     # 3S practical minimum (3x3.0V/cell)
BAM_VOLTAGE_DROP_GAIN: float = 0.2  # UNCHANGED: bam-fit for XL330, not re-identified for XC330/3S
BAM_MAX_CURRENT: float = 0.91 # XC330-T288-T firmware current limit [A] (Robotis control table: 910 mA default/max; was 1.75A for XL330-M288-T)

# Overcurrent safety: emergency torque-off when the summed |present_current| of all
# motors stays above OVERCURRENT_CUTOFF_A for OVERCURRENT_DEBOUNCE_TICKS consecutive ticks.
# Goal: cut the robot before a current spike (e.g. all motors snapping during a fall) trips the BMS.
PRESENT_CURRENT_UNIT_A: float = 0.001   # XL330 present_current register unit (1.0 mA/LSB)
OVERCURRENT_CUTOFF_A: float = 15.0      # total pack current threshold (CALIBRATE: below BMS trip, above normal walk peak). Re-check against the actual 3S BMS in use (the BOM-listed candidate is rated 40A) now that both the battery and motors changed.
# The get-up move drives all 21 joints at once through large, aggressive corrective
# motions (unlike steady-state walking) and legitimately draws more current — up to
# 21 * BAM_MAX_CURRENT =~ 19.1 A if every motor saturated simultaneously, already a
# hard ceiling from the actuator model's own per-motor current limit. 25 A gives a
# safety margin above that theoretical max (still well under the 40 A BMS rating)
# instead of tripping on get-up's normal current profile.
OVERCURRENT_CUTOFF_A_GETUP: float = 25.0
OVERCURRENT_DEBOUNCE_TICKS: int = 2     # consecutive over-threshold ticks before cutting

# Current proxy used when present_current is NOT read (Observer.observe_current = False), so the
# safety needs no extra bus transaction. Reproduces the bam XL330 m6 voltage-controlled model from
# data already read (present_position, present_velocity) and the command target:
#   duty = clip(PROXY_KP * PROXY_ERROR_GAIN * (target - q), ±PROXY_MAX_PWM)
#   I    = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R      then |I| capped at BAM_MAX_CURRENT
PROXY_KT: float = 1.043                  # XC330-T288-T torque constant [Nm/A], bam-fit 2026-09-21 (was 0.366 for XL330 m6; Robotis-datasheet estimate had been 1.150)
PROXY_R: float = 10.007                  # XC330-T288-T coil resistance [Ohm], bam-fit 2026-09-21 (was 2.811, the XL330 m6 bam-fit value reused as a placeholder)
PROXY_VIN: float = BAM_VIN               # supply voltage [V]
PROXY_ERROR_GAIN: float = 0.0028773775   # duty cycle per (kp * rad), XL330 encoder/gain scaling. Encoder resolution (4096 counts/rev, 12-bit) is the same across the X-series including XC330, so this may still be valid, but it has not been re-verified for XC330-T288-T.
PROXY_MAX_PWM: float = 1.0               # max duty cycle magnitude
PROXY_KP: int = KP_RL                    # firmware P gain assumed by the proxy (walking regime)
OVERCURRENT_PROXY_DELAY_TICKS: int = 3   # number of ticks to delay the proxy current estimate

# Velocity command limits [m/s, m/s, rad/s], applied centrally to every input source.
# Input sources emit normalized commands in [-1, 1]; scale_velocity() maps them to these.
# Rotation gets a wider range when turning in place (vx = vy = 0) than while translating.
VX_MAX: float = 0.7
VX_MAX_BACKWARD: float = 0.5  # backward (vx < 0) is capped lower than forward
VY_MAX: float = 0.3
VTHETA_MAX_STATIONARY: float = 3.0
VTHETA_MAX_MOVING: float = 1.5

# IMU (BMI088) I2C bus number on the Raspberry Pi
IMU_I2C_BUS: int = 1

# WXYZ rotation from IMU sensor coordinates into trunk/body coordinates.
# This is the MJCF site's child(sensor)-to-parent(body) orientation.
IMU_MOUNT_QUAT: tuple[float, float, float, float] = (0.5, -0.5, -0.5, 0.5)

# Observation DoF ordering
OBSERVATION_DOF_ORDER = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll"
]
