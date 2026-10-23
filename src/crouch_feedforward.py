"""IK tracking crouch feed-forward: the leg joint offsets of a two-foot target.

The same file is ``mjlab_microban/robot/crouch_feedforward.py`` (training) and
``src/crouch_feedforward.py`` on the robot (microban); both repositories test
that it gives the same numbers.  It needs numpy only.

When both published foot targets are above the floor band, the operator
crouches: the trunk goes down (and sideways) with both soles left where HOME
has them.  The leg angles of that pose are solved here (``solve_offsets``,
both soles at their HOME pose relative to a trunk moved by -(dx, dy, dz) in
the HOME-levelled trunk frame, the trunk's orientation that of HOME) and
tabulated once (``build_table``).  The IK tracking policy (the frozen walker plus its
residual) then works around that pose instead of HOME:

* action: leg target = HOME + offset + raw output;
* observation: leg joint_pos = measured - HOME - offset; the previous action
  stays the raw output (so it is relative to the same shifted pose).

``leg_offsets(table, foot)`` gives the target offsets of a batch of published
foot targets, and ``follow(offsets, target)`` moves the offsets in use one
policy step towards them.  Both use only operations numpy arrays and torch
tensors share, so the robot runs them on numpy and training on torch with the
same code.

The two-foot target of a row is (mean dx, mean dy, the lower dz) of the two
published targets.  Its offsets are zero unless both targets are above the
floor band and inside the two-foot support (no target, one foot up with the
other at zero, walking, and the passage between two single-foot targets where
both feet are up beyond the support).  On a held two-foot target the feet are
equal, so this is that target.  Taking the lower foot's height keeps the
offsets continuous while one foot goes up from a crouch inside the support.

The offsets in use follow their target at a bounded rate (``MAX_STEP_RAD`` per
policy step).  HOME's knee is straight, so the offsets grow like sqrt(dz) and
the target jumps where a foot target enters or leaves the floor band (about
0.67 rad, joint-space norm, at the first published 4.8 mm), where a target
leaves the support, and on the robot where the teleop drops both feet at once
(the trigger released, a twist command, the live bound: 1.36 rad from a
16 mm crouch).  The bound is the rate of a two-foot target ramped at the
teleop's 0.12 m/s at 7 mm (0.21 rad per step, less above), so a crouch
ramped in from no target arrives two steps (0.04 s) after its target and
every jump becomes a ramp of the same speed in training and on the robot
(0.14 s for dropping a 16 mm crouch).
"""

from __future__ import annotations

import math

import numpy as np

CONTRACT = "microban_pico_crouch_feedforward_v2"
# The twelve leg joints, in the order of the offsets.
LEG_JOINT_NAMES = tuple(
    f"{side}_{joint}"
    for side in ("right", "left")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
)
# A published foot target at or below this height is exactly zero (the teleop's
# floor band).
FLOOR_BAND_M = 0.0025
# The tabulated two-foot targets: the trained support (|dx|, |dy| <= 10 mm,
# dz <= 20 mm, IK_TRACKING_BOTH_FEET_TARGET_*); a foot target outside it gives zero.
XY_LIMIT_M = 0.01
Z_MAX_M = 0.02
XY_NODES = 9
# Nodes uniform in sqrt(dz): HOME's knee is straight, so the knee angle grows
# like sqrt(dz).
SQRT_Z_NODES = 13
# The offsets in use move by at most this joint-space distance (Euclidean norm
# of the twelve offsets) per policy step of POLICY_DT_S: 10 rad/s.
POLICY_DT_S = 0.02
MAX_STEP_RAD = 0.2

# The leg chains of robot.xml: (body pos, body quat w-x-y-z) from the trunk to
# the foot body, each body's joint on its z axis at its origin, then the foot
# site.  Quaternions are normalised as MuJoCo does.
_H = 0.7071067811865476
_CHAINS = {
    "right": (
        ((0.0, -0.036, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((-0.0375, 0.0, -0.0214), (0.5, -0.5, 0.5, -0.5)),
        ((-0.0145, 0.0, 0.0385), (0.0, _H, 0.0, -_H)),
        ((-0.005, -0.062, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((0.008, -0.0612131, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((-0.0385, 0.0, -0.0145), (0.0, -_H, 0.0, -_H)),
    ),
    "left": (
        ((0.0, 0.036, 0.0), (0.0, 0.0, 0.0, -1.0)),
        ((0.0375, 0.0, -0.0214), (0.5, -0.5, -0.5, 0.5)),
        ((-0.0145, 0.0, 0.0385), (_H, 0.0, -_H, 0.0)),
        ((0.005, 0.062, 0.0), (1.0, 0.0, 0.0, 0.0)),
        ((-0.008, 0.0612131, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.0385, 0.0, -0.0145), (_H, 0.0, -_H, 0.0)),
    ),
}
_SITES = {
    "right": ((0.0, -0.023, 0.042), (0.5, -0.5, -0.5, -0.5)),
    "left": ((0.0, -0.023, 0.042), (0.5, -0.5, -0.5, -0.5)),
}
_SOLVE_ITERATIONS = 60
_SOLVE_TOLERANCE = 1.0e-12
# The bent-knee start of the solver (the straight HOME knee is singular).
_START_KNEE_RAD = 0.6


def _quat_matrix(quat) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rot_z(angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    zero, one = np.zeros_like(angle), np.ones_like(angle)
    return np.stack(
        (np.stack((c, -s, zero), -1), np.stack((s, c, zero), -1), np.stack((zero, zero, one), -1)),
        -2,
    )


def leg_fk(side: str, q: np.ndarray):
    """Foot site pose in the trunk frame for leg angles ``q`` (N, 6), plus each
    joint's origin and axis (for the Jacobian).  Returns (pos (N, 3),
    rot (N, 3, 3), origins (N, 6, 3), axes (N, 6, 3))."""

    q = np.asarray(q, dtype=np.float64)
    rot = np.broadcast_to(np.eye(3), (len(q), 3, 3))
    pos = np.zeros((len(q), 3))
    origins, axes = [], []
    for index, (body_pos, body_quat) in enumerate(_CHAINS[side]):
        pos = pos + rot @ np.asarray(body_pos)
        rot = rot @ _quat_matrix(body_quat)
        origins.append(pos)
        axes.append(rot[:, :, 2])
        rot = rot @ _rot_z(q[:, index])
    site_pos, site_quat = _SITES[side]
    pos = pos + rot @ np.asarray(site_pos)
    rot = rot @ _quat_matrix(site_quat)
    return pos, rot, np.stack(origins, 1), np.stack(axes, 1)


def _leg_home(home_joint_pos_rad, side: str) -> np.ndarray:
    names = LEG_JOINT_NAMES[:6] if side == "right" else LEG_JOINT_NAMES[6:]
    return np.array([float(home_joint_pos_rad[name]) for name in names])


def solve_offsets(home_joint_pos_rad, trunk_pitch_rad: float, offsets) -> np.ndarray:
    """Exact leg offsets (N, 12) from HOME for two-foot targets ``offsets``
    (N, 3; dx, dy, dz in the HOME-levelled trunk frame): both foot sites keep
    their HOME pose while the trunk moves by -(dx, dy, dz)."""

    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    c, s = math.cos(trunk_pitch_rad), math.sin(trunk_pitch_rad)
    # Levelled-frame vectors in the trunk frame: R_y(-pitch).
    levelled_to_trunk = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    shift = offsets @ levelled_to_trunk.T
    result = []
    for side in ("right", "left"):
        home = _leg_home(home_joint_pos_rad, side)
        home_pos, home_rot, _, _ = leg_fk(side, home[None])
        goal_pos = home_pos + shift
        q = np.tile(home, (len(offsets), 1))
        q[:, 3] += _START_KNEE_RAD
        for _ in range(_SOLVE_ITERATIONS):
            pos, rot, origins, axes = leg_fk(side, q)
            delta = home_rot @ np.swapaxes(rot, 1, 2)
            angle_error = 0.5 * np.stack(
                (
                    delta[:, 2, 1] - delta[:, 1, 2],
                    delta[:, 0, 2] - delta[:, 2, 0],
                    delta[:, 1, 0] - delta[:, 0, 1],
                ),
                -1,
            )
            error = np.concatenate((goal_pos - pos, angle_error), -1)
            if np.abs(error).max() < _SOLVE_TOLERANCE:
                break
            linear = np.cross(axes, pos[:, None, :] - origins)
            jacobian = np.concatenate((linear, axes), -1).swapaxes(1, 2)
            q = q + np.linalg.solve(jacobian, error[..., None])[..., 0]
        else:
            raise RuntimeError("crouch leg IK did not converge")
        if not np.all(q[:, 3] >= 0.0):
            # The other branch: the knee bent backwards.
            raise RuntimeError("crouch leg IK found a backward knee")
        result.append(q - home)
    return np.concatenate(result, -1)


def table_axes() -> tuple[np.ndarray, np.ndarray]:
    """The grid: dx = dy nodes (m) and sqrt(dz) nodes (sqrt(m))."""

    xy = np.linspace(-XY_LIMIT_M, XY_LIMIT_M, XY_NODES)
    sqrt_z = np.linspace(math.sqrt(FLOOR_BAND_M), math.sqrt(Z_MAX_M), SQRT_Z_NODES)
    return xy, sqrt_z


def build_table(home_joint_pos_rad, trunk_pitch_rad: float) -> dict[str, np.ndarray]:
    """The offsets at every grid node, for ``leg_offsets``.  Values are
    (dx, dy, sqrt(dz), 12); the index vectors are the hat-function centres."""

    xy, sqrt_z = table_axes()
    grid = np.stack(np.meshgrid(xy, xy, sqrt_z**2, indexing="ij"), -1).reshape(-1, 3)
    values = solve_offsets(home_joint_pos_rad, trunk_pitch_rad, grid)
    return {
        "values": values.reshape(XY_NODES, XY_NODES, SQRT_Z_NODES, len(LEG_JOINT_NAMES)),
        "xy_index": np.arange(XY_NODES, dtype=np.float64),
        "z_index": np.arange(SQRT_Z_NODES, dtype=np.float64),
    }


def leg_offsets(table, foot):
    """Target leg offsets (N, 12) for published foot targets ``foot`` (N, 6:
    left dx, dy, dz, right dx, dy, dz).  ``table`` is ``build_table``'s, as
    numpy arrays or as torch tensors on ``foot``'s device (the same code runs
    on both)."""

    left_z, right_z = foot[:, 2], foot[:, 5]
    inside = (
        (abs(foot[:, 0]) <= XY_LIMIT_M)
        & (abs(foot[:, 1]) <= XY_LIMIT_M)
        & (abs(foot[:, 3]) <= XY_LIMIT_M)
        & (abs(foot[:, 4]) <= XY_LIMIT_M)
        & (left_z <= Z_MAX_M)
        & (right_z <= Z_MAX_M)
    )
    active = (left_z > FLOOR_BAND_M) & (right_z > FLOOR_BAND_M) & inside
    dx = (foot[:, 0] + foot[:, 3]) * 0.5
    dy = (foot[:, 1] + foot[:, 4]) * 0.5
    dz = (left_z + right_z - abs(left_z - right_z)) * 0.5
    xy_step = 2.0 * XY_LIMIT_M / (XY_NODES - 1)
    z_low, z_high = math.sqrt(FLOOR_BAND_M), math.sqrt(Z_MAX_M)
    z_step = (z_high - z_low) / (SQRT_Z_NODES - 1)
    ux = ((dx + XY_LIMIT_M) / xy_step).clip(0.0, XY_NODES - 1)
    uy = ((dy + XY_LIMIT_M) / xy_step).clip(0.0, XY_NODES - 1)
    uz = ((dz.clip(FLOOR_BAND_M, Z_MAX_M) ** 0.5 - z_low) / z_step).clip(0.0, SQRT_Z_NODES - 1)
    # Piecewise-linear (hat) weights per axis: trilinear interpolation.
    wx = (1.0 - abs(ux[:, None] - table["xy_index"][None, :])).clip(0.0, None)
    wy = (1.0 - abs(uy[:, None] - table["xy_index"][None, :])).clip(0.0, None)
    wz = (1.0 - abs(uz[:, None] - table["z_index"][None, :])).clip(0.0, None)
    values = table["values"]
    rows = foot.shape[0]
    joints = values.shape[-1]
    by_x = (wx @ values.reshape(XY_NODES, -1)).reshape(rows, XY_NODES, -1)
    by_y = (wy[:, :, None] * by_x).sum(1).reshape(rows, SQRT_Z_NODES, joints)
    return (wz[:, :, None] * by_y).sum(1) * active[:, None]


def follow(offsets, target):
    """The offsets in use after one policy step (N, 12): ``offsets`` moved
    towards ``target`` by at most ``MAX_STEP_RAD`` (joint-space norm), along
    the straight line between them; ``target`` itself within reach (zero
exactly when it is zero)."""

    step = target - offsets
    norm = ((step * step).sum(-1) ** 0.5)[:, None]
    return offsets + step * (MAX_STEP_RAD / norm.clip(MAX_STEP_RAD, None))
