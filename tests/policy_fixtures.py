"""Synthetic policies carrying the deployed (centered HOME, +-pi servo range) contracts.

``WALK_POLICY_FIXTURE`` is a tiny real ONNX walk actor (see
fixtures/make_walk_policy_fixture.py); ``FakeSession`` lets a test vary one
metadata field or the output without writing a new ONNX file.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from constants import (
    HOME_ROOT_POS_Z_M,
    HOME_ROOT_QUAT_WXYZ,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    SERVO_TARGET_RANGE_RAD,
)
from home_pose import HOME_CONTRACTS

WALK_POLICY_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "walk_policy_v3.onnx"
# mjlab's natural joint order (robot.joint_names) as in exported Microban ONNX.
JOINT_NAMES = ["head", "neck_roll", "neck_pitch", *OBSERVATION_DOF_ORDER]
ACTION_COUNT = len(OBSERVATION_DOF_ORDER)
WALK_OBS_WIDTH = 3 + 3 + 3 * ACTION_COUNT + 3
GETUP_OBS_WIDTH = 3 + 3 + 3 * ACTION_COUNT
# The fixture actor: action[i] = WALK_POSITION_GAIN * joint_pos_residual[i] + WALK_BIAS[i].
WALK_POSITION_GAIN = 0.5
WALK_BIAS = [0.01 * ((index % 5) - 2) + 0.005 for index in range(ACTION_COUNT)]

# The pre-unification HOME (hip pitch -10 deg, ankle pitch 0) a stale model
# may still have been trained at.
OLD_HOME = {
    **NEUTRAL_POSE,
    "left_hip_pitch": -0.17453292519943295,
    "right_hip_pitch": -0.17453292519943295,
    "left_ankle_pitch": 0.0,
    "right_ankle_pitch": 0.0,
}


def csv(values) -> str:
    return ",".join(repr(float(value)) for value in values)


def home_pose_stamp(home: dict[str, float] | None = None) -> str:
    """The exporters' full-precision HOME stamp (walk home_pose, get-up home_pose)."""
    home = NEUTRAL_POSE if home is None else home
    return json.dumps(
        {
            "root_pos_m": [0.0, 0.0, HOME_ROOT_POS_Z_M],
            "root_quat_wxyz": list(HOME_ROOT_QUAT_WXYZ),
            "joint_pos_rad": {name: float(home[name]) for name in sorted(home)},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def walk_contract_metadata(home: dict[str, float] | None = None) -> dict[str, str]:
    home = NEUTRAL_POSE if home is None else home
    return {
        "walk_contract_version": HOME_CONTRACTS["walk_contract_version"],
        "home_pose": home_pose_stamp(home),
        "previous_action_semantics": "raw_policy_output",
        "joint_names": ",".join(JOINT_NAMES),
        "default_joint_pos": csv(home[name] for name in JOINT_NAMES),
        "action_joint_names": ",".join(OBSERVATION_DOF_ORDER),
        "action_scale": "1.0",
        "action_clip_lower": csv([-SERVO_TARGET_RANGE_RAD] * ACTION_COUNT),
        "action_clip_upper": csv([SERVO_TARGET_RANGE_RAD] * ACTION_COUNT),
        "observation_names": "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command",
        "command_names": "twist",
        "run_path": "synthetic_test_fixture",
    }


def getup_contract_metadata(home: dict[str, float] | None = None) -> dict[str, str]:
    """What mjlab_microban's export_getup_onnx.py writes at this HOME.

    The contract and checkpoint stamp are the HOME's (config/home_pose.yaml):
    "v5" / "v5" at the centered HOME, "v6" / "v6" at the forward-lean one.

    default_joint_pos goes through mjlab's 3-decimal CSV formatter; the +-pi
    servo-range clip is written at full precision (mjlab_microban 1290a1e).
    """
    home = NEUTRAL_POSE if home is None else home
    return {
        "microban_getup_contract": HOME_CONTRACTS["getup_contract_version"],
        "microban_getup_angular_velocity_frame": "imu_sensor_xyz",
        "microban_getup_previous_action_semantics": "raw_policy_output",
        "joint_names": ",".join(JOINT_NAMES),
        "default_joint_pos": ",".join(f"{home[name]:.3f}" for name in JOINT_NAMES),
        "action_joint_names": ",".join(OBSERVATION_DOF_ORDER),
        "observation_names": "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions",
        "action_clip_lower": csv([-SERVO_TARGET_RANGE_RAD] * ACTION_COUNT),
        "action_clip_upper": csv([SERVO_TARGET_RANGE_RAD] * ACTION_COUNT),
        "action_scale": "1.0",
        "microban_getup_checkpoint_contract_stamp": (
            HOME_CONTRACTS["getup_checkpoint_stamp"] or HOME_CONTRACTS["getup_contract_version"]
        ),
        "checkpoint_sha256": "0" * 64,
        "microban_getup_home_pose": home_pose_stamp(home),
    }


class FakeSession:
    """Just enough of onnxruntime.InferenceSession for the move constructors."""

    def __init__(self, metadata, *, input_width, output_width=ACTION_COUNT, outputs=None):
        self.metadata = dict(metadata)
        self.input_width = input_width
        self.output_width = output_width
        # Queue of raw actions returned by successive run() calls (last one repeats).
        self.outputs = list(outputs) if outputs is not None else [[0.0] * output_width]
        self.calls: list[list[float]] = []

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)

    def get_inputs(self):
        return [SimpleNamespace(name="obs", shape=[1, self.input_width])]

    def get_outputs(self):
        return [SimpleNamespace(name="actions", shape=[1, self.output_width])]

    def run(self, _names, feeds):
        self.calls.append(list(feeds["obs"][0]))
        action = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return [[list(action)]]
