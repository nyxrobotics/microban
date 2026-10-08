"""Synthetic sessions carrying contract microban-policy-1 at this process's HOME.

``contract_metadata(kind)`` is what the training exporter writes for a
walk / getup / pico policy trained at config/home_pose.yaml's HOME (or at
another ``home``); ``FakeSession`` lets a test vary one metadata field or the
outputs without an ONNX file, and ``LinearWalkSession`` is a tiny walk actor
whose output a test can predict.  The real policy package of a training dry run
is in tests/fixtures/policies (see test_policy_contract.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from constants import (
    HOME_PROJECTED_GRAVITY,
    HOME_ROOT_POS_Z_M,
    HOME_ROOT_QUAT_WXYZ,
    NEUTRAL_POSE,
    OBSERVATION_DOF_ORDER,
    SERVO_TARGET_RANGE_RAD,
)
import policy_contract as pc

FIXTURES = Path(__file__).resolve().parent / "fixtures"
POLICY_PACKAGE = FIXTURES / "policies"
LEAN_HOME_YAML = FIXTURES / "home_pose_forward_lean.yaml"
CENTERED_HOME_YAML = FIXTURES / "home_pose_centered.yaml"

# mjlab's natural joint order (robot.joint_names) as in exported Microban ONNX.
JOINT_NAMES = ["head", "neck_roll", "neck_pitch", *OBSERVATION_DOF_ORDER]
ACTION_COUNT = len(OBSERVATION_DOF_ORDER)
WIDTHS = {kind: sum(width for _, width in schema) for kind, schema in pc.OBSERVATION_SCHEMAS.items()}
WALK_OBS_WIDTH = WIDTHS["walk"]
GETUP_OBS_WIDTH = WIDTHS["getup"]
PICO_OBS_WIDTH = WIDTHS["pico"]
WALK_CHECKPOINT_SHA256 = "1" * 64
PICO_RAW_ACTION_GUARD = 24.0
# The linear walk actor: action[i] = WALK_POSITION_GAIN * joint_pos_residual[i] + WALK_BIAS[i].
WALK_POSITION_GAIN = 0.5
WALK_BIAS = [0.01 * ((index % 5) - 2) + 0.005 for index in range(ACTION_COUNT)]

# A HOME other than this robot's (hip pitch -10 deg, ankle pitch 0).
OTHER_HOME = {
    **NEUTRAL_POSE,
    "left_hip_pitch": -0.17453292519943295,
    "right_hip_pitch": -0.17453292519943295,
    "left_ankle_pitch": 0.0,
    "right_ankle_pitch": 0.0,
}


def csv(values) -> str:
    return ",".join(repr(float(value)) for value in values)


def home_pose_stamp(home: dict[str, float] | None = None) -> str:
    """The exporters' full-precision ``home_pose`` stamp."""
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


def home_observation(kind: str) -> list[float]:
    """The observation of the robot standing still at HOME."""
    row = [0.0] * WIDTHS[kind]
    row[3:6] = [float(value) for value in HOME_PROJECTED_GRAVITY]
    return row


# A PICO self-test row with a lifted left foot and moved arms (inside the box).
PICO_TARGET_ROW_FOOT = [0.01, 0.0, 0.03, 0.0, 0.0, 0.0]
PICO_TARGET_ROW_ARMS = [-0.8, 0.3, -0.5, 0.4, -0.2, 0.1]


def self_test_rows(kind: str) -> list[list[float]]:
    """HOME rows; a PICO self-test also records a row with foot and arm targets."""
    rows = [home_observation(kind) for _ in range(pc.SELF_TEST_MIN_ROWS)]
    if kind == "pico":
        rows[0][69:81] = PICO_TARGET_ROW_FOOT + PICO_TARGET_ROW_ARMS
    return rows


def contract_metadata(
    kind: str,
    home: dict[str, float] | None = None,
    *,
    self_test_actions: list[list[float]] | None = None,
) -> dict[str, str]:
    """Contract microban-policy-1 metadata of a ``kind`` policy at ``home``."""
    home = NEUTRAL_POSE if home is None else home
    rows = self_test_rows(kind)
    actions = self_test_actions or [[0.0] * ACTION_COUNT] * len(rows)
    metadata = {
        "microban_policy_contract": pc.POLICY_CONTRACT,
        "microban_policy_kind": kind,
        "microban_recipe": pc.RECIPES[kind],
        "home_pose": home_pose_stamp(home),
        "joint_names": ",".join(JOINT_NAMES),
        "default_joint_pos": csv(home[name] for name in JOINT_NAMES),
        "action_joint_names": ",".join(OBSERVATION_DOF_ORDER),
        "action_scale": "1.0",
        "action_clip_lower": csv([-SERVO_TARGET_RANGE_RAD] * ACTION_COUNT),
        "action_clip_upper": csv([SERVO_TARGET_RANGE_RAD] * ACTION_COUNT),
        "observation_schema_json": json.dumps([list(term) for term in pc.OBSERVATION_SCHEMAS[kind]]),
        "observation_joint_names": ",".join(pc.OBSERVATION_JOINT_NAMES[kind]),
        "previous_action_semantics": "raw_policy_output",
        "base_ang_vel_frame": "imu_sensor_xyz",
        "control_hz": "50",
        "checkpoint_filename": "model_8999.pt",
        "checkpoint_iteration": "8999",
        "checkpoint_sha256": WALK_CHECKPOINT_SHA256 if kind == "walk" else "2" * 64,
        "gate_status": "pass",
        "gate_report_sha256": "3" * 64,
        "self_test_observations_json": json.dumps(rows),
        "self_test_actions_json": json.dumps(actions),
    }
    if kind == "pico":
        metadata.update(
            {
                "pico_walk_checkpoint_sha256": WALK_CHECKPOINT_SHA256,
                "pico_target_frame": pc.PICO_TARGET_FRAME,
                "pico_foot_target_lower_json": json.dumps(list(pc.PICO_FOOT_TARGET_LOWER)),
                "pico_foot_target_upper_json": json.dumps(list(pc.PICO_FOOT_TARGET_UPPER)),
                "pico_both_feet_target_lower_json": json.dumps(list(pc.PICO_BOTH_FEET_TARGET_LOWER)),
                "pico_both_feet_target_upper_json": json.dumps(list(pc.PICO_BOTH_FEET_TARGET_UPPER)),
                "pico_arm_target_json": json.dumps(
                    {
                        "contract": pc.PICO_ARM_TARGET_CONTRACT,
                        "joint_names": list(pc.PICO_ARM_TARGET_JOINTS),
                        "lower_rad": list(pc.PICO_ARM_TARGET_LOWER_RAD),
                        "upper_rad": list(pc.PICO_ARM_TARGET_UPPER_RAD),
                        "slew_rad_s": 4.0,
                    },
                    separators=(",", ":"),
                ),
                "pico_raw_action_guard_json": json.dumps([PICO_RAW_ACTION_GUARD] * ACTION_COUNT),
                "pico_curriculum_json": json.dumps(
                    {
                        "critic_warmup": 1000,
                        "arm_start": 1000,
                        "foot_start": 4000,
                        "foot_tighten": 6000,
                        "total": 9000,
                    }
                ),
                "pico_active_adapter_columns_json": json.dumps(list(pc.PICO_ADAPTER_COLUMNS)),
            }
        )
    return metadata


def walk_contract_metadata(home: dict[str, float] | None = None) -> dict[str, str]:
    return contract_metadata("walk", home)


def getup_contract_metadata(home: dict[str, float] | None = None) -> dict[str, str]:
    return contract_metadata("getup", home)


def pico_contract_metadata(home: dict[str, float] | None = None) -> dict[str, str]:
    return contract_metadata("pico", home)


class FakeSession:
    """Just enough of onnxruntime.InferenceSession for the move constructors."""

    def __init__(
        self,
        metadata,
        *,
        input_width,
        output_width=ACTION_COUNT,
        outputs=None,
        input_name="obs",
        output_name="actions",
        input_type="tensor(float)",
        output_type="tensor(float)",
    ):
        self.metadata = dict(metadata)
        self.input_width = input_width
        self.output_width = output_width
        self.input_name = input_name
        self.output_name = output_name
        self.input_type = input_type
        self.output_type = output_type
        # Queue of raw actions returned by successive run() calls (last one repeats).
        self.outputs = list(outputs) if outputs is not None else [[0.0] * output_width]
        self.calls: list[list[float]] = []

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self.metadata)

    def get_inputs(self):
        return [SimpleNamespace(name=self.input_name, shape=[1, self.input_width], type=self.input_type)]

    def get_outputs(self):
        return [SimpleNamespace(name=self.output_name, shape=[1, self.output_width], type=self.output_type)]

    def compute(self, observation: list[float]):
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]

    def run(self, _names, feeds):
        observation = list(np.asarray(feeds[self.input_name], dtype=np.float64).reshape(-1))
        if len(observation) != self.input_width:
            raise ValueError(f"fed {len(observation)} observations to a {self.input_width}-wide input")
        self.calls.append(observation)
        return [np.asarray([self.compute(observation)])]


class LinearWalkSession(FakeSession):
    """action[i] = WALK_POSITION_GAIN * joint_pos_residual[i] + WALK_BIAS[i]."""

    def __init__(self, metadata=None):
        super().__init__(
            walk_contract_metadata() if metadata is None else metadata, input_width=WALK_OBS_WIDTH
        )

    def compute(self, observation: list[float]):
        return [
            WALK_POSITION_GAIN * observation[6 + index] + WALK_BIAS[index]
            for index in range(ACTION_COUNT)
        ]


def fake_session(kind: str, metadata=None, **kwargs) -> FakeSession:
    return FakeSession(
        contract_metadata(kind) if metadata is None else metadata,
        input_width=kwargs.pop("input_width", WIDTHS[kind]),
        **kwargs,
    )
