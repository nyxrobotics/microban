"""Regenerate tests/fixtures/walk_policy_v2.onnx, a tiny synthetic walk actor.

The runtime venv has no ``onnx`` package, so run this with an environment that
does (for example the mjlab_microban venv), from the repository root:

    PYTHONPATH=src <python-with-onnx> tests/fixtures/make_walk_policy_fixture.py

The actor is linear: action[i] = 0.5 * joint_pos_residual[i] + BIAS[i], which
lets tests predict its output.  Its metadata is the deployed walking contract
(walk_contract_version v2_centered_home_clip157) with a full-precision
default_joint_pos equal to NEUTRAL_POSE.
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_fixtures import (  # noqa: E402
    ACTION_COUNT,
    WALK_BIAS,
    WALK_OBS_WIDTH,
    WALK_POLICY_FIXTURE,
    WALK_POSITION_GAIN,
    walk_contract_metadata,
)


def main() -> None:
    weight = np.zeros((WALK_OBS_WIDTH, ACTION_COUNT), dtype=np.float32)
    for index in range(ACTION_COUNT):
        weight[6 + index, index] = WALK_POSITION_GAIN
    bias = np.asarray(WALK_BIAS, dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "weight"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["actions"]),
        ],
        "walk_policy_fixture",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, WALK_OBS_WIDTH])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_COUNT])],
        initializer=[
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for key, value in walk_contract_metadata().items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, WALK_POLICY_FIXTURE)
    print(f"wrote {WALK_POLICY_FIXTURE}")


if __name__ == "__main__":
    main()
