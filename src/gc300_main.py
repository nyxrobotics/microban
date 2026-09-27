"""Compare GC300 walking with the 2026-09-27 morning pose and walk actor.

Keep the current PICO runtime and its authenticated ONNX source files intact.
The GC300 entry point selects the older pose and actor only in its own process.
"""

import math

import main as robot_main
from constants import NEUTRAL_POSE
from input.gc300_input import Gc300InputSource
from moves import walk as walk_module
from moves.walk import WalkMove


# The final commit from the morning of 2026-09-27 was 6a6cae4 (11:38 JST).
# walk.onnx is byte-identical at that commit and at this branch's base.
GC300_AGENT_NAME = "walk.onnx"
GC300_MORNING_NEUTRAL_POSE = {
    **NEUTRAL_POSE,
    "left_hip_pitch": math.radians(-10.0),
    "right_hip_pitch": math.radians(-10.0),
    "left_ankle_pitch": 0.0,
    "right_ankle_pitch": 0.0,
    "left_shoulder_pitch": math.radians(10.0),
    "right_shoulder_pitch": math.radians(10.0),
}


class MorningGc300Scheduler(robot_main.Scheduler):
    """Use the old A/R3-off stance while keeping the current gate logic."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hardware_neutral_pose = dict(GC300_MORNING_NEUTRAL_POSE)


def main() -> None:
    robot_main.build_input_source = Gc300InputSource
    walk_module.AGENT_NAME = GC300_AGENT_NAME
    # WalkMove's normal stop and feedback fallbacks also need the old neutral.
    # Rebinding this module variable affects only the GC300 process; no source
    # authenticated by the PICO actor is edited on this comparison branch.
    walk_module.NEUTRAL_POSE = GC300_MORNING_NEUTRAL_POSE
    robot_main.PolicySelectableWalkMove = WalkMove
    robot_main.Scheduler = MorningGc300Scheduler
    print("GC300 comparison: 2026-09-27 morning neutral and walk.onnx", flush=True)
    robot_main.main()


if __name__ == "__main__":
    main()
