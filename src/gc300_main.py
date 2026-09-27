"""Run the robot-local GC300 with its separately trained walking actor.

Keep the PICO policy-bound ``main.py`` and ``walk.onnx`` unchanged.  Select the
GC300 input and actor before entering the regular setup, scheduler, and cleanup.
"""

from functools import partial
import math

import main as robot_main
from input.gc300_input import Gc300InputSource
from moves import walk as walk_module
from moves.walk import WalkMove


GC300_AGENT_NAME = "gc300_walk.onnx"
# In model joint coordinates both ankles use the same sign.  With soles planted,
# negative ankle pitch leans the trunk forward; servo mirror signs are applied
# later by RobotController.
GC300_FORWARD_ANKLE_BIAS_RAD = -math.radians(2.0)


def main() -> None:
    robot_main.build_input_source = Gc300InputSource
    walk_module.AGENT_NAME = GC300_AGENT_NAME
    robot_main.PolicySelectableWalkMove = partial(
        WalkMove, ankle_pitch_bias_rad=GC300_FORWARD_ANKLE_BIAS_RAD
    )
    print("GC300 active-policy ankle pitch bias: -2 degrees (forward)", flush=True)
    robot_main.main()


if __name__ == "__main__":
    main()
