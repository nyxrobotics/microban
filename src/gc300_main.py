"""Run the robot-local GC300 with its separately trained walking actor.

Keep the PICO policy-bound ``main.py`` and ``walk.onnx`` unchanged.  Select the
GC300 input and actor before entering the regular setup, scheduler, and cleanup.
"""

import main as robot_main
from input.gc300_input import Gc300InputSource
from moves import walk as walk_module
from moves.walk import WalkMove


GC300_AGENT_NAME = "gc300_walk.onnx"


def main() -> None:
    robot_main.build_input_source = Gc300InputSource
    walk_module.AGENT_NAME = GC300_AGENT_NAME
    robot_main.PolicySelectableWalkMove = WalkMove
    robot_main.main()


if __name__ == "__main__":
    main()
