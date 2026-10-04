"""Run the robot-local GC300 with the shared walking actor.

GC300 uses the same single walking model (``walk.onnx``), the same forward-lean
HOME (``NEUTRAL_POSE``) and the same target rule as every other input source;
there is no GC300-specific model or ankle bias.  Only the input source and the
plain ``WalkMove`` (no PICO policy selector) differ from ``main.py``.
"""

import main as robot_main
from input.gc300_input import Gc300InputSource
from moves import walk as walk_module
from moves.walk import WalkMove


GC300_AGENT_NAME = walk_module.AGENT_NAME


def main() -> None:
    robot_main.build_input_source = Gc300InputSource
    robot_main.PolicySelectableWalkMove = WalkMove
    robot_main.main()


if __name__ == "__main__":
    main()
