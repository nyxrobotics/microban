# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Run the robot-local GC300: main.py with the GC300 input source and the
walking policy (``WalkMove``) in place of the PICO policy selector.
"""

import main as robot_main
from input.gc300_input import Gc300InputSource
from moves.walk import WalkMove


def main() -> None:
    robot_main.build_input_source = Gc300InputSource
    robot_main.PolicySelectableWalkMove = WalkMove
    robot_main.main()


if __name__ == "__main__":
    main()
