"""Run the regular robot runtime with the robot-local GC300 input source.

Keep the policy-bound ``main.py`` unchanged.  The input source is selected here
before entering its existing setup, scheduler, and cleanup path.
"""

import main as robot_main
from input.gc300_input import Gc300InputSource


def main() -> None:
    robot_main.build_input_source = Gc300InputSource
    robot_main.main()


if __name__ == "__main__":
    main()
