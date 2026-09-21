# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Open the robot's MJCF model in a MeshCat browser viewer with a Tk slider
per joint and Visual/Collision/Inertial checkboxes, for interactively
checking mesh alignment and joint limits. See robot_viewer.py for details.

Usage:
    PYTHONPATH=src uv run python3 src/debug/view_mjcf_sliders.py [model_dir]
"""

import sys

from robot_viewer import run

if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "src/model/mjcf"
    run(model_dir, mjcf=True, default_visual=True, default_collision=False, title="Joint sliders - MJCF")
