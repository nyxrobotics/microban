# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Open the robot's URDF in a MeshCat browser viewer with a Tk slider per
joint and Visual/Collision/Inertial checkboxes, for checking meshes/origins.

Loads src/model/urdf/robot.urdf directly (not the MJCF used by the sim/main
viewer), so it reflects exactly what's authored in the URDF - useful right
after hand-editing link origins or adding new links, before regenerating the
MJCF from it. robot.urdf carries no <visual> geometry (that's MJCF-only in
this project - see view_mjcf_sliders.py for the actual meshes), so the
Visual checkbox is a no-op here; Collision is on by default instead, since
that's what robot.urdf actually defines. See robot_viewer.py for details.

Usage:
    PYTHONPATH=src uv run python3 src/debug/view_urdf.py [model_dir]
"""

import sys

from robot_viewer import run

if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "src/model/urdf"
    run(model_dir, mjcf=False, default_visual=False, default_collision=True, title="Joint sliders - URDF")
