# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Parametric BAM pendulum arm generator (XC330-T288-T actuator identification).

A straight bar: a 4-hole pivot pattern near one end matching the servo's stock
horn (screw straight into the horn's own 4 holes - no separate horn adapter
needed), and a row of holes further out to hang a known mass (bolt/nuts) at an
adjustable, precisely-measurable radius - the "length" bam's record script
wants is whichever mass-hole's x you actually use.

The pivot hole pattern (4x ~2mm holes on a 6.0mm-radius bolt circle, 90 deg
apart) was measured directly off cad/stl/idle_horn.stl - the same horn CAD
already used and verified on this robot's other joints - by cross-sectioning
the mesh and fitting circles to the small loops. ROBOTIS's own XC330-T288-T
dimensional drawing is a PDF/image (not machine-readable here) so this is the
best available source; re-measure against the datasheet PDF or calipers on
the real horn if that pattern is ever suspect.

Needs trimesh + a CSG backend, not project dependencies (this runs once to
produce pendulum_arm.stl, not part of the identification pipeline itself):
    uv run --with trimesh --with manifold3d --with numpy-stl \\
        python3 tools/actuator_id/make_pendulum_arm.py
"""
from pathlib import Path

import numpy as np
import trimesh

LENGTH = 200.0       # mm, pivot to far end
WIDTH = 22.0          # mm (wide enough to clear the 12mm horn bolt circle)
THICK = 4.0           # mm, PETG-CF at 100% infill per this project's convention

PIVOT_X = 12.0                  # mm from the arm's pivot end to the horn's center
PIVOT_HOLE_D = 2.0               # mm, matches idle_horn.stl's measured mounting holes
PIVOT_BOLT_CIRCLE_RADIUS = 6.0   # mm, measured from idle_horn.stl
PIVOT_HOLE_COUNT = 4             # measured from idle_horn.stl (90 deg apart)

MASS_HOLE_D = 4.2     # mm, clearance for M4 bolt (hang nuts/washers of known mass)
MASS_HOLE_START = 50.0   # mm from pivot center
MASS_HOLE_SPACING = 20.0 # mm between mass holes
MASS_HOLE_COUNT = 8      # holes at 50, 70, ..., 190 mm from the pivot center

FILLET_SEGMENTS = 32


def hole(diameter: float, x: float, y: float = 0.0) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=diameter / 2, height=THICK * 3, sections=FILLET_SEGMENTS)
    c.apply_translation([x, y, 0])
    return c


def main() -> None:
    arm = trimesh.creation.box(extents=[LENGTH, WIDTH, THICK])
    # box() is centered at the origin; shift so x=0 is the pivot end
    arm.apply_translation([LENGTH / 2, 0, 0])

    cutters = []
    for i in range(PIVOT_HOLE_COUNT):
        angle = 2 * np.pi * i / PIVOT_HOLE_COUNT
        x = PIVOT_X + PIVOT_BOLT_CIRCLE_RADIUS * np.cos(angle)
        y = PIVOT_BOLT_CIRCLE_RADIUS * np.sin(angle)
        cutters.append(hole(PIVOT_HOLE_D, x, y))
    for i in range(MASS_HOLE_COUNT):
        x = MASS_HOLE_START + i * MASS_HOLE_SPACING
        cutters.append(hole(MASS_HOLE_D, x))

    result = arm
    for cutter in cutters:
        result = result.difference(cutter)

    out_path = Path(__file__).parent / "pendulum_arm.stl"
    result.export(out_path)
    print(f"Exported {out_path}: {LENGTH}x{WIDTH}x{THICK} mm, "
          f"pivot bolt circle at x={PIVOT_X}mm (r={PIVOT_BOLT_CIRCLE_RADIUS}mm, "
          f"{PIVOT_HOLE_COUNT} holes), mass holes at "
          f"{[MASS_HOLE_START + i*MASS_HOLE_SPACING for i in range(MASS_HOLE_COUNT)]} mm")
    print(f"Volume: {result.volume:.1f} mm^3, watertight: {result.is_watertight}")


if __name__ == "__main__":
    main()
