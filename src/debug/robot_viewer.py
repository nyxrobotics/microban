# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Shared MeshCat + Tk-sliders viewer, used by view_urdf.py and
view_mjcf_sliders.py - both just point this at a different model directory.

Visual / Collision / Inertial (CoM) checkboxes: Visual and Collision toggle
the two geometry sets a model can define per body (pinocchio's
MeshcatVisualizer.displayVisuals/displayCollisions); Inertial draws a small
sphere at each link's center of mass (from its <inertial> origin), useful for
spotting a mass origin that doesn't sit where the mesh/collision actually is.
robot.urdf has no <visual> meshes in this project (see view_urdf.py), so that
checkbox is simply a no-op there.
"""

import tkinter as tk

import numpy as np
import placo
from placo_utils.visualization import get_viewer, point_viz, robot_viz

FREEJOINT_SUFFIXES = ("_freejoint",)
COM_MARKER_COLOR = 0x00FFFF
COM_MARKER_RADIUS = 0.006


def run(model_dir: str, mjcf: bool, default_visual: bool, default_collision: bool, title: str = "Joint sliders") -> None:
    robot = placo.RobotWrapper(model_dir, placo.Flags.mjcf) if mjcf else placo.RobotWrapper(model_dir)
    viz = robot_viz(robot)
    viz.display(robot.state.q)

    joint_names = [n for n in robot.joint_names() if not n.endswith(FREEJOINT_SUFFIXES)]

    # (frame_name, local CoM offset) for every body whose world frame we can
    # resolve unambiguously - a body/joint name collision (e.g. "head" is
    # both a joint and its body) makes get_T_world_frame() raise, so those
    # few links are just skipped for the CoM marker.
    com_links: list[tuple[str, np.ndarray]] = []
    for i in range(1, robot.model.njoints):
        name = robot.model.names[i]
        inertia = robot.model.inertias[i]
        if inertia.mass <= 0:
            continue
        try:
            robot.get_T_world_frame(name)
        except (RuntimeError, ValueError):
            continue
        com_links.append((name, np.array(inertia.lever).reshape(3)))

    root = tk.Tk()
    root.title(title)

    show_inertial = tk.BooleanVar(value=False)

    def update_com_markers() -> None:
        if show_inertial.get():
            for name, lever in com_links:
                T = robot.get_T_world_frame(name)
                com_world = (T @ np.append(lever, 1.0))[:3]
                point_viz(f"com_{name}", com_world, radius=COM_MARKER_RADIUS, color=COM_MARKER_COLOR)
        else:
            get_viewer()["point"].delete()

    def refresh() -> None:
        robot.update_kinematics()
        viz.display(robot.state.q)
        update_com_markers()

    def make_slider_callback(name: str):
        def callback(value: str) -> None:
            robot.set_joint(name, float(value))
            refresh()
        return callback

    toggles = tk.Frame(root)
    toggles.pack(fill="x", padx=8, pady=6)

    show_visual = tk.BooleanVar(value=default_visual)
    show_collision = tk.BooleanVar(value=default_collision)
    tk.Checkbutton(toggles, text="Visual", variable=show_visual, command=lambda: viz.displayVisuals(show_visual.get())).pack(side="left", padx=4)
    tk.Checkbutton(toggles, text="Collision", variable=show_collision, command=lambda: viz.displayCollisions(show_collision.get())).pack(side="left", padx=4)
    tk.Checkbutton(toggles, text="Inertial (CoM)", variable=show_inertial, command=update_com_markers).pack(side="left", padx=4)

    viz.displayVisuals(show_visual.get())
    viz.displayCollisions(show_collision.get())

    for name in joint_names:
        lower, upper = robot.get_joint_limits(name)
        row = tk.Frame(root)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=name, width=20, anchor="w").pack(side="left")
        tk.Scale(
            row,
            from_=lower,
            to=upper,
            resolution=0.001,
            orient="horizontal",
            length=300,
            command=make_slider_callback(name),
        ).pack(side="left")

    print("Slider window ready. Toggle Visual/Collision/Inertial, move a slider to update the view.")
    root.mainloop()
