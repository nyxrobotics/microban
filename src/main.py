# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time
import numpy as np
import os
from pathlib import Path

from constants import MOTOR_TO_ID, NEUTRAL_POSE, KP_DEFAULT
from robot_controller import RobotController
from scheduler import Scheduler
from input.input_source import InputSource
from input.keyboard_input import KeyboardInputSource
from moves.hmd_head import HmdHeadTrackingMove
from moves.getup import GetupMove
from moves.pico_arms import PicoArmTrackingMove
from moves.policy_selector import PolicySelectableWalkMove
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from moves.walk import WalkMove

PID_FILE = Path("/tmp/microban_scheduler.pid")


def _another_session_running() -> bool:
    """True if a live control loop already owns the PID file (e.g. launched by the
    gamepad daemon). Avoids two instances fighting over the motor bus."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
        os.kill(pid, 0)
    except (ValueError, OSError):
        return False
    return True

# Which moves can be toggled from the gamepad / keyboard. "getup" is also switched
# automatically by the scheduler on a sustained fall (see Scheduler._update_getup_override);
# the key/button here is a manual override for testing. A/B/R3 are reserved for get-up policy testing
# (see GamepadInputSource) and cannot be reassigned here.
MOVE_KEYS = {"h": "head", "s": "squat", "v": "walk", "g": "getup"}
GAMEPAD_BUTTON_MOVES = {"X": "walk"}


def build_input_source() -> InputSource:
    """Use the gamepad when one is connected, otherwise fall back to the keyboard.

    Override with MICROBAN_INPUT=keyboard|gamepad|network. Network mode listens for
    the external VR bridge on MICROBAN_NETWORK_PORT (default 5555).
    """
    requested = os.environ.get("MICROBAN_INPUT", "auto").lower()

    if requested == "network":
        from input.network_input import NetworkInputSource

        port = int(os.environ.get("MICROBAN_NETWORK_PORT", "5555"))
        stale_after_s = float(os.environ.get("MICROBAN_NETWORK_STALE_S", "0.3"))
        allowed_remote = os.environ.get("MICROBAN_NETWORK_ALLOWED_IP") or None
        allowed_remote_file = os.environ.get("MICROBAN_NETWORK_ALLOWED_IP_FILE") or None
        return NetworkInputSource(
            port=port,
            stale_after_s=stale_after_s,
            allowed_remote=allowed_remote,
            allowed_remote_file=allowed_remote_file,
        )

    if requested not in ("auto", "keyboard", "gamepad"):
        raise ValueError(
            f"Unknown MICROBAN_INPUT={requested!r}; use auto, keyboard, gamepad, or network."
        )

    if requested in ("auto", "gamepad"):
        from input.gamepad_input import GamepadInputSource, find_gamepad_path

        if find_gamepad_path() is not None:
            return GamepadInputSource(button_moves=GAMEPAD_BUTTON_MOVES)
        if requested == "gamepad":
            raise RuntimeError("MICROBAN_INPUT=gamepad but no gamepad was found.")
        print("No gamepad detected; using keyboard input.")

    return KeyboardInputSource(move_keys=MOVE_KEYS)


def ramp_to_neutral(controller: RobotController, duration_s: float = 2.0) -> None:
    """Ramp all motors smoothly to neutral position before starting the control loop."""
    motor_ids = list(MOTOR_TO_ID.values())
    initial_positions = np.array(controller.sync_read_present_position(motor_ids))
    target_neutral = np.array([NEUTRAL_POSE[name] for name in MOTOR_TO_ID])

    print("Ramping all motors to neutral position...")
    start_time = time.perf_counter()
    elapsed_time = 0.0

    while elapsed_time < duration_s:
        elapsed_time = time.perf_counter() - start_time
        progress = min(elapsed_time / duration_s, 1.0)
        target_positions = initial_positions + progress * (target_neutral - initial_positions)
        controller.sync_write_goal_position(motor_ids, target_positions.flatten().tolist())

    controller.sync_write_goal_position(motor_ids, target_neutral.tolist())
    print("All motors reached neutral position.")


def main() -> None:
    if _another_session_running():
        print("A control loop is already running (see the PID file); aborting.")
        return

    PID_FILE.write_text(f"{os.getpid()}\n", encoding="ascii")

    controller: RobotController | None = None
    scheduler: Scheduler | None = None
    try:
        input_source = build_input_source()
        controls_motor_power = bool(
            getattr(input_source, "controls_motor_power", False)
        )
        force_start_off_value = os.environ.get(
            "MICROBAN_START_TORQUE_OFF", "0"
        ).strip().lower()
        if force_start_off_value not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                "MICROBAN_START_TORQUE_OFF must be 0/1, false/true, or no/yes"
            )
        force_start_off = force_start_off_value in {"1", "true", "yes"}
        if force_start_off and not controls_motor_power:
            raise ValueError(
                "MICROBAN_START_TORQUE_OFF requires an input source with the "
                "B/A/R3 hardware-power gate (use MICROBAN_INPUT=network or gamepad)"
            )
        serial_hold_value = os.environ.get(
            "MICROBAN_SERIAL_HOLD_LAST_ON_ERROR", "0"
        ).strip().lower()
        if serial_hold_value not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                "MICROBAN_SERIAL_HOLD_LAST_ON_ERROR must be 0/1, "
                "false/true, or no/yes"
            )
        serial_hold_on_error = serial_hold_value in {"1", "true", "yes"}

        controller = RobotController()
        motor_ids = list(MOTOR_TO_ID.values())
        start_limp = controls_motor_power or force_start_off
        if start_limp:
            # This is deliberately the first motor-bus write after opening the
            # controller; a prior process may have exited without clearing the
            # servos' persistent torque-enable registers.
            controller.sync_write_torque_enable(
                motor_ids, [False] * len(motor_ids)
            )
        controller.sync_write_status_return_level(motor_ids, [1] * len(motor_ids))
        controller.sync_write_kp(motor_ids, [KP_DEFAULT] * len(motor_ids))

        if start_limp:
            print(
                "Hardware-gated startup: all-joint torque OFF; press A to enable "
                "torque and return slowly to neutral."
            )
        else:
            controller.sync_write_torque_enable(
                motor_ids, [True] * len(motor_ids)
            )
            ramp_to_neutral(controller)

        scheduler = Scheduler(
            frequency_hz=50.0,
            controller=controller,
            input_source=input_source,
            hardware_power_control=controls_motor_power,
            serial_hold_on_error=serial_hold_on_error,
            moves={
                "head": RotateHeadMove(),
                "squat": SquatMove(),
                "walk": PolicySelectableWalkMove(controller=controller),
                # Ordered after walk so the right-trigger direct IK path owns
                # only the six arm joints in both standing and walking modes.
                "pico_arms": PicoArmTrackingMove(controller=controller),
                "hmd_head": HmdHeadTrackingMove(),
                "getup": GetupMove(controller=controller),
            },
        )

        for move in scheduler.registered_moves.values():
            move.preload()

        # Flush stale UART bytes accumulated during preload
        for _ in range(10):
            try:
                controller.sync_read_present_position(motor_ids)
            except RuntimeError:
                pass

        scheduler.run()

    finally:
        # Scheduler.run() owns normal cleanup. Cover initialization/preload failures
        # that happen after torque was enabled but before its finally block begins.
        if controller is not None and (scheduler is None or not scheduler._cleanup_done):
            try:
                motor_ids = list(MOTOR_TO_ID.values())
                controller.sync_write_torque_enable(motor_ids, [False] * len(motor_ids))
            finally:
                shutdown = getattr(controller, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        if PID_FILE.exists():
            PID_FILE.unlink()


if __name__ == "__main__":
    main()
