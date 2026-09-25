#!/usr/bin/env python3
"""Best-effort standalone all-joint torque-off for service boundaries.

This utility intentionally has no policy, input, IMU, or goal-position path.  A
systemd service uses it before opening the normal runtime and again after that
runtime exits, including after an initialization failure.
"""

from __future__ import annotations

import os
from pathlib import Path

from rustypot import Xl330PyController

from constants import MOTOR_TO_ID

PID_FILE = Path("/tmp/microban_scheduler.pid")


def _live_scheduler_pid() -> int | None:
    """Return the current bus owner's PID without disturbing that process."""

    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
        if pid <= 1:
            return None
        os.kill(pid, 0)
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    except PermissionError:
        return pid
    return pid


def main() -> None:
    live_pid = _live_scheduler_pid()
    if live_pid is not None:
        raise SystemExit(
            f"Refusing to touch the motor bus: scheduler PID {live_pid} is alive."
        )
    motor_ids = list(MOTOR_TO_ID.values())
    controller = Xl330PyController(
        serial_port="/dev/ttyAMA0", baudrate=1_000_000, timeout=0.1
    )
    controller.sync_write_torque_enable(motor_ids, [False] * len(motor_ids))
    print(f"All-joint torque OFF ({len(motor_ids)} motors).", flush=True)


if __name__ == "__main__":
    main()
