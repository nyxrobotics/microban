#!/usr/bin/env python3
"""Best-effort standalone all-joint torque-off for service boundaries.

This utility intentionally has no policy, input, IMU, or goal-position path.  A
systemd service uses it before opening the normal runtime and again after that
runtime exits, including after an initialization failure.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from rustypot import Xl330PyController

from constants import MOTOR_TO_ID

PID_FILE = Path("/tmp/microban_scheduler.pid")
TORQUE_OFF_DEADLINE_S = 1.0
SERIAL_TIMEOUT_S = 0.005


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


def _torque_enable_value(raw: object) -> float:
    """Convert rustypot's scalar or single-element response to a register value."""

    if isinstance(raw, (list, tuple)):
        if len(raw) != 1:
            raise ValueError("torque-enable response must contain one value")
        raw = raw[0]
    if hasattr(raw, "item"):
        raw = raw.item()
    return float(raw)


def main() -> None:
    live_pid = _live_scheduler_pid()
    if live_pid is not None:
        raise SystemExit(
            f"Refusing to touch the motor bus: scheduler PID {live_pid} is alive."
        )
    motor_ids = list(MOTOR_TO_ID.values())
    controller = Xl330PyController(
        serial_port="/dev/ttyAMA0", baudrate=1_000_000, timeout=SERIAL_TIMEOUT_S
    )
    deadline = time.monotonic() + TORQUE_OFF_DEADLINE_S
    unconfirmed = set(motor_ids)
    known_non_off: set[int] = set()
    last_write_error: str | None = None

    while unconfirmed and time.monotonic() < deadline:
        pending = [motor_id for motor_id in motor_ids if motor_id in unconfirmed]
        # A sync-write is sent to the protocol's broadcast ID, with one entry
        # for each servo that has not yet confirmed its torque is off.
        try:
            controller.sync_write_torque_enable(pending, [False] * len(pending))
        except (RuntimeError, OSError, ValueError) as exc:
            # The packet may still have reached some servos; read back each
            # register before deciding which IDs need another broadcast.
            last_write_error = str(exc)

        for motor_id in pending:
            if time.monotonic() >= deadline:
                break
            try:
                enabled = _torque_enable_value(controller.read_torque_enable(motor_id))
            except (RuntimeError, OSError, TypeError, ValueError):
                continue
            if enabled == 0.0:
                unconfirmed.discard(motor_id)
                known_non_off.discard(motor_id)
            else:
                known_non_off.add(motor_id)
        if unconfirmed:
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))

    if unconfirmed:
        print(
            f"Torque OFF unconfirmed after {TORQUE_OFF_DEADLINE_S:.1f} s; "
            f"servo IDs: {sorted(unconfirmed)}",
            flush=True,
        )
        if last_write_error is not None:
            print(f"Last torque OFF broadcast error: {last_write_error}", flush=True)
        if known_non_off:
            raise SystemExit(
                "Torque OFF failed: servo IDs still reported non-OFF: "
                f"{sorted(known_non_off)}"
            )
    else:
        print(f"All-joint torque OFF confirmed ({len(motor_ids)} motors).", flush=True)


if __name__ == "__main__":
    main()
