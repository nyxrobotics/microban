# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time
import math
from collections import deque
from pathlib import Path
from typing import Optional


from constants import (
    MOTOR_TO_ID,
    KP_HARDWARE_NEUTRAL,
    NEUTRAL_POSE,
    BAM_MAX_CURRENT,
    OVERCURRENT_CUTOFF_A,
    OVERCURRENT_CUTOFF_A_GETUP,
    OVERCURRENT_DEBOUNCE_TICKS,
    OVERCURRENT_PROXY_DELAY_TICKS,
    PROXY_KT,
    PROXY_R,
    PROXY_VIN,
    PROXY_ERROR_GAIN,
    PROXY_MAX_PWM,
    PROXY_KP,
)
from controller import ControllerProtocol
from home_pose import tilt_from_home_rad
from imu_reader import imu_quat_to_body
from observer import Observer, Observation, RobotState
from input.input_source import InputSource, UserInput, scale_velocity
from moves.getup import (
    _RECOVERY_MAX_DT_S,
    _RECOVERY_MIN_DT_S,
    _RECOVERY_SLEW_RATE_RAD_S,
)
from moves.move import MotorCommand, Move, MoveState
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from moves.walk import WalkMove


GETUP_AUTO_TIMEOUT_S = 20.0  # Match the get-up training episode length.
# Hand-back from the get-up actor to walk requires a settled stance: trunk
# tilt from the HOME attitude (the angle between the measured projected gravity
# and constants.HOME_PROJECTED_GRAVITY, i.e. HOME's 10 deg forward lean is
# zero tilt) below this angle for this many consecutive ticks (0.2 s at 50 Hz).
# The stand debounce alone let the walk take over while still leaning up to
# ~25 deg. Measured from vertical, the 10 deg HOME lean would leave only 2 deg
# of margin and a robot standing at HOME would rarely be handed back.
GETUP_HANDBACK_SETTLE_TILT_DEG = 12.0
GETUP_HANDBACK_SETTLE_TICKS = 10
# Stand detection (fallen -> standing debounce) uses the same HOME-relative
# tilt: standing means within acos(0.9) = 25.84 deg of the HOME attitude, the
# radius the vertical rule (projected_gravity z < -0.9) had at the upright
# HOME. Measured from vertical it would be a cone around the wrong attitude,
# leaving 15.8 deg of forward sway (the direction HOME leans) but 35.8 deg of
# backward lean from where the get-up actor actually stands.
GETUP_STAND_TILT_DEG = math.degrees(math.acos(0.9))
# Standing balance before the stance first settles is still the get-up
# transient: the get-up actor is still pulling the trunk upright (13-16 A in
# the current proxy in the runtime sim, against ~2 A once settled). It keeps
# the get-up overcurrent limit for at most this many ticks after the stand
# (1.5 s at 50 Hz), followed by the usual bounded tail; a stance that never
# settles then runs under the normal limit like any standing balance.
GETUP_SETTLE_CUTOFF_MAX_TICKS = 75


def _trunk_roll_pitch(body_quat: list[float]) -> tuple[float, float] | None:
    """Same formula as walk.py/hmd_head.py's own private helpers (kept as a
    third small copy rather than a shared import, matching that existing
    duplication), used here only to report trunk tilt alongside head/neck
    telemetry (see NetworkInputSource.set_head_telemetry)."""
    if len(body_quat) != 4 or not all(math.isfinite(value) for value in body_quat):
        return None
    norm = math.sqrt(sum(value * value for value in body_quat))
    if norm < 1e-6:
        return None
    w, x, y, z = (value / norm for value in body_quat)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


class Scheduler:
    def __init__(
        self,
        frequency_hz: float = 50.0,
        controller: ControllerProtocol = None,
        stop_flag_path: str = "/tmp/microban_scheduler.stop",
        input_source: Optional[InputSource] = None,
        moves: Optional[dict[str, Move]] = None,
        imu_max_age_s: float = 0.1,
        # On hardware, warn after this interval while holding the last goals.
        # Other input modes retain the historical terminal safety timeout.
        imu_shutdown_after_s: float = 0.75,
        hardware_power_control: bool | None = None,
        serial_hold_on_error: bool = False,
    ):
        if not math.isfinite(imu_max_age_s) or imu_max_age_s <= 0.0:
            raise ValueError("imu_max_age_s must be finite and positive")
        if not math.isfinite(imu_shutdown_after_s) or imu_shutdown_after_s <= 0.0:
            raise ValueError("imu_shutdown_after_s must be finite and positive")
        self.dt = 1.0 / frequency_hz
        self.controller = controller
        self.stop_flag_path = Path(stop_flag_path)
        self._cleanup_done = False
        self.input_source = input_source
        self._hardware_power_control = (
            bool(getattr(input_source, "controls_motor_power", False))
            if hardware_power_control is None
            else bool(hardware_power_control)
        )
        self._serial_hold_on_error = serial_hold_on_error
        self.observer = Observer(self.controller)

        # All moves are registered here. They only run when activated via user_input.active_moves.
        self.registered_moves: dict[str, Move] = moves if moves is not None else {
            "head": RotateHeadMove(),
            "squat": SquatMove(),
            "walk": WalkMove(),
        }

        self.loop_start_time = time.perf_counter()
        self._serial_errors = 0
        self._last_serial_warn_s = 0.0
        self._serial_hold_since_s: float | None = None
        self._serial_hold_extended = False
        self._last_good_robot_state: RobotState | None = None
        self._last_sent_targets: dict[str, float] | None = None
        self._serial_write_errors = 0
        self._serial_write_hold_pending = False
        self._serial_write_hold_since_s: float | None = None
        self._network_hold_active = False
        self._last_goal_write_ms = 0.0
        self._timing_window_start = time.perf_counter()
        self._timing_ticks = 0
        self._timing_overruns = 0
        self._timing_max_overrun_ms = 0.0
        self._timing_position_ms = 0.0
        self._timing_velocity_ms = 0.0
        self._timing_write_ms = 0.0
        self._last_imu_print_s: float = 0.0
        self._last_imu_stale_warn_s: float = 0.0
        self._imu_max_age_s = imu_max_age_s
        self._imu_shutdown_after_s = imu_shutdown_after_s
        self._imu_unsafe_since_s: float | None = None
        self._imu_extended_hold_warned = False
        self._safety_hold_active = False
        self._overcurrent_ticks = 0

        # Global real-hardware B/A/R3 state.  This lives above individual
        # moves because limp/neutral must own all 21 joints, including the
        # neck and direct-arm overlay. None forces a physical torque write on
        # the first scheduler tick instead of assuming startup state.
        self._hardware_torque_enabled: bool | None = None
        self._hardware_policy_enabled = False
        self._hardware_policy_eligible = False
        self._hardware_neutral_targets: dict[str, float] | None = None
        self._hardware_neutral_last_time_s: float | None = None

        # Fall / getup auto-switch (only relevant when a "getup" move is registered).
        # Fall: projected_gravity[2] > -0.5, a trunk tilt above 60 deg from vertical
        # (-1 is upright, 0 on its side). It matches WalkMove's and PicoHybridMove's own
        # safety-stop criterion and stays measured from vertical: a fall is a physical
        # attitude, HOME's 10 deg forward lean still leaves 50 deg before it trips, and a
        # HOME-relative cone would let a backward fall go to 70 deg before triggering.
        # Stand: tilt from the HOME attitude below GETUP_STAND_TILT_DEG.
        self._fall_threshold = -0.5
        self._stand_tilt_rad = math.radians(GETUP_STAND_TILT_DEG)
        self._fall_debounce_ticks = 15  # ~0.3 s at 50 Hz
        self._stand_debounce_ticks = 20  # ~0.4 s at 50 Hz
        self._fallen_tick_count = 0
        self._standing_tick_count = 0
        # Consecutive IMU-valid ticks with tilt below the hand-back settle
        # angle; see GETUP_HANDBACK_SETTLE_*.
        self._settle_tilt_rad = math.radians(GETUP_HANDBACK_SETTLE_TILT_DEG)
        self._settled_tick_count = 0
        # One log line per deferred hand-back (not one per tick).
        self._handback_defer_logged = False
        # Standing-balance ticks since the stand while the stance has not yet
        # settled; None once it settled (or when not balancing).
        self._getup_settle_wait_ticks: int | None = None
        self._getup_active_override = False
        # Post-get-up standing balance. Once stood up, GetupMove stays on as the
        # standing balancer (its actor is trained to stand still and recover
        # from pushes) until a "walk" move that can itself balance is
        # requested (see _walk_can_take_over) and the stance has settled
        # (GETUP_HANDBACK_SETTLE_*). While True, the override is still
        # on but the get-up time limit is not counting.
        self._getup_balancing = False
        # Operator's requested active_moves when balancing began; on sources
        # without a hardware power gate, a change to it releases balancing.
        self._getup_balance_request: frozenset[str] | None = None
        # Keep detecting falls even if the installed get-up actor lacks the
        # current training/deployment contract. Such a fall returns toward
        # neutral instead of running the incompatible actor.
        self._getup_auto_trigger_enabled = True
        self._getup_auto_started_s: float | None = None
        self._getup_auto_failed = False
        self._getup_failed_hold_targets: dict[str, float] | None = None
        self._getup_failed_hold_last_time_s: float | None = None
        self._getup_failed_gain_restored: set[str] = set()
        self._getup_failed_stale_names: set[str] = set()
        self._fall_pending_hold_targets: dict[str, float] | None = None
        # History of sent target_angles, to align the current proxy with the delayed feedback:
        # the oldest entry is the command issued OVERCURRENT_PROXY_DELAY_TICKS ticks ago.
        self._cmd_history: deque[dict[str, float]] = deque(maxlen=OVERCURRENT_PROXY_DELAY_TICKS + 1)
        # Checks left at the get-up cutoff after it last applied: a command
        # written under it stays in _cmd_history (and, on the robot, in the
        # servos' delayed response) for up to maxlen ticks. Bounded by maxlen.
        self._getup_cutoff_tail_ticks = 0
        self._getup_balance_tail_ticks = 0

    def run(self):
        print(f"Starting control loop at {1 / self.dt:.1f} Hz", end="\r\n", flush=True)
        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

        if self.input_source:
            self.input_source.start()

        try:
            while True:
                if self.stop_flag_path.exists():
                    print("Stop requested through stop flag", end="\r\n", flush=True)
                    break

                start_time = time.perf_counter()
                self._last_goal_write_ms = 0.0

                # Read robot observations and user input
                try:
                    robot_state = self.observer.read_state(self.dt)
                except RuntimeError as e:
                    self._serial_errors += 1
                    if self._serial_hold_on_error:
                        if self._serial_hold_since_s is None:
                            self._serial_hold_since_s = start_time
                        held_for_s = start_time - self._serial_hold_since_s
                        if start_time - self._last_serial_warn_s >= 1.0:
                            print(
                                f"Warning: serial read error ({self._serial_errors} "
                                f"missed ticks); holding the previous motor goals: {e}",
                                end="\r\n",
                                flush=True,
                            )
                            self._last_serial_warn_s = start_time
                        # Keep the operator's B button and the network deadman
                        # effective even while all feedback reads are failing.
                        if self.input_source is not None:
                            if held_for_s >= 0.3:
                                self.input_source.set_motion_inhibited(True)
                                self._serial_hold_extended = True
                            snapshot = self.input_source.read()
                            if (
                                self._hardware_power_control
                                and snapshot.torque_enabled is False
                            ):
                                try:
                                    self._disable_hardware_torque()
                                except RuntimeError as off_error:
                                    # Leave the gate armed to retry the OFF write
                                    # on the next tick; do not claim it succeeded.
                                    print(
                                        f"Warning: torque-off retry needed: {off_error}",
                                        end="\r\n",
                                        flush=True,
                                    )
                        if self._last_sent_targets is not None:
                            self._cmd_history.append(dict(self._last_sent_targets))
                        self._pace_tick(start_time)
                        continue
                    if self._serial_errors >= 3:
                        print(f"Serial communication error: {e}", end="\r\n", flush=True)
                        break
                    print(f"Warning: serial read error (attempt {self._serial_errors}/3): {e}", end="\r\n", flush=True)
                    continue
                if self._serial_hold_on_error and self._serial_errors:
                    if self._serial_hold_extended:
                        self._safety_hold_active = True
                    self._serial_hold_since_s = None
                    self._serial_hold_extended = False
                self._serial_errors = 0

                robot_state.time_s = start_time - self.loop_start_time
                self._last_good_robot_state = robot_state
                user_input = self.input_source.read() if self.input_source else UserInput()
                user_input.velocity = scale_velocity(user_input.velocity)
                if self.input_source:
                    set_head_telemetry = getattr(
                        self.input_source, "set_head_telemetry", None
                    )
                    if callable(set_head_telemetry):
                        trunk_angles = _trunk_roll_pitch(robot_state.body_quat)
                        if trunk_angles is not None:
                            positions = robot_state.motor_positions
                            set_head_telemetry(
                                head=float(positions.get("head", 0.0)),
                                neck_roll=float(positions.get("neck_roll", 0.0)),
                                neck_pitch=float(positions.get("neck_pitch", 0.0)),
                                trunk_roll=trunk_angles[0],
                                trunk_pitch=trunk_angles[1],
                            )
                if user_input.hold_last_targets:
                    if not self._network_hold_active:
                        print(
                            "Network input unavailable; holding last motor goals and torque",
                            end="\r\n",
                            flush=True,
                        )
                        self._network_hold_active = True
                    if self._last_sent_targets is not None:
                        self._cmd_history.append(dict(self._last_sent_targets))
                    self._pace_tick(start_time)
                    continue
                if self._network_hold_active:
                    print("Network input resumed", end="\r\n", flush=True)
                    self._network_hold_active = False
                obs = Observation(robot_state=robot_state, user_input=user_input)

                imu_status = None
                imu_status_getter = getattr(self.controller, "get_imu_status", None)
                if callable(imu_status_getter):
                    try:
                        imu_status = imu_status_getter()
                    except Exception as exc:
                        imu_status = {"valid": False, "age_s": math.inf, "error": str(exc)}
                imu_safe, imu_reason = self._imu_is_safe(robot_state, imu_status)
                if not imu_safe:
                    if self._imu_unsafe_since_s is None:
                        self._imu_unsafe_since_s = start_time
                    if (start_time - self._last_imu_stale_warn_s) >= 1.0:
                        print(f"Warning: unsafe IMU ({imu_reason}); holding measured pose", end="\r\n", flush=True)
                        self._last_imu_stale_warn_s = start_time
                else:
                    self._imu_unsafe_since_s = None
                    self._imu_extended_hold_warned = False

                try:
                    hardware_mode, hardware_command = self._apply_hardware_gate(
                        obs,
                        allow_initial_enable=imu_safe,
                    )
                except RuntimeError as e:
                    if not self._serial_hold_on_error:
                        raise
                    if self._serial_write_hold_since_s is None:
                        self._serial_write_hold_since_s = start_time
                    if start_time - self._last_serial_warn_s >= 1.0:
                        print(
                            f"Warning: serial hardware-gate write error; holding "
                            f"previous motor goals and retrying: {e}",
                            end="\r\n",
                            flush=True,
                        )
                        self._last_serial_warn_s = start_time
                    if (
                        self.input_source is not None
                        and start_time - self._serial_write_hold_since_s >= 0.3
                    ):
                        self.input_source.set_motion_inhibited(True)
                        self._serial_hold_extended = True
                    if self._last_sent_targets is not None:
                        self._cmd_history.append(dict(self._last_sent_targets))
                    self._pace_tick(start_time)
                    continue
                if self._serial_write_hold_since_s is not None and not self._serial_write_hold_pending:
                    if self._serial_hold_extended:
                        self._safety_hold_active = True
                    self._serial_write_hold_since_s = None
                    self._serial_hold_extended = False

                # R3-off hands control to the A neutral gate and explicitly
                # clears a latched get-up fault. Otherwise keep the fault
                # latched even after the IMU reports upright, so policy cannot
                # resume while the neutral return is still in progress.
                if hardware_mode == "neutral" and self._getup_auto_failed:
                    self._getup_auto_failed = False
                    self._getup_auto_started_s = None
                    self._getup_failed_hold_targets = None
                    self._getup_failed_hold_last_time_s = None
                    self._getup_failed_gain_restored.clear()
                    self._getup_failed_stale_names.clear()

                # Fall / getup auto-switch: forces "getup" on (and "walk" off) after a
                # sustained fall, and hands back to the user's own "walk" toggle once
                # stood up (and stable) again -- but only to a walk move that can
                # balance, and only once the tilt has settled below
                # GETUP_HANDBACK_SETTLE_TILT_DEG; until then get-up stays on as
                # the standing balancer, and
                # leaves on R3-off/B/actor fault (or, without that gate, any
                # move-toggle change). No-op if "getup" isn't
                # registered.
                fall_pending = False
                if "getup" in self.registered_moves and self._getup_auto_trigger_enabled:
                    getup_move = self.registered_moves["getup"]
                    model_ready = getattr(getup_move, "model_ready", True)
                    self._update_settle_count(
                        obs.robot_state.projected_gravity if imu_safe else None
                    )
                    # Evaluated on the operator's own request, before get-up
                    # rewrites active_moves below; only relevant while get-up
                    # owns the robot.
                    walk_ready = (
                        self._getup_active_override
                        and self._walk_can_take_over(obs.user_input)
                    )
                    # Every hand-back to walk (from standing balance, or
                    # directly at the stand debounce) also waits for a settled
                    # stance; until then the get-up actor keeps balancing.
                    walk_takes_over = (
                        walk_ready
                        and self._settled_tick_count >= GETUP_HANDBACK_SETTLE_TICKS
                    )
                    # The operator's own request, before the rewrite below.
                    requested_moves = frozenset(obs.user_input.active_moves)
                    if self._getup_balancing:
                        # An unsafe IMU is not a release reason: the safety hold
                        # below freezes the goals, and balancing resumes through
                        # GetupMove.on_safety_resume once the IMU is valid again
                        # (exactly as a get-up attempt rides through it).
                        release_reason = (
                            "policy output withheld" if hardware_mode != "policy"
                            else "get-up actor unavailable"
                            if self._getup_auto_failed or not model_ready
                            else "balancing walk requested" if walk_takes_over
                            # Sources without the B/A/R3 gate (keyboard, MuJoCo
                            # viewer) have no other way out: any change to the
                            # operator's move toggles hands control back.
                            else "operator changed active moves"
                            if (
                                not self._hardware_power_control
                                and self._getup_balance_request is not None
                                and requested_moves != self._getup_balance_request
                            )
                            else None
                        )
                        if release_reason is not None:
                            self._release_getup_balance(release_reason)
                    if imu_safe:
                        fall_pending = self._update_getup_override(
                            obs.robot_state.projected_gravity,
                            balance_when_standing=(
                                hardware_mode == "policy"
                                and model_ready
                                and not self._getup_auto_failed
                                and not walk_takes_over
                            ),
                        )
                    if self._getup_balancing and walk_ready and not walk_takes_over:
                        if not self._handback_defer_logged:
                            self._handback_defer_logged = True
                            print(
                                "Get-up hand-back to walk deferred until the stance "
                                "settles (tilt from HOME < "
                                f"{GETUP_HANDBACK_SETTLE_TILT_DEG:g} deg "
                                f"for {GETUP_HANDBACK_SETTLE_TICKS} ticks; now "
                                f"{self._tilt_deg(obs.robot_state.projected_gravity)})",
                                end="\r\n", flush=True,
                            )
                    else:
                        self._handback_defer_logged = False
                    if not self._getup_balancing:
                        self._getup_settle_wait_ticks = None
                    elif self._getup_settle_wait_ticks is not None:
                        if self._settled_tick_count >= GETUP_HANDBACK_SETTLE_TICKS:
                            self._getup_settle_wait_ticks = None
                        else:
                            self._getup_settle_wait_ticks += 1
                    if not self._getup_balancing:
                        self._getup_balance_request = None
                    elif self._getup_balance_request is None:
                        self._getup_balance_request = requested_moves
                    if self._getup_active_override:
                        if self._getup_balancing:
                            # The balancer is the sole owner. Motion input is no
                            # longer inhibited (so a walk request is visible and a
                            # released trigger can re-arm the deadman), but none of
                            # it reaches a move until walk takes over at zero speed.
                            obs.user_input.active_moves = {"getup"}
                            obs.user_input.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
                        else:
                            obs.user_input.active_moves = (obs.user_input.active_moves | {"getup"}) - {"walk"}
                        # Tell the walk owner (PolicySelectableWalkMove) that
                        # its next start follows get-up.
                        walk_move = self.registered_moves.get("walk")
                        seed_from_getup = getattr(
                            walk_move, "seed_next_start_from_getup", None
                        )
                        if callable(seed_from_getup):
                            seed_from_getup()
                        if not model_ready and hardware_mode == "policy":
                            self._stop_auto_getup("get-up model contract unavailable; returning to neutral")
                        can_attempt = imu_safe and hardware_mode == "policy" and model_ready
                        # The time limit bounds one attempt from the fall to
                        # standing; it never runs while standing-balancing.
                        timing = can_attempt and not self._getup_balancing
                        if timing and self._getup_auto_started_s is None:
                            self._getup_auto_started_s = start_time
                            print("Automatic get-up attempt started", end="\r\n", flush=True)
                        if timing and self._getup_auto_started_s is not None:
                            elapsed_s = start_time - self._getup_auto_started_s
                            if elapsed_s >= GETUP_AUTO_TIMEOUT_S:
                                self._stop_auto_getup(
                                    f"{GETUP_AUTO_TIMEOUT_S:g}-second time limit"
                                )
                        obs.user_input.getup_armed = can_attempt and not self._getup_auto_failed
                    else:
                        self._getup_auto_started_s = None
                    # Keep an actor fault latched until R3 is switched off or
                    # B turns torque off, including after the robot is upright.
                    fall_pending = fall_pending or self._getup_auto_failed
                    if not fall_pending:
                        self._fall_pending_hold_targets = None

                # Keep the network deadman disarmed for the complete fault/fall/get-up
                # interval. Removing the inhibit does not arm it: a later released
                # trigger snapshot is required before walking can resume.
                # Standing-balance is the exception: the deadman is live again
                # (see the active_moves rewrite above), exactly as after a
                # hand-back, so walking still needs a fresh trigger release.
                motion_inhibited = (
                    not imu_safe
                    or fall_pending
                    or (self._getup_active_override and not self._getup_balancing)
                    or hardware_mode != "policy"
                    or (self._serial_write_hold_pending and self._serial_hold_extended)
                )
                if self.input_source:
                    self.input_source.set_motion_inhibited(motion_inhibited)

                hold_for_safety = (
                    hardware_mode != "limp" and (not imu_safe or fall_pending)
                )
                if (
                    not imu_safe
                    and hardware_mode != "limp"
                    and self._imu_unsafe_since_s is not None
                    and start_time - self._imu_unsafe_since_s >= self._imu_shutdown_after_s
                ):
                    if self._hardware_power_control:
                        if not self._imu_extended_hold_warned:
                            print(
                                "IMU remains unavailable; holding previous motor goals and torque",
                                end="\r\n",
                                flush=True,
                            )
                            self._imu_extended_hold_warned = True
                    else:
                        print(
                            f"IMU remained unsafe for {self._imu_shutdown_after_s:.2f} s "
                            "— stopping control loop and disabling torque",
                            end="\r\n",
                            flush=True,
                        )
                        break

                # Update move states and dispatch one call per move per tick
                if hardware_mode == "limp":
                    # Explicit B or process startup: torque is physically off
                    # and no goal position is written. Network gaps are held
                    # earlier without changing the hardware gate.
                    command = MotorCommand(target_angles={})
                    self._safety_hold_active = False
                    self._serial_write_hold_pending = False
                    self._serial_write_hold_since_s = None
                elif hold_for_safety:
                    # A transient IMU outage must not turn the joints into a
                    # compliant follower or end the hardware control process.
                    # Freeze the exact last goal; policy output stays inhibited
                    # until the IMU is valid again.
                    if not imu_safe and self._hardware_power_control and self._last_sent_targets:
                        command = MotorCommand(target_angles=dict(self._last_sent_targets))
                        if hardware_mode == "neutral":
                            self._hardware_neutral_targets = dict(self._last_sent_targets)
                            self._hardware_neutral_last_time_s = robot_state.time_s
                    elif fall_pending and self._getup_auto_failed:
                        # A failed get-up returns toward neutral at a bounded
                        # speed. Following measured position every tick would
                        # make torque-on joints compliant while still fallen.
                        if hardware_mode == "neutral":
                            # The A/R3-off gate owns neutral return in this mode.
                            # A later R3-on must seed from the current pose.
                            self._getup_failed_hold_targets = None
                            self._getup_failed_hold_last_time_s = None
                            self._getup_failed_gain_restored.clear()
                            self._getup_failed_stale_names.clear()
                            command = hardware_command
                        else:
                            command = self._failed_getup_hold_command(robot_state)
                    elif fall_pending:
                        if hardware_mode == "neutral":
                            self._fall_pending_hold_targets = None
                            command = hardware_command
                        else:
                            command = self._fall_pending_hold_command(robot_state)
                    else:
                        command = self._measured_hold_command(robot_state)
                    if command is None:
                        print(
                            "Invalid motor feedback during safety hold — stopping control "
                            "loop and disabling torque",
                            end="\r\n",
                            flush=True,
                        )
                        break
                    self._safety_hold_active = True
                elif self._serial_write_hold_pending:
                    # A failed sync write has an uncertain bus outcome. Reissue
                    # the last successful write and keep policy state frozen.
                    command = MotorCommand(target_angles=dict(self._last_sent_targets or {}))
                elif hardware_mode == "neutral":
                    # A, or R3 toggled back off: the global gate already built
                    # a bounded all-joint neutral-return command.
                    if hardware_command is None:
                        print(
                            "Invalid motor feedback during neutral return — "
                            "disabling torque",
                            end="\r\n",
                            flush=True,
                        )
                        self._disable_hardware_torque()
                        continue
                    command = hardware_command
                    self._safety_hold_active = False
                else:
                    if self._safety_hold_active:
                        for move in self.registered_moves.values():
                            move.on_safety_resume(obs)
                        self._safety_hold_active = False
                    command = MotorCommand()
                    for name, move in self.registered_moves.items():
                        in_active = name in obs.user_input.active_moves

                        if in_active and move.state == MoveState.INACTIVE:
                            move.state = MoveState.STARTING
                        elif not in_active and move.state in (MoveState.STARTING, MoveState.ACTIVE):
                            move.state = MoveState.STOPPING

                        if move.state == MoveState.STARTING:
                            move.on_start(obs, command)
                        elif move.state == MoveState.ACTIVE:
                            move.step(obs, command)
                        elif move.state == MoveState.STOPPING:
                            move.on_stop(obs, command)

                    getup_move = self.registered_moves.get("getup")
                    if (
                        getup_move is not None
                        and getup_move.state == MoveState.ACTIVE
                        and getattr(getup_move, "policy_faulted", False)
                    ):
                        self._stop_auto_getup("get-up actor output exceeded bounds")
                        held = self._failed_getup_hold_command(robot_state)
                        if held is not None:
                            command = held
                        self._safety_hold_active = True

                # Overcurrent safety: estimate the current from the command that was actually
                # active when the (delayed) position/velocity feedback was sampled — the oldest
                # buffered command (== DELAY_TICKS old once full). This reconstructs the real
                # (delayed) current instead of pairing a fresh target with stale feedback, which
                # would inflate the error term and false-trigger at gait start.
                if hardware_mode == "limp":
                    self._cmd_history.clear()
                    self._getup_cutoff_tail_ticks = 0
                    self._getup_balance_tail_ticks = 0
                    self._overcurrent_ticks = 0
                else:
                    aligned_targets = (
                        self._cmd_history[0]
                        if self._cmd_history
                        else command.target_angles
                    )
                    # The higher get-up limit covers the bounded get-up
                    # transient only; standing balance can last indefinitely
                    # and runs under the normal limit.
                    # The fall-debounce hold is part of the same fall
                    # transient: the delay-aligned proxy still pairs the
                    # walk actor's last (falling) goals with a snapped hold.
                    getup_cutoff = (
                        "getup" in obs.user_input.active_moves
                        and not self._getup_balancing
                    ) or (fall_pending and not self._getup_auto_failed)
                    # After the get-up (or fall hold) stops -- hand-back to
                    # walk, standing balance, or a stop -- its last goals,
                    # often saturated at the servo range, are still what the
                    # delayed feedback answers to: the proxy pairs them with
                    # _cmd_history[0] and the servos are still executing them.
                    # Keep the get-up limit until every command in the history
                    # was written after it stopped (maxlen ticks), then the
                    # normal limit applies again.
                    # The first maxlen standing-balance ticks right after an
                    # attempt still write the get-up actor's transient goals
                    # (the hand-back to walk always passes through one), so
                    # they re-arm the tail too: the get-up limit then lasts at
                    # most 2 * maxlen checks past the attempt's last tick.
                    # Standing balance that has not yet settled since the
                    # stand is still the get-up transient (bounded by
                    # GETUP_SETTLE_CUTOFF_MAX_TICKS) and counts as the attempt.
                    if (
                        self._getup_balancing
                        and self._getup_settle_wait_ticks is not None
                        and self._getup_settle_wait_ticks <= GETUP_SETTLE_CUTOFF_MAX_TICKS
                    ):
                        getup_cutoff = True
                    if getup_cutoff:
                        self._getup_cutoff_tail_ticks = self._cmd_history.maxlen
                        self._getup_balance_tail_ticks = self._cmd_history.maxlen
                    elif self._getup_balancing and self._getup_balance_tail_ticks > 0:
                        self._getup_balance_tail_ticks -= 1
                        self._getup_cutoff_tail_ticks = self._cmd_history.maxlen
                        getup_cutoff = True
                    elif self._getup_cutoff_tail_ticks > 0:
                        self._getup_cutoff_tail_ticks -= 1
                        getup_cutoff = True
                    cutoff = (
                        OVERCURRENT_CUTOFF_A_GETUP if getup_cutoff
                        else OVERCURRENT_CUTOFF_A
                    )
                    if self._check_overcurrent(robot_state, aligned_targets, cutoff):
                        break

                    # Send command to motors
                    try:
                        self._send_to_motors(
                            command, neutral_return=hardware_mode == "neutral"
                        )
                    except RuntimeError as e:
                        if not self._serial_hold_on_error:
                            raise
                        self._serial_write_errors += 1
                        self._serial_write_hold_pending = True
                        if hardware_mode == "neutral" and self._last_sent_targets is not None:
                            # The neutral slew must not advance past a failed
                            # goal write while a previous goal is held.
                            self._hardware_neutral_targets = dict(self._last_sent_targets)
                            self._hardware_neutral_last_time_s = robot_state.time_s
                        if self._serial_write_hold_since_s is None:
                            self._serial_write_hold_since_s = start_time
                        if start_time - self._serial_write_hold_since_s >= 0.3:
                            self._serial_hold_extended = True
                            if self.input_source is not None:
                                self.input_source.set_motion_inhibited(True)
                        if start_time - self._last_serial_warn_s >= 1.0:
                            print(
                                f"Warning: serial goal write error; holding the "
                                f"previous motor goals: {e}",
                                end="\r\n",
                                flush=True,
                            )
                            self._last_serial_warn_s = start_time
                        if self._last_sent_targets is not None:
                            self._cmd_history.append(dict(self._last_sent_targets))
                    else:
                        self._serial_write_errors = 0
                        if command.target_angles:
                            self._remember_goal_write(command.target_angles)
                        if self._last_sent_targets is not None:
                            self._cmd_history.append(dict(self._last_sent_targets))
                        if self._serial_write_hold_pending:
                            self._serial_write_hold_pending = False
                            self._serial_write_hold_since_s = None
                            if self._serial_hold_extended:
                                self._safety_hold_active = True
                            self._serial_hold_extended = False

                # IMU / gyro terminal display
                if obs.user_input.show_imu and (start_time - self._last_imu_print_s) >= 0.5:
                    acc = obs.robot_state.acc
                    gyro = obs.robot_state.gyro
                    quat = obs.robot_state.quat
                    print("--------------------------------------------", end="\r\n", flush=True)
                    print(f"Policy: {'ON' if hardware_mode == 'policy' else 'OFF'}", end="\r\n", flush=True)
                    if gyro:
                        gx, gy, gz = gyro
                        print(f"Gyro: gx={gx:+.3f}  gy={gy:+.3f}  gz={gz:+.3f} rad/s", end="\r\n", flush=True)
                    if acc:
                        ax, ay, az = acc
                        print(f"Acc:  ax={ax:+.3f}  ay={ay:+.3f}  az={az:+.3f} g", end="\r\n", flush=True)
                    if quat:
                        w, x, y, z = quat
                        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
                        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
                        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                        print(f"IMU:  roll={roll:+.1f}°  pitch={pitch:+.1f}°  yaw={yaw:+.1f}°", end="\r\n", flush=True)
                        bw, bx, by, bz = imu_quat_to_body((w, x, y, z))
                        b_roll = math.degrees(math.atan2(2 * (bw * bx + by * bz), 1 - 2 * (bx * bx + by * by)))
                        b_pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (bw * by - bz * bx)))))
                        b_yaw = math.degrees(math.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz)))
                        print(f"Body: roll={b_roll:+.1f}°  pitch={b_pitch:+.1f}°  yaw={b_yaw:+.1f}°", end="\r\n", flush=True)
                    self._last_imu_print_s = start_time

                self._pace_tick(start_time)

        except KeyboardInterrupt:
            print("Control loop interrupted by user", end="\r\n", flush=True)
        finally:
            self._cleanup()

    def _apply_hardware_gate(
        self,
        obs: Observation,
        *,
        allow_initial_enable: bool,
    ) -> tuple[str, MotorCommand | None]:
        """Apply the real-hardware B/A/R3 state above every move.

        Returns ``("limp", empty-command)``, ``("neutral", all-joint-command)``
        or ``("policy", None)``.  Enabling policy is deliberately a two-step
        operation: at least one torque-on/policy-off (A) snapshot must be seen
        before a policy-on (R3) snapshot is honored.  This prevents a bridge
        restart or a stale held button from energizing and driving in one tick.
        """
        if not self._hardware_power_control:
            return "policy", None

        requested_torque = obs.user_input.torque_enabled is True
        requested_policy = (
            requested_torque and obs.user_input.policy_enabled is True
        )

        if not requested_torque:
            self._disable_hardware_torque()
            self._reset_moves_for_hardware_gate()
            return "limp", MotorCommand(target_angles={})

        just_enabled = False
        if self._hardware_torque_enabled is not True:
            # Never turn torque on while the IMU/feedback safety precondition is
            # unknown.  Keep observing packets and retry when it becomes valid.
            if not allow_initial_enable:
                self._disable_hardware_torque()
                self._reset_moves_for_hardware_gate()
                return "limp", MotorCommand(target_angles={})

            measured = self._finite_joint_positions(obs.robot_state)
            if measured is None:
                self._disable_hardware_torque()
                self._reset_moves_for_hardware_gate()
                return "limp", MotorCommand(target_angles={})

            # For an unreadable joint, seed the neutral goal itself. The A
            # return is feedforward; it must not wait for a position packet.
            neutral_seed = dict(measured)
            for name in getattr(self.controller, "stale_motor_names", ()):
                if name in neutral_seed:
                    neutral_seed[name] = NEUTRAL_POSE[name]

            motor_ids = list(MOTOR_TO_ID.values())
            self.controller.sync_write_kp(
                motor_ids, [KP_HARDWARE_NEUTRAL] * len(motor_ids)
            )
            # Replace old goal registers before enabling torque. Responsive
            # joints start from their measured pose; unreadable joints receive
            # neutral directly, as requested for feedforward A return.
            write_neutral = getattr(
                self.controller,
                "sync_write_neutral_goal_position",
                self.controller.sync_write_goal_position,
            )
            write_neutral(motor_ids, [neutral_seed[name] for name in MOTOR_TO_ID])
            self._remember_goal_write(neutral_seed)
            # Keep the gate uncertain until the bounded ON transaction returns.
            # Unconfirmed IDs are retried in the background after the healthy
            # joints start their neutral return.
            self._hardware_torque_enabled = None
            self.controller.sync_write_torque_enable(
                motor_ids, [True] * len(motor_ids)
            )
            self._remember_goal_write(neutral_seed)
            self._hardware_torque_enabled = True
            self._hardware_policy_enabled = False
            self._hardware_neutral_targets = dict(self._last_sent_targets or neutral_seed)
            self._hardware_neutral_last_time_s = obs.robot_state.time_s
            self._cmd_history.clear()
            self._getup_cutoff_tail_ticks = 0
            self._getup_balance_tail_ticks = 0
            self._overcurrent_ticks = 0
            just_enabled = True
            print(
                "Hardware gate: torque ON requested; policy withheld, returning to neutral",
                end="\r\n",
                flush=True,
            )

        # A's false-policy snapshot is the explicit authorization which makes
        # a later R3 true-policy snapshot eligible.  A true-policy packet seen
        # directly from limp is ignored until that intermediate state arrives.
        if not requested_policy:
            self._hardware_policy_eligible = True
        effective_policy = requested_policy and self._hardware_policy_eligible

        if effective_policy:
            if not self._hardware_policy_enabled:
                print(
                    "Hardware gate: normal policy output enabled",
                    end="\r\n",
                    flush=True,
                )
            self._hardware_policy_enabled = True
            self._hardware_neutral_targets = None
            self._hardware_neutral_last_time_s = None
            return "policy", None

        if self._hardware_policy_enabled:
            measured = self._finite_joint_positions(obs.robot_state)
            if measured is None:
                self._disable_hardware_torque()
                self._reset_moves_for_hardware_gate()
                return "limp", MotorCommand(target_angles={})
            self._hardware_neutral_targets = measured
            self._hardware_neutral_last_time_s = obs.robot_state.time_s
            motor_ids = list(MOTOR_TO_ID.values())
            # Remove the last learned target before changing gains.  Without
            # this ordering, raising an RL joint to neutral holding gain could
            # briefly amplify its error against a stale policy goal.
            neutral_seed = dict(measured)
            for name in getattr(self.controller, "stale_motor_names", ()):
                previous_goal = (self._last_sent_targets or {}).get(name)
                if previous_goal is not None and math.isfinite(previous_goal):
                    neutral_seed[name] = previous_goal
            write_neutral = getattr(
                self.controller,
                "sync_write_neutral_goal_position",
                self.controller.sync_write_goal_position,
            )
            write_neutral(motor_ids, [neutral_seed[name] for name in MOTOR_TO_ID])
            self._remember_goal_write(neutral_seed)
            self._hardware_neutral_targets = dict(self._last_sent_targets or neutral_seed)
            self.controller.sync_write_kp(
                motor_ids, [KP_HARDWARE_NEUTRAL] * len(motor_ids)
            )
            print(
                "Hardware gate: policy withheld; returning to neutral",
                end="\r\n",
                flush=True,
            )
        elif not just_enabled and self._hardware_neutral_targets is None:
            measured = self._finite_joint_positions(obs.robot_state)
            if measured is None:
                self._disable_hardware_torque()
                self._reset_moves_for_hardware_gate()
                return "limp", MotorCommand(target_angles={})
            self._hardware_neutral_targets = measured
            self._hardware_neutral_last_time_s = obs.robot_state.time_s

        self._hardware_policy_enabled = False
        self._reset_moves_for_hardware_gate()
        command = (
            MotorCommand(target_angles=dict(self._last_sent_targets or {}))
            if self._serial_write_hold_pending
            else self._step_hardware_neutral(obs.robot_state)
        )
        if command is None:
            self._disable_hardware_torque()
            return "limp", MotorCommand(target_angles={})
        return "neutral", command

    @staticmethod
    def _finite_joint_positions(robot_state) -> dict[str, float] | None:
        measured: dict[str, float] = {}
        for name in MOTOR_TO_ID:
            try:
                value = float(robot_state.motor_positions[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(value):
                return None
            measured[name] = value
        return measured

    def _step_hardware_neutral(self, robot_state) -> MotorCommand | None:
        targets = self._hardware_neutral_targets
        if targets is None or set(targets) != set(MOTOR_TO_ID):
            return None

        try:
            now = float(robot_state.time_s)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(now):
            return None
        previous = self._hardware_neutral_last_time_s
        raw_dt = self.dt if previous is None else now - previous
        dt = max(
            _RECOVERY_MIN_DT_S,
            min(_RECOVERY_MAX_DT_S, raw_dt if math.isfinite(raw_dt) else self.dt),
        )
        self._hardware_neutral_last_time_s = now
        max_step = _RECOVERY_SLEW_RATE_RAD_S * dt

        updated: dict[str, float] = {}
        for name in MOTOR_TO_ID:
            current = targets[name]
            neutral = NEUTRAL_POSE[name]
            delta = max(-max_step, min(max_step, neutral - current))
            value = current + delta
            if not math.isfinite(value):
                return None
            targets[name] = value
            updated[name] = value
        return MotorCommand(target_angles=updated)

    def _reset_moves_for_hardware_gate(self) -> None:
        """Discard every learned/interpolated state while policy is withheld."""
        for move in self.registered_moves.values():
            move.state = MoveState.INACTIVE

    def _disable_hardware_torque(self) -> None:
        if self._hardware_torque_enabled is not False:
            motor_ids = list(MOTOR_TO_ID.values())
            self._hardware_torque_enabled = None
            self.controller.sync_write_torque_enable(
                motor_ids, [False] * len(motor_ids)
            )
            print(
                "Hardware gate: all-joint torque disabled",
                end="\r\n",
                flush=True,
            )
        self._hardware_torque_enabled = False
        self._hardware_policy_enabled = False
        self._hardware_policy_eligible = False
        self._hardware_neutral_targets = None
        self._hardware_neutral_last_time_s = None
        self._getup_failed_hold_targets = None
        self._getup_failed_hold_last_time_s = None
        self._getup_failed_gain_restored.clear()
        self._getup_failed_stale_names.clear()
        self._fall_pending_hold_targets = None
        self._getup_auto_started_s = None
        self._getup_auto_failed = False
        self._cmd_history.clear()
        self._getup_cutoff_tail_ticks = 0
        self._getup_balance_tail_ticks = 0
        self._last_sent_targets = None
        self._serial_write_hold_pending = False
        self._serial_write_hold_since_s = None
        self._overcurrent_ticks = 0

    def _remember_goal_write(self, targets: dict[str, float]) -> None:
        """Track the last goal register value for every joint written so far."""
        actual_targets = getattr(self.controller, "last_goal_targets", None)
        if actual_targets is not None:
            self._last_sent_targets = dict(actual_targets)
            return
        if self._last_sent_targets is None:
            self._last_sent_targets = {}
        self._last_sent_targets.update(targets)

    def _pace_tick(self, start_time: float) -> None:
        now = time.perf_counter()
        elapsed_s = now - start_time
        self._timing_ticks += 1
        self._timing_position_ms += self.observer.last_position_ms
        self._timing_velocity_ms += self.observer.last_velocity_ms
        self._timing_write_ms += self._last_goal_write_ms
        if elapsed_s > self.dt:
            self._timing_overruns += 1
            self._timing_max_overrun_ms = max(
                self._timing_max_overrun_ms, (elapsed_s - self.dt) * 1000.0
            )
        if now - self._timing_window_start >= 1.0:
            ticks = self._timing_ticks
            print(
                f"Control timing: ticks={ticks} overruns={self._timing_overruns} "
                f"max_late={self._timing_max_overrun_ms:.2f} ms "
                f"position_avg={self._timing_position_ms / ticks:.2f} ms "
                f"velocity_avg={self._timing_velocity_ms / ticks:.2f} ms "
                f"write_avg={self._timing_write_ms / ticks:.2f} ms "
                f"groups={getattr(self.controller, 'state_group_count', '?')} "
                f"stale={len(getattr(self.controller, 'stale_motor_names', ()))}",
                end="\r\n", flush=True,
            )
            self._timing_window_start = now
            self._timing_ticks = 0
            self._timing_overruns = 0
            self._timing_max_overrun_ms = 0.0
            self._timing_position_ms = 0.0
            self._timing_velocity_ms = 0.0
            self._timing_write_ms = 0.0
        remaining_s = self.dt - (time.perf_counter() - start_time)
        if remaining_s > 0:
            time.sleep(remaining_s)

    @staticmethod
    def _finite_vector(value, length: int) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != length:
            return False
        try:
            return all(math.isfinite(float(item)) for item in value)
        except (TypeError, ValueError, OverflowError):
            return False

    def _imu_is_safe(self, robot_state, status) -> tuple[bool, str]:
        """Validate freshness and every IMU-derived policy input."""
        if status is not None:
            if not isinstance(status, dict) or status.get("valid") is not True:
                return False, "reader reports invalid data"
            try:
                age_s = float(status.get("age_s", math.inf))
            except (TypeError, ValueError, OverflowError):
                return False, "reader age is invalid"
            if not math.isfinite(age_s) or not 0.0 <= age_s <= self._imu_max_age_s:
                return False, f"sample age {age_s!r} s exceeds {self._imu_max_age_s:.3f} s"

        if not self._finite_vector(robot_state.gyro, 3):
            return False, "gyro is missing or non-finite"
        if not self._finite_vector(robot_state.quat, 4):
            return False, "quaternion is missing or non-finite"
        quat_norm = math.sqrt(sum(float(value) ** 2 for value in robot_state.quat))
        if not 0.5 <= quat_norm <= 1.5:
            return False, f"quaternion norm {quat_norm:.3g} is invalid"
        if not self._finite_vector(robot_state.body_quat, 4):
            return False, "body quaternion is missing or non-finite"
        body_quat_norm = math.sqrt(sum(float(value) ** 2 for value in robot_state.body_quat))
        if not 0.5 <= body_quat_norm <= 1.5:
            return False, f"body quaternion norm {body_quat_norm:.3g} is invalid"
        if not self._finite_vector(robot_state.projected_gravity, 3):
            return False, "projected gravity is missing or non-finite"
        gravity_norm = math.sqrt(
            sum(float(value) ** 2 for value in robot_state.projected_gravity)
        )
        if not 0.5 <= gravity_norm <= 1.5:
            return False, f"projected-gravity norm {gravity_norm:.3g} is invalid"
        return True, ""

    @staticmethod
    def _measured_hold_command(robot_state) -> MotorCommand | None:
        targets: dict[str, float] = {}
        for name in MOTOR_TO_ID:
            value = robot_state.motor_positions.get(name)
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(numeric):
                return None
            targets[name] = numeric
        return MotorCommand(target_angles=targets)

    def _failed_getup_hold_command(self, robot_state) -> MotorCommand | None:
        """Return toward neutral after an actor fault without a target jump."""
        try:
            now = float(robot_state.time_s)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(now):
            return None

        if self._getup_failed_hold_targets is None:
            measured = self._measured_hold_command(robot_state)
            if measured is None:
                return None
            self._getup_failed_hold_targets = dict(measured.target_angles)
            self._getup_failed_hold_last_time_s = now
            self._getup_failed_stale_names = set(
                getattr(self.controller, "stale_motor_names", ())
            ) | set(getattr(self.controller, "proxy_ignored_motor_names", ()))
            # The actor reduced the 18 policy joints to P=125. Seed the freshly
            # measured goal before raising their holding gain to P=900.
            if self._hardware_policy_enabled:
                self._restore_failed_getup_gain(self._getup_failed_hold_targets)
            return measured

        previous = self._getup_failed_hold_last_time_s
        raw_dt = self.dt if previous is None else now - previous
        dt = max(
            _RECOVERY_MIN_DT_S,
            min(_RECOVERY_MAX_DT_S, raw_dt if math.isfinite(raw_dt) else self.dt),
        )
        self._getup_failed_hold_last_time_s = now
        max_step = _RECOVERY_SLEW_RATE_RAD_S * dt
        stale_names = set(getattr(self.controller, "stale_motor_names", ()))
        stale_names.update(getattr(self.controller, "proxy_ignored_motor_names", ()))
        for name, current in self._getup_failed_hold_targets.items():
            if name in stale_names:
                # No command reaches an unresponsive servo. Keep its last goal
                # instead of accumulating a large unsent neutral movement.
                self._getup_failed_stale_names.add(name)
                continue
            if name in self._getup_failed_stale_names:
                try:
                    measured = float(robot_state.motor_positions[name])
                except (KeyError, TypeError, ValueError, OverflowError):
                    return None
                if not math.isfinite(measured):
                    return None
                self._getup_failed_hold_targets[name] = measured
                self._getup_failed_stale_names.discard(name)
                continue
            neutral = NEUTRAL_POSE[name]
            delta = max(-max_step, min(max_step, neutral - current))
            self._getup_failed_hold_targets[name] = current + delta
        if self._hardware_policy_enabled:
            # A servo excluded during the initial failure may recover later.
            # Restore its gain only after its current goal has been seeded.
            self._restore_failed_getup_gain(self._getup_failed_hold_targets)
        return MotorCommand(target_angles=dict(self._getup_failed_hold_targets))

    def _restore_failed_getup_gain(self, targets: dict[str, float]) -> None:
        """Seed responsive joints, then restore neutral holding gain once."""
        ignored = set(getattr(self.controller, "stale_motor_names", ()))
        ignored.update(getattr(self.controller, "proxy_ignored_motor_names", ()))
        names = [
            name for name in MOTOR_TO_ID
            if name not in ignored and name not in self._getup_failed_gain_restored
        ]
        if not names:
            return
        try:
            ids = [MOTOR_TO_ID[name] for name in names]
            self.controller.sync_write_goal_position(
                ids, [targets[name] for name in names]
            )
            self._remember_goal_write(targets)

            # RobotController can defer a goal write for a recovering joint.
            # Never raise its gain against a goal older than this measured pose.
            actual_goals = getattr(self.controller, "last_goal_targets", None)
            if actual_goals is not None:
                ready = []
                for name in names:
                    try:
                        actual = float(actual_goals[name])
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue
                    if math.isfinite(actual) and abs(actual - targets[name]) <= 0.03:
                        ready.append(name)
                names = ready
            if names:
                ids = [MOTOR_TO_ID[name] for name in names]
                self.controller.sync_write_kp(
                    ids, [KP_HARDWARE_NEUTRAL] * len(ids)
                )
                self._getup_failed_gain_restored.update(names)
                print(
                    f"Get-up fallback: neutral holding gain restored on {len(ids)} joints",
                    end="\r\n", flush=True,
                )
        except (RuntimeError, OSError) as exc:
            print(
                f"Warning: get-up fallback gain restore failed; keeping current gain: {exc}",
                end="\r\n", flush=True,
            )

    def _fall_pending_hold_command(self, robot_state) -> MotorCommand | None:
        """Hold one measured pose during the fall debounce window."""
        if self._fall_pending_hold_targets is None:
            measured = self._measured_hold_command(robot_state)
            if measured is None:
                return None
            self._fall_pending_hold_targets = dict(measured.target_angles)
        return MotorCommand(target_angles=dict(self._fall_pending_hold_targets))

    def _walk_can_take_over(self, user_input: UserInput) -> bool:
        """True when the requested walk move may own the legs after a get-up.

        It must be requested, commanded to stand still (so the hand-back itself
        never starts a walk) and report that it actively balances.
        """
        walk_move = self.registered_moves.get("walk")
        if walk_move is None or "walk" not in user_input.active_moves:
            return False
        try:
            moving = any(
                abs(float(user_input.velocity.get(axis, 0.0))) > 1e-9
                for axis in ("vx", "vy", "vtheta")
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return not moving and walk_move.can_balance(user_input) is True

    def _update_settle_count(self, projected_gravity: list[float] | None) -> None:
        """Count consecutive ticks with tilt from HOME below the settle angle.

        An unsafe IMU (None) or invalid vector resets the count: settling must
        be observed, never assumed.
        """
        tilt = (
            None
            if projected_gravity is None or not self._finite_vector(projected_gravity, 3)
            else tilt_from_home_rad(projected_gravity)
        )
        if tilt is not None and tilt < self._settle_tilt_rad:
            self._settled_tick_count += 1
        else:
            self._settled_tick_count = 0

    def _tilt_deg(self, projected_gravity: list[float]) -> str:
        """Tilt from the HOME attitude, for the hand-back log line."""
        if not self._finite_vector(projected_gravity, 3):
            return "unknown"
        tilt = tilt_from_home_rad(projected_gravity)
        return "unknown" if tilt is None else f"{math.degrees(tilt):.1f} deg from HOME"

    def _release_getup_balance(self, reason: str) -> None:
        """End standing-balance and hand back as after a normal get-up."""
        print(f"Get-up balance released: {reason}", end="\r\n", flush=True)
        self._getup_balancing = False
        self._getup_balance_request = None
        self._getup_active_override = False
        self._getup_auto_started_s = None

    def _update_getup_override(
        self,
        projected_gravity: list[float],
        *,
        balance_when_standing: bool = False,
    ) -> bool:
        """Update fall/stand debounce; return True during the pre-get-up hold.

        On standing up, the override is released (the caller passes
        ``balance_when_standing=False`` only when a balancing walk is requested
        and the stance has settled, or when no policy may run), or, with
        ``balance_when_standing``, kept as standing balance. A new fall while
        balancing turns it back into a get-up attempt with a fresh time limit.
        """
        if not self._finite_vector(projected_gravity, 3):
            return False

        gz = float(projected_gravity[2])
        tilt = tilt_from_home_rad(projected_gravity)
        if gz > self._fall_threshold:
            self._fallen_tick_count += 1
            self._standing_tick_count = 0
        elif tilt is not None and tilt < self._stand_tilt_rad:
            self._standing_tick_count += 1
            self._fallen_tick_count = 0
        else:
            self._fallen_tick_count = 0
            self._standing_tick_count = 0

        fallen = self._fallen_tick_count >= self._fall_debounce_ticks
        if not self._getup_active_override:
            if fallen:
                self._getup_active_override = True
        elif self._getup_balancing:
            if fallen:
                self._getup_balancing = False
                self._getup_auto_started_s = None
                print(
                    "Fall detected while balancing; new get-up attempt",
                    end="\r\n", flush=True,
                )
        elif self._standing_tick_count >= self._stand_debounce_ticks:
            if balance_when_standing:
                self._getup_balancing = True
                self._getup_auto_started_s = None
                self._getup_settle_wait_ticks = 0
                print(
                    "Get-up: standing; get-up actor keeps balancing until a "
                    "balancing walk is requested and the stance has settled",
                    end="\r\n", flush=True,
                )
            else:
                self._getup_active_override = False
        return not self._getup_active_override and self._fallen_tick_count > 0

    def _stop_auto_getup(self, reason: str) -> None:
        """Stop this attempt and keep torque on during neutral return."""
        if not self._getup_auto_failed:
            print(f"Automatic get-up stopped: {reason}", end="\r\n", flush=True)
        self._getup_auto_failed = True

    def _cleanup(self) -> None:
        """Disable torque, stop input source, and clear stop artifacts."""
        if self._cleanup_done:
            return

        self._cleanup_done = True

        if self.input_source:
            self.input_source.stop()

        shutdown = getattr(self.controller, "shutdown", None)
        if callable(shutdown):
            shutdown()

        motor_ids = list(MOTOR_TO_ID.values())
        self.controller.sync_write_torque_enable(motor_ids, [False] * len(motor_ids))
        print("Torque disabled on all motors", end="\r\n", flush=True)

        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

    def _estimate_total_current(self, robot_state, target_angles: dict[str, float]) -> float:
        """Estimate the total pack current [A] from whichever signal is available.

        - If present_current was read (Observer.observe_current = True), use the measured
          sum of |current| over all motors.
        - Otherwise, fall back to the bam XL330 m6 current proxy (no extra bus read), from
          the (delay-aligned) command target, present_position and present_velocity:
            duty = clip(PROXY_KP * PROXY_ERROR_GAIN * (target - q), ±PROXY_MAX_PWM)
            I    = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R, |I| capped at BAM_MAX_CURRENT
          The back-EMF term (PROXY_KT * dq) keeps the estimate low when the motor is moving
          toward an overshot RL target, while still flagging stalled motors (low dq, high error).
        """
        currents = robot_state.motor_currents
        stale_names = getattr(
            self.controller,
            "proxy_ignored_motor_names",
            getattr(self.controller, "stale_motor_names", frozenset()),
        )
        if currents:
            return sum(abs(c) for name, c in currents.items() if name not in stale_names)

        positions = robot_state.motor_positions
        velocities = robot_state.motor_velocities
        total = 0.0
        for name, target in target_angles.items():
            if name in stale_names:
                continue
            pos = positions.get(name)
            if pos is None:
                continue
            dq = velocities.get(name, 0.0)
            duty = PROXY_KP * PROXY_ERROR_GAIN * (target - pos)
            duty = max(-PROXY_MAX_PWM, min(PROXY_MAX_PWM, duty))
            current = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R
            total += min(abs(current), BAM_MAX_CURRENT)
        return total

    def _check_overcurrent(self, robot_state, target_angles: dict[str, float], cutoff: float) -> bool:
        """Report when the estimated total pack current stays above `cutoff`
        for OVERCURRENT_DEBOUNCE_TICKS consecutive ticks.

        When True, the run loop breaks and _cleanup() disables torque on every motor,
        leaving the robot compliant so the BMS does not trip on the current spike.
        """
        total_current = self._estimate_total_current(robot_state, target_angles)
        if total_current >= cutoff:
            self._overcurrent_ticks += 1
        else:
            self._overcurrent_ticks = 0

        if self._overcurrent_ticks >= OVERCURRENT_DEBOUNCE_TICKS:
            print(
                f"Overcurrent safety triggered: {total_current:.2f} A (threshold "
                f"{cutoff:.2f} A) — disabling torque",
                end="\r\n",
                flush=True,
            )
            return True
        return False

    def _send_to_motors(
        self, command: MotorCommand, *, neutral_return: bool = False
    ):
        """Send one batched goal position command from the composed command dict."""
        if not command.target_angles:
            return

        motor_ids = [MOTOR_TO_ID[name] for name in command.target_angles]
        target_positions = list(command.target_angles.values())

        write_start = time.perf_counter()
        try:
            write_goals = self.controller.sync_write_goal_position
            if neutral_return:
                write_goals = getattr(
                    self.controller, "sync_write_neutral_goal_position", write_goals
                )
            write_goals(motor_ids, target_positions)
        finally:
            self._last_goal_write_ms = (time.perf_counter() - write_start) * 1000.0
