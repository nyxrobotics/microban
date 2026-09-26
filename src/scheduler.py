# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time
import math
from collections import deque
from pathlib import Path
from typing import Optional


from constants import (
    MOTOR_TO_ID,
    KP_DEFAULT,
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
from imu_reader import imu_quat_to_body
from observer import Observer, Observation, RobotState
from input.input_source import InputSource, UserInput, scale_velocity
from moves.move import MotorCommand, Move, MoveState
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from moves.walk import WalkMove


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
        # Thresholds on projected_gravity[2]: -1 is perfectly upright, 0 is on its side.
        # Fall threshold matches WalkMove's own safety-stop criterion, for consistency.
        self._fall_threshold = -0.5
        self._stand_threshold = -0.9
        self._fall_debounce_ticks = 15  # ~0.3 s at 50 Hz
        self._stand_debounce_ticks = 20  # ~0.4 s at 50 Hz
        self._fallen_tick_count = 0
        self._standing_tick_count = 0
        self._getup_active_override = False
        # getup.onnx was retrained with the raw/pre-clip observation bug fixed
        # (the "last action" term now records the actually-applied, clipped
        # target instead of the raw un-clipped network output) and redeployed
        # 2026-09-26. Auto-trigger uses bounded target speed, progress, and
        # time limits. The manual "g" toggle remains available.
        self._getup_auto_trigger_enabled = True
        self._getup_auto_started_s: float | None = None
        self._getup_auto_best_gravity_z = 1.0
        self._getup_auto_failed = False
        # History of sent target_angles, to align the current proxy with the delayed feedback:
        # the oldest entry is the command issued OVERCURRENT_PROXY_DELAY_TICKS ticks ago.
        self._cmd_history: deque[dict[str, float]] = deque(maxlen=OVERCURRENT_PROXY_DELAY_TICKS + 1)

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
                            if snapshot.hold_last_targets:
                                self.input_source.set_motion_inhibited(True)
                                self._safety_hold_active = True
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
                if user_input.hold_last_targets:
                    if not self._network_hold_active:
                        print(
                            "Network input unavailable; holding last motor goals and torque",
                            end="\r\n",
                            flush=True,
                        )
                        self._network_hold_active = True
                    if self.input_source is not None:
                        self.input_source.set_motion_inhibited(True)
                    self._safety_hold_active = True
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

                # Fall / getup auto-switch: forces "getup" on (and "walk" off) after a
                # sustained fall, and hands back to the user's own "walk" toggle once
                # stood up (and stable) again. No-op if "getup" isn't registered.
                fall_pending = False
                if "getup" in self.registered_moves and self._getup_auto_trigger_enabled:
                    if imu_safe:
                        fall_pending = self._update_getup_override(
                            obs.robot_state.projected_gravity
                        )
                    if self._getup_active_override:
                        obs.user_input.active_moves = (obs.user_input.active_moves | {"getup"}) - {"walk"}
                        can_attempt = imu_safe and hardware_mode == "policy"
                        if can_attempt and self._getup_auto_started_s is None:
                            self._getup_auto_started_s = start_time
                            self._getup_auto_best_gravity_z = float(
                                obs.robot_state.projected_gravity[2]
                            )
                            print("Automatic get-up attempt started", end="\r\n", flush=True)
                        if can_attempt and self._getup_auto_started_s is not None:
                            self._getup_auto_best_gravity_z = min(
                                self._getup_auto_best_gravity_z,
                                float(obs.robot_state.projected_gravity[2]),
                            )
                            elapsed_s = start_time - self._getup_auto_started_s
                            if elapsed_s >= 8.0 or (
                                elapsed_s >= 4.0 and self._getup_auto_best_gravity_z > -0.65
                            ):
                                self._stop_auto_getup("time limit or no upright progress")
                        obs.user_input.getup_armed = can_attempt and not self._getup_auto_failed
                        fall_pending = fall_pending or self._getup_auto_failed
                    else:
                        self._getup_auto_started_s = None
                        self._getup_auto_failed = False

                # Keep the network deadman disarmed for the complete fault/fall/get-up
                # interval. Removing the inhibit does not arm it: a later released
                # trigger snapshot is required before walking can resume.
                motion_inhibited = (
                    not imu_safe
                    or fall_pending
                    or self._getup_active_override
                    or hardware_mode != "policy"
                    or (self._serial_write_hold_pending and self._serial_hold_extended)
                )
                if self.input_source:
                    self.input_source.set_motion_inhibited(motion_inhibited)
                    set_head_telemetry = getattr(
                        self.input_source, "set_head_telemetry", None
                    )
                    if callable(set_head_telemetry):
                        trunk_angles = _trunk_roll_pitch(obs.robot_state.body_quat)
                        if trunk_angles is not None:
                            positions = obs.robot_state.motor_positions
                            set_head_telemetry(
                                head=float(positions.get("head", 0.0)),
                                neck_roll=float(positions.get("neck_roll", 0.0)),
                                neck_pitch=float(positions.get("neck_pitch", 0.0)),
                                trunk_roll=trunk_angles[0],
                                trunk_pitch=trunk_angles[1],
                            )

                hold_for_safety = (
                    hardware_mode != "limp" and (not imu_safe or fall_pending)
                )
                if (
                    not imu_safe
                    and hardware_mode != "limp"
                    and self._imu_unsafe_since_s is not None
                    and start_time - self._imu_unsafe_since_s >= self._imu_shutdown_after_s
                ):
                    print(
                        f"IMU remained unsafe for {self._imu_shutdown_after_s:.2f} s "
                        "— stopping control loop and disabling torque",
                        end="\r\n",
                        flush=True,
                    )
                    break

                # Update move states and dispatch one call per move per tick
                if hardware_mode == "limp":
                    # B, link loss, or process startup: torque is physically
                    # off and no goal position is written.
                    command = MotorCommand(target_angles={})
                    self._safety_hold_active = False
                    self._serial_write_hold_pending = False
                    self._serial_write_hold_since_s = None
                elif hold_for_safety:
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
                        self._getup_active_override
                        and getup_move is not None
                        and getattr(getup_move, "policy_faulted", False)
                    ):
                        self._stop_auto_getup("get-up actor output exceeded bounds")
                        held = self._measured_hold_command(robot_state)
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
                    self._overcurrent_ticks = 0
                else:
                    aligned_targets = (
                        self._cmd_history[0]
                        if self._cmd_history
                        else command.target_angles
                    )
                    cutoff = (
                        OVERCURRENT_CUTOFF_A_GETUP
                        if "getup" in obs.user_input.active_moves
                        else OVERCURRENT_CUTOFF_A
                    )
                    if self._check_overcurrent(robot_state, aligned_targets, cutoff):
                        break

                    # Send command to motors
                    try:
                        self._send_to_motors(command)
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

            motor_ids = list(MOTOR_TO_ID.values())
            self.controller.sync_write_kp(
                motor_ids, [KP_DEFAULT] * len(motor_ids)
            )
            # A servo can retain an old goal register while torque is off.
            # Seed every goal with the freshly measured pose *before* enabling
            # torque, otherwise A could cause a short jump toward that stale
            # goal before the first neutral-slew tick is written.
            self.controller.sync_write_goal_position(
                motor_ids, [measured[name] for name in MOTOR_TO_ID]
            )
            self._remember_goal_write(measured)
            self.controller.sync_write_torque_enable(
                motor_ids, [True] * len(motor_ids)
            )
            self._hardware_torque_enabled = True
            self._hardware_policy_enabled = False
            self._hardware_neutral_targets = measured
            self._hardware_neutral_last_time_s = obs.robot_state.time_s
            self._cmd_history.clear()
            self._overcurrent_ticks = 0
            just_enabled = True
            print(
                "Hardware gate: torque enabled; policy withheld, returning to neutral",
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
            # this ordering, raising an RL joint from KP_RL to KP_DEFAULT could
            # briefly amplify its error against a stale policy goal.
            self.controller.sync_write_goal_position(
                motor_ids, [measured[name] for name in MOTOR_TO_ID]
            )
            self._remember_goal_write(measured)
            self.controller.sync_write_kp(
                motor_ids, [KP_DEFAULT] * len(motor_ids)
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
        dt = max(0.001, min(0.1, raw_dt if math.isfinite(raw_dt) else self.dt))
        self._hardware_neutral_last_time_s = now
        max_step = 0.5 * dt

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
        self._cmd_history.clear()
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

    def _update_getup_override(self, projected_gravity: list[float]) -> bool:
        """Update fall/stand debounce; return True during the pre-get-up hold."""
        if not self._finite_vector(projected_gravity, 3):
            return False

        gz = float(projected_gravity[2])
        if gz > self._fall_threshold:
            self._fallen_tick_count += 1
            self._standing_tick_count = 0
        elif gz < self._stand_threshold:
            self._standing_tick_count += 1
            self._fallen_tick_count = 0
        else:
            self._fallen_tick_count = 0
            self._standing_tick_count = 0

        if not self._getup_active_override and self._fallen_tick_count >= self._fall_debounce_ticks:
            self._getup_active_override = True
        elif self._getup_active_override and self._standing_tick_count >= self._stand_debounce_ticks:
            self._getup_active_override = False
        return not self._getup_active_override and self._fallen_tick_count > 0

    def _stop_auto_getup(self, reason: str) -> None:
        """Stop this attempt while retaining torque and measured-pose hold."""
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

    def _send_to_motors(self, command: MotorCommand):
        """Send one batched goal position command from the composed command dict."""
        if not command.target_angles:
            return

        motor_ids = [MOTOR_TO_ID[name] for name in command.target_angles]
        target_positions = list(command.target_angles.values())

        write_start = time.perf_counter()
        try:
            self.controller.sync_write_goal_position(motor_ids, target_positions)
        finally:
            self._last_goal_write_ms = (time.perf_counter() - write_start) * 1000.0
