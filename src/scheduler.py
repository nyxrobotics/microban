# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

import time
import math
from collections import deque
from pathlib import Path
from typing import Optional


from constants import (
    MOTOR_TO_ID,
    BAM_MAX_CURRENT,
    OVERCURRENT_CUTOFF_A,
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
from observer import Observer, Observation
from input.input_source import InputSource, UserInput, scale_velocity
from moves.move import MotorCommand, Move, MoveState
from moves.rotate_head import RotateHeadMove
from moves.squat import SquatMove
from moves.walk import WalkMove


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

        self.observer = Observer(self.controller)

        # All moves are registered here. They only run when activated via user_input.active_moves.
        self.registered_moves: dict[str, Move] = moves if moves is not None else {
            "head": RotateHeadMove(),
            "squat": SquatMove(),
            "walk": WalkMove(),
        }

        self.loop_start_time = time.perf_counter()
        self._serial_errors = 0
        self._last_imu_print_s: float = 0.0
        self._last_imu_stale_warn_s: float = 0.0
        self._imu_max_age_s = imu_max_age_s
        self._imu_shutdown_after_s = imu_shutdown_after_s
        self._imu_unsafe_since_s: float | None = None
        self._safety_hold_active = False
        self._overcurrent_ticks = 0
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

                # Read robot observations and user input
                try:
                    robot_state = self.observer.read_state(self.dt)
                except RuntimeError as e:
                    self._serial_errors += 1
                    if self._serial_errors >= 3:
                        print(f"Serial communication error: {e}", end="\r\n", flush=True)
                        break
                    print(f"Warning: serial read error (attempt {self._serial_errors}/3): {e}", end="\r\n", flush=True)
                    continue
                self._serial_errors = 0

                robot_state.time_s = start_time - self.loop_start_time
                user_input = self.input_source.read() if self.input_source else UserInput()
                user_input.velocity = scale_velocity(user_input.velocity)
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

                motion_inhibited = not imu_safe
                if self.input_source:
                    self.input_source.set_motion_inhibited(motion_inhibited)

                hold_for_safety = not imu_safe
                if (
                    not imu_safe
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
                if hold_for_safety:
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

                # Overcurrent safety: estimate the current from the command that was actually
                # active when the (delayed) position/velocity feedback was sampled — the oldest
                # buffered command (== DELAY_TICKS old once full). This reconstructs the real
                # (delayed) current instead of pairing a fresh target with stale feedback, which
                # would inflate the error term and false-trigger at gait start.
                aligned_targets = self._cmd_history[0] if self._cmd_history else command.target_angles
                if self._check_overcurrent(robot_state, aligned_targets):
                    break

                # Send command to motors
                self._cmd_history.append(dict(command.target_angles))
                self._send_to_motors(command)

                # IMU / gyro terminal display
                if obs.user_input.show_imu and (start_time - self._last_imu_print_s) >= 0.5:
                    acc = obs.robot_state.acc
                    gyro = obs.robot_state.gyro
                    quat = obs.robot_state.quat
                    print("--------------------------------------------", end="\r\n", flush=True)
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

                # Sleep to keep a fixed control frequency
                elapsed_time = time.perf_counter() - start_time
                sleep_time = self.dt - elapsed_time

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    ms_late = -sleep_time * 1000
                    print(f"Warning: control loop overrun by {ms_late:.2f} ms", end="\r\n", flush=True)

        except KeyboardInterrupt:
            print("Control loop interrupted by user", end="\r\n", flush=True)
        finally:
            self._cleanup()

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
        if currents:
            return sum(abs(c) for c in currents.values())

        positions = robot_state.motor_positions
        velocities = robot_state.motor_velocities
        total = 0.0
        for name, target in target_angles.items():
            pos = positions.get(name)
            if pos is None:
                continue
            dq = velocities.get(name, 0.0)
            duty = PROXY_KP * PROXY_ERROR_GAIN * (target - pos)
            duty = max(-PROXY_MAX_PWM, min(PROXY_MAX_PWM, duty))
            current = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R
            total += min(abs(current), BAM_MAX_CURRENT)
        return total

    def _check_overcurrent(self, robot_state, target_angles: dict[str, float]) -> bool:
        """Report when the estimated total pack current stays above OVERCURRENT_CUTOFF_A
        for OVERCURRENT_DEBOUNCE_TICKS consecutive ticks.

        When True, the run loop breaks and _cleanup() disables torque on every motor,
        leaving the robot compliant so the BMS does not trip on the current spike.
        """
        total_current = self._estimate_total_current(robot_state, target_angles)
        if total_current >= OVERCURRENT_CUTOFF_A:
            self._overcurrent_ticks += 1
        else:
            self._overcurrent_ticks = 0

        if self._overcurrent_ticks >= OVERCURRENT_DEBOUNCE_TICKS:
            print(
                f"Overcurrent safety triggered: {total_current:.2f} A (threshold "
                f"{OVERCURRENT_CUTOFF_A:.2f} A) — disabling torque",
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

        self.controller.sync_write_goal_position(motor_ids, target_positions)