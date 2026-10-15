# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 nyxrobotics

"""Select PICO locomotion or whole-body tracking; hold position if no actor is available.

Whole-body tracking (moves/wholebody.py) runs while the PC's reference stream
says the operator started it (CALIBRATED and ACTIVE) and the wire asks for
``pico_teleop`` or ``wholebody`` with the policy enabled (R3); the PICO
triggers and joystick are not used meanwhile.  It starts only from a robot
standing still on the standing policy, as training's episodes start: the left
trigger released (``balance_only``), the trunk within
WHOLEBODY_START_TILT_DEG of the HOME attitude and the legs still for
WHOLEBODY_START_TICKS ticks; until then the activation waits and the standing
policy keeps the robot (a walking or stepping robot is never switched).  When
the PC stops, the stream
goes stale, the trunk leaves the reference's attitude or ``walk`` is asked
for, the move holds, returns to the HOME stance and hands the robot to the
standing policy: the PICO policy when it is installed, else the position
hold.  A stopped or refused activation is not tracked again; the PC's next
start (ACTIVE off then on, or RESTART) is.  While tracking, the arm overlay
(pico_arms) and the HMD head move yield (``WholebodyClaim``): the policy
drives the arms and the move drives the neck to the reference's neck angles.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path

from constants import KP_HARDWARE_NEUTRAL, MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from home_pose import tilt_from_home_rad
from moves.move import MotorCommand, Move, MoveState, WholebodyClaim
from moves.pico_arms import PICO_ARM_JOINT_ORDER, PicoArmTargetHold
from moves.pico_hybrid import AGENT_NAME as PICO_AGENT_NAME
from moves.pico_hybrid import PicoHybridMove
from moves.wholebody import HANDOFF_TILT_DEG, WholebodyMove
from observer import Observation
from wholebody_contract import POLICY_FILE as WHOLEBODY_POLICY_FILE

POLICY_NAMES = frozenset({"walk", "pico_teleop", "wholebody"})
# Whole-body tracking starts after this many ticks (0.5 s at 50 Hz) of standing
# still on the standing policy: the trunk within the hand-off's tilt of HOME
# and every leg joint slower than WHOLEBODY_START_LEG_SPEED_RAD_S.  The PICO
# policy keeps stepping for 0.4-1.4 s after the trigger is released with its
# trunk within 11 deg of HOME (runtime simulation): the trunk alone does not
# tell a step from a stand; the legs move at 1.7-4.6 rad/s while stepping and
# below 0.4 rad/s standing.
WHOLEBODY_START_TILT_DEG = HANDOFF_TILT_DEG
WHOLEBODY_START_LEG_SPEED_RAD_S = 1.0
WHOLEBODY_START_TICKS = 25
LEG_JOINTS = tuple(name for name in OBSERVATION_DOF_ORDER if name not in PICO_ARM_JOINT_ORDER)

# Builds the learned move from the installed policy (tests inject their own).
LearnedMoveFactory = Callable[[ControllerProtocol | None, Path], Move]
PolicyFingerprint = tuple[int, int, int, int]
ALL_MOTOR_IDS = list(MOTOR_TO_ID.values())


def _pico_hybrid_factory(arm_target_hold: PicoArmTargetHold | None) -> LearnedMoveFactory:
    def build(controller: ControllerProtocol | None, policy_path: Path) -> Move:
        return PicoHybridMove(
            controller=controller,
            policy_path=policy_path,
            arm_target_hold=arm_target_hold,
        )

    return build


class _HoldPositionMove(Move):
    """Keep the last complete body goal."""

    def __init__(
        self,
        controller: ControllerProtocol | None,
        neutral_return_duration_s: float = 0.8,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._neutral_return_duration_s = neutral_return_duration_s
        self._pending_targets: dict[str, float] | None = None
        self._held_targets: dict[str, float] = {}
        self._stop_start_time_s: float | None = None
        self._stop_start_angles: dict[str, float] = {}
        # Set by every start: the hold may take over from GetupMove or from a
        # failed PICO actor, both of which leave every joint at the soft
        # learned-policy gain (KP_RL). A static hold at that gain cannot carry
        # the robot and slowly topples it; a static hold is P900 on all joints.
        self._restore_gain_pending = False

    def can_balance(self, user_input) -> bool:
        # A fixed goal hold has no feedback: it cannot keep the robot upright.
        _ = user_input
        return False

    def seed_targets(self, targets: Mapping[str, float] | None) -> None:
        if targets is None:
            self._pending_targets = None
            return
        try:
            complete = {
                name: float(targets[name]) for name in OBSERVATION_DOF_ORDER
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            self._pending_targets = None
            return
        self._pending_targets = (
            complete if all(math.isfinite(value) for value in complete.values()) else None
        )

    def _measured_targets(self, obs: Observation) -> dict[str, float]:
        measured: dict[str, float] = {}
        for name in OBSERVATION_DOF_ORDER:
            previous = self._held_targets.get(name, NEUTRAL_POSE[name])
            try:
                value = float(obs.robot_state.motor_positions[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                value = previous
            measured[name] = value if math.isfinite(value) else previous
        return measured

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        self._held_targets = self._pending_targets or self._measured_targets(obs)
        self._pending_targets = None
        self._stop_start_time_s = None
        self._stop_start_angles = {}
        self._restore_gain_pending = True
        command.target_angles.update(self._held_targets)
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        _ = obs
        if self._restore_gain_pending:
            # First step after the start (the next tick after a get-up handoff,
            # whose on_stop wrote KP_RL and the measured goals; the same tick
            # after a PICO failure). Re-seed the held goals first, then raise
            # every joint, the camera joints included, so P900 never acts on an
            # old saturated get-up target.
            self._restore_gain_pending = False
            if self._controller is not None:
                ids = [MOTOR_TO_ID[name] for name in OBSERVATION_DOF_ORDER]
                self._controller.sync_write_goal_position(
                    ids, [self._held_targets[name] for name in OBSERVATION_DOF_ORDER]
                )
                self._controller.sync_write_kp(
                    ALL_MOTOR_IDS, [KP_HARDWARE_NEUTRAL] * len(ALL_MOTOR_IDS)
                )
        command.target_angles.update(self._held_targets)

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if "getup" in obs.user_input.active_moves:
            self.state = MoveState.INACTIVE
            return
        if self._stop_start_time_s is None:
            self._stop_start_time_s = float(obs.robot_state.time_s)
            self._stop_start_angles = self._measured_targets(obs)
        elapsed = max(0.0, float(obs.robot_state.time_s) - self._stop_start_time_s)
        fraction = min(1.0, elapsed / max(1e-6, self._neutral_return_duration_s))
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        for name in OBSERVATION_DOF_ORDER:
            start = self._stop_start_angles[name]
            command.target_angles[name] = (
                NEUTRAL_POSE[name]
                if fraction >= 1.0
                else start + (NEUTRAL_POSE[name] - start) * blend
            )
        if fraction >= 1.0:
            if self._controller is not None:
                self._controller.sync_write_kp(
                    ALL_MOTOR_IDS, [KP_HARDWARE_NEUTRAL] * len(ALL_MOTOR_IDS)
                )
            self.state = MoveState.INACTIVE

    def on_safety_resume(self, obs: Observation) -> None:
        self._held_targets = self._measured_targets(obs)
        self._stop_start_time_s = None
        self._stop_start_angles = {}


class PolicySelectableWalkMove(Move):
    """Contain optional learned-policy faults behind a fixed-position hold.

    A missing/rejected model or an exception from learned-policy start/inference
    holds the last complete body goal in the same scheduler cycle. Once degraded,
    an activation stays on hold until the
    trigger is released, avoiding a surprise policy switch when tracking recovers.
    Atomic file replacements can be parsed in the background and used on a later
    activation without restarting the process.
    """

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        pico_policy_path: str | Path = Path("src/agents") / PICO_AGENT_NAME,
        *,
        fallback_move: Move | None = None,
        pico_move: Move | None = None,
        learned_move_factory: LearnedMoveFactory | None = None,
        arm_target_hold: PicoArmTargetHold | None = None,
        wholebody_reference=None,
        wholebody_policy_path: str | Path = Path("src/agents") / WHOLEBODY_POLICY_FILE,
        wholebody_move: Move | None = None,
        wholebody_claim: WholebodyClaim | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._pico_policy_path = Path(pico_policy_path)
        # The PICO policy observes and holds the arm targets the direct-arm
        # overlay writes into ``arm_target_hold`` (shared with PicoArmTrackingMove).
        self._learned_move_factory = learned_move_factory or _pico_hybrid_factory(
            arm_target_hold
        )
        self._children: dict[str, Move] = {
            "walk": fallback_move or _HoldPositionMove(controller=controller),
        }
        if pico_move is not None:
            self._children["pico_teleop"] = pico_move
        # The whole-body move needs the reference receiver; without one the
        # mode is unavailable (and requesting it stands on the standing policy).
        self._wholebody_reference = wholebody_reference
        self._wholebody_policy_path = Path(wholebody_policy_path)
        self._injected_wholebody = wholebody_move
        self.wholebody_unavailable_reason: str | None = None
        self._last_learned_targets: dict[str, float] | None = None
        self._claim = wholebody_claim if wholebody_claim is not None else WholebodyClaim()

        self._selected_name: str | None = None
        self._selected: Move | None = None
        self._fallback_latched = False
        self._fallback_reason: str | None = None
        self._last_pico_targets: dict[str, float] | None = None
        self._last_balance_only = False
        # Consecutive ticks standing still on the standing policy (see
        # _count_standing); whole-body tracking starts at WHOLEBODY_START_TICKS.
        self._standing_ticks = 0

        self._reload_lock = threading.Lock()
        self._reload_thread: threading.Thread | None = None
        self._reload_result: tuple[
            PolicyFingerprint, Move | None, BaseException | None
        ] | None = None
        self._loaded_fingerprint: PolicyFingerprint | None = (
            self._policy_fingerprint() if pico_move is not None else None
        )
        self._last_attempted_fingerprint: PolicyFingerprint | None = None

    @property
    def effective_policy(self) -> str | None:
        """Policy currently owning locomotion joints, for diagnostics only."""

        return self._selected_name

    @property
    def fallback_reason(self) -> str | None:
        """Latest learned-policy degradation reason; never a control gate."""

        return self._fallback_reason

    @property
    def fallback_latched(self) -> bool:
        return self._fallback_latched

    def _policy_fingerprint(self) -> PolicyFingerprint | None:
        try:
            status = self._pico_policy_path.stat()
        except OSError:
            return None
        if not self._pico_policy_path.is_file():
            return None
        return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)

    @staticmethod
    def _describe_failure(stage: str, exc: BaseException) -> str:
        detail = str(exc).strip()
        suffix = f": {detail}" if detail else ""
        return f"learned policy {stage} failed ({type(exc).__name__}){suffix}"

    def _record_fallback(self, reason: str) -> None:
        self._fallback_latched = True
        self._fallback_reason = reason
        action = (
            "holding last body targets"
            if isinstance(self._children["walk"], _HoldPositionMove)
            else "continuing with walk"
        )
        print(f"PICO teleop fallback: {reason}; {action}", flush=True)

    def _build_learned_move(self) -> Move:
        child = self._learned_move_factory(
            self._controller,
            self._pico_policy_path,
        )
        child.preload()
        return child

    def _preload_wholebody(self) -> None:
        if self._injected_wholebody is not None:
            self._injected_wholebody.preload()
            self._children["wholebody"] = self._injected_wholebody
            return
        if self._wholebody_reference is None:
            self.wholebody_unavailable_reason = "no whole-body reference receiver"
            return
        if not self._wholebody_policy_path.is_file():
            self.wholebody_unavailable_reason = (
                f"whole-body policy is not installed at {self._wholebody_policy_path}"
            )
            return
        try:
            child = WholebodyMove(
                reference=self._wholebody_reference,
                controller=self._controller,
                policy_path=self._wholebody_policy_path,
            )
            child.preload()
        except Exception as exc:  # noqa: BLE001 - optional parser/runtime boundary
            self.wholebody_unavailable_reason = self._describe_failure("whole-body load", exc)
            print(f"Whole-body tracking unavailable: {self.wholebody_unavailable_reason}", flush=True)
            return
        self._children["wholebody"] = child
        self.wholebody_unavailable_reason = None
        print(
            f"Whole-body policy {child.contract.checkpoint_filename} loaded "
            f"(self-test {child.self_test_rows} rows"
            f"{', DRY RUN' if child.contract.dry_run else ''})",
            flush=True,
        )

    def preload(self) -> None:
        # The fixed-position hold is always available; an injected fallback
        # move (tests) is preloaded the same way.
        self._children["walk"].preload()
        self._preload_wholebody()

        injected = self._children.get("pico_teleop")
        if injected is not None:
            try:
                injected.preload()
            except Exception as exc:  # noqa: BLE001 - optional plugin boundary
                self._children.pop("pico_teleop", None)
                self._fallback_reason = self._describe_failure("preload", exc)
            return

        fingerprint = self._policy_fingerprint()
        if fingerprint is None:
            self._fallback_reason = (
                f"learned policy is not installed at {self._pico_policy_path}"
            )
            return
        self._last_attempted_fingerprint = fingerprint
        try:
            self._children["pico_teleop"] = self._build_learned_move()
        except Exception as exc:  # noqa: BLE001 - optional parser/runtime boundary
            self._fallback_reason = self._describe_failure("load", exc)
            return
        self._loaded_fingerprint = fingerprint
        self._fallback_reason = None

    def request_learned_reload(self, *, force: bool = False) -> bool:
        """Load the installed learned policy off the control-loop thread.

        A valid current policy remains available while a replacement is checked.
        The result is adopted on a later control tick, but a fallback-latched
        activation never hot-swaps back from the baseline.
        """

        fingerprint = self._policy_fingerprint()
        if fingerprint is None:
            self._fallback_reason = (
                f"learned policy is not installed at {self._pico_policy_path}"
            )
            return False
        with self._reload_lock:
            if self._reload_thread is not None and self._reload_thread.is_alive():
                return False
            if not force and fingerprint in (
                self._loaded_fingerprint,
                self._last_attempted_fingerprint,
            ):
                return False
            self._last_attempted_fingerprint = fingerprint
            self._reload_result = None
            worker = threading.Thread(
                target=self._reload_worker,
                args=(fingerprint,),
                name="pico-policy-reload",
                daemon=True,
            )
            self._reload_thread = worker
            worker.start()
        return True

    def _reload_worker(self, fingerprint: PolicyFingerprint) -> None:
        child: Move | None = None
        error: BaseException | None = None
        try:
            child = self._build_learned_move()
        except Exception as exc:  # noqa: BLE001 - retain worker/ORT diagnostics
            error = exc
        with self._reload_lock:
            self._reload_result = (fingerprint, child, error)

    def _poll_reload(self) -> None:
        with self._reload_lock:
            result = self._reload_result
            if result is None:
                return
            self._reload_result = None
            self._reload_thread = None
        fingerprint, child, error = result
        if error is not None or child is None:
            assert error is not None
            self._fallback_reason = self._describe_failure("reload", error)
            return
        # The artifact may have been atomically replaced again during parsing.
        if fingerprint != self._policy_fingerprint():
            return
        self._children["pico_teleop"] = child
        self._loaded_fingerprint = fingerprint

    def _notice_replacement(self) -> None:
        fingerprint = self._policy_fingerprint()
        if fingerprint is not None and fingerprint != self._loaded_fingerprint:
            self.request_learned_reload()

    @staticmethod
    def _requested_policy(obs: Observation) -> str:
        name = obs.user_input.locomotion_policy
        if name not in POLICY_NAMES:
            raise RuntimeError(f"unsupported locomotion policy: {name!r}")
        return name

    def _start_child(
        self,
        name: str,
        child: Move,
        obs: Observation,
        command: MotorCommand,
    ) -> None:
        if self._selected is not None and self._selected is not child:
            self._selected.state = MoveState.INACTIVE
        child.state = MoveState.STARTING
        child.on_start(obs, command)
        if child.state != MoveState.ACTIVE:
            raise RuntimeError(f"locomotion child {name!r} failed to start")
        self._selected_name = (
            "hold" if name == "walk" and isinstance(child, _HoldPositionMove) else name
        )
        self._selected = child
        self._claim.active = name == "wholebody"
        self.state = MoveState.ACTIVE

    def _remember_pico_targets(self, command: MotorCommand) -> None:
        try:
            targets = {
                name: float(command.target_angles[name])
                for name in OBSERVATION_DOF_ORDER
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if all(math.isfinite(value) for value in targets.values()):
            self._last_pico_targets = targets

    def _start_fallback(
        self,
        obs: Observation,
        command: MotorCommand,
        *,
        step_now: bool,
    ) -> None:
        previous = self._selected
        fallback = self._children["walk"]
        if isinstance(fallback, _HoldPositionMove):
            fallback.seed_targets(
                self._last_pico_targets
                if self._selected_name == "pico_teleop"
                else None
            )
        if previous is not None and previous is not fallback:
            previous.state = MoveState.INACTIVE
        self._start_child("walk", fallback, obs, command)
        if step_now:
            fallback.step(obs, command)

    def _fallback_from_exception(
        self,
        stage: str,
        exc: BaseException,
        obs: Observation,
        command: MotorCommand,
        *,
        step_now: bool,
    ) -> None:
        self._children.pop("pico_teleop", None)
        self._loaded_fingerprint = None
        reason = self._describe_failure(stage, exc)
        self._record_fallback(reason)
        # Parsing may finish during this fallback activation; adoption waits for the
        # next trigger release/press boundary.
        if self._policy_fingerprint() is not None:
            self.request_learned_reload(force=True)
        # A missing file must not overwrite the more useful runtime failure above.
        self._fallback_reason = reason
        self._start_fallback(obs, command, step_now=step_now)

    def can_balance(self, user_input) -> bool:
        """Whether on_start for ``user_input`` would select a balancing child.

        Mirrors on_start's choice: a loaded learned policy when it is
        requested and its tracking is not degraded, otherwise the fallback
        child (the static _HoldPositionMove on the PICO runtime); for
        ``wholebody``, the standing child (tracking never starts at on_start).
        A finished background reload is adopted first, exactly as on_start would.
        """
        requested = user_input.locomotion_policy
        if requested not in POLICY_NAMES:
            return False
        self._poll_reload()
        if requested == "wholebody":
            return self._standing_child()[1].can_balance(user_input)
        if requested == "pico_teleop" and not user_input.learned_policy_degraded:
            learned = self._children.get("pico_teleop")
            if learned is not None:
                return learned.can_balance(user_input)
        return self._children["walk"].can_balance(user_input)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        self._poll_reload()
        self._notice_replacement()
        self._fallback_latched = False
        self._last_pico_targets = None
        self._last_balance_only = bool(obs.user_input.balance_only)
        self._standing_ticks = 0
        requested = self._requested_policy(obs)
        if requested == "wholebody":
            # Tracking starts in step, once the robot stands still on the
            # standing policy (an activation of the PC waits for it).
            self._last_learned_targets = None
            refusal = self._wholebody_start_refusal()
            if refusal is not None:
                self._fallback_reason = refusal
            self._start_standing(obs, command, step_now=False)
            return
        child = self._children.get(requested)

        if requested == "pico_teleop" and child is None:
            reason = self._fallback_reason or (
                f"learned policy is unavailable at {self._pico_policy_path}"
            )
            self._record_fallback(reason)
            self._start_fallback(obs, command, step_now=False)
            return

        assert child is not None
        try:
            self._start_child(requested, child, obs, command)
            if requested == "pico_teleop":
                self._remember_pico_targets(command)
        except Exception as exc:
            if requested != "pico_teleop":
                raise
            self._fallback_from_exception(
                "start",
                exc,
                obs,
                command,
                step_now=False,
            )

    def step(self, obs: Observation, command: MotorCommand) -> None:
        self._poll_reload()
        if self._selected is None or self._selected_name is None:
            raise RuntimeError("locomotion selector is active without a child")

        balance_only = bool(obs.user_input.balance_only)
        if balance_only and not self._last_balance_only:
            # A trigger release permits one retry, not one retry per balance tick.
            self._fallback_latched = False
        self._last_balance_only = balance_only
        self._count_standing(obs)

        requested = self._requested_policy(obs)
        if (
            requested == "wholebody"
            or self._selected_name == "wholebody"
            or (requested == "pico_teleop" and self._wholebody_ready() and self._wholebody_wanted())
        ):
            self._step_wholebody_mode(obs, command, requested)
            return
        if self._selected_name == "pico_teleop" and requested != "pico_teleop":
            if obs.user_input.learned_policy_degraded:
                # Optional tracking failed. Retain trigger/stick input and latch the
                # baseline so tracking recovery cannot switch policies mid-stride.
                self._record_fallback("body tracking degraded")
            else:
                # An explicit wire-policy change to walk selects the fallback.
                # The PICO bridge always requests pico_teleop and uses only the
                # left-trigger active_moves state as its momentary enable.
                self._fallback_reason = "learned policy deselected on the wire"
                self._fallback_latched = False
            self._start_fallback(obs, command, step_now=True)
            return

        if (
            self._selected_name == "pico_teleop"
            and obs.user_input.learned_policy_degraded
        ):
            # Optional tracking failed. Retain trigger/stick input and immediately
            # use the baseline; do not switch back during this activation.
            self._record_fallback("body tracking degraded")
            self._start_fallback(obs, command, step_now=True)
            return

        if self._selected_name in ("walk", "hold"):
            if obs.user_input.learned_policy_degraded and not self._fallback_latched:
                # The activation may already be on fallback when a learned-policy
                # packet arrives with invalid trackers. Latch just as strictly as
                # degradation that occurred after learned ownership began.
                self._record_fallback("body tracking degraded")
            if requested == "pico_teleop" and not self._fallback_latched:
                learned = self._children.get("pico_teleop")
                if learned is None:
                    reason = self._fallback_reason or (
                        f"learned policy is unavailable at {self._pico_policy_path}"
                    )
                    self._record_fallback(reason)
                else:
                    try:
                        self._last_pico_targets = None
                        self._start_child("pico_teleop", learned, obs, command)
                        self._remember_pico_targets(command)
                        self._selected.step(obs, command)
                        self._remember_pico_targets(command)
                        return
                    except Exception as exc:  # noqa: BLE001 - learned child only
                        self._fallback_from_exception(
                            "handoff",
                            exc,
                            obs,
                            command,
                            step_now=True,
                        )
                        return
            # A fault/tracker fallback remains latched despite recovery until release.
            self._selected.step(obs, command)
            return

        try:
            self._selected.step(obs, command)
            self._remember_pico_targets(command)
        except Exception as exc:  # noqa: BLE001 - learned child only
            self._fallback_from_exception(
                "inference",
                exc,
                obs,
                command,
                step_now=True,
            )

    # ------------------------------------------------------------ whole-body mode
    def _standing_child(self) -> tuple[str, Move]:
        """The policy that stands the robot in whole-body mode: PICO, else the hold."""
        pico = self._children.get("pico_teleop")
        return ("pico_teleop", pico) if pico is not None else ("walk", self._children["walk"])

    def _wholebody_start_refusal(self) -> str | None:
        child = self._children.get("wholebody")
        if child is None:
            return self.wholebody_unavailable_reason or "whole-body policy unavailable"
        return child.start_refusal()

    def _wholebody_wanted(self) -> bool:
        """The PC started an activation that may be tracked now."""
        return self._wholebody_start_refusal() is None

    def _count_standing(self, obs: Observation) -> None:
        """Count the ticks the standing policy has kept the robot still.

        Still: the left trigger released (no velocity or foot target reaches
        the policy), the trunk within WHOLEBODY_START_TILT_DEG of the HOME
        attitude and no leg joint moving at WHOLEBODY_START_LEG_SPEED_RAD_S
        (not stepping).  Any other tick, and every tick of whole-body
        tracking, starts the count again.
        """
        tilt = tilt_from_home_rad(obs.robot_state.projected_gravity)
        speeds = [obs.robot_state.motor_velocities.get(name) for name in LEG_JOINTS]
        still = (
            self._selected_name not in (None, "wholebody")
            and bool(obs.user_input.balance_only)
            and tilt is not None
            and tilt < math.radians(WHOLEBODY_START_TILT_DEG)
            and all(
                speed is not None and abs(float(speed)) < WHOLEBODY_START_LEG_SPEED_RAD_S
                for speed in speeds
            )
        )
        self._standing_ticks = self._standing_ticks + 1 if still else 0

    def _wholebody_ready(self) -> bool:
        """The robot has stood still long enough for whole-body tracking to start."""
        return self._standing_ticks >= WHOLEBODY_START_TICKS

    def _remember_learned_targets(self, command: MotorCommand) -> None:
        try:
            targets = {name: float(command.target_angles[name]) for name in OBSERVATION_DOF_ORDER}
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if all(math.isfinite(value) for value in targets.values()):
            self._last_learned_targets = targets

    def _start_standing(self, obs: Observation, command: MotorCommand, *, step_now: bool) -> None:
        name, child = self._standing_child()
        if isinstance(child, _HoldPositionMove):
            child.seed_targets(self._last_learned_targets)
        try:
            self._start_child(name, child, obs, command)
            if step_now:
                child.step(obs, command)
        except Exception as exc:  # noqa: BLE001 - learned child only
            if name == "walk":
                raise
            self._fallback_from_exception("start", exc, obs, command, step_now=step_now)

    def _try_start_wholebody(self, obs: Observation, command: MotorCommand) -> bool:
        """Start whole-body tracking; on refusal record why and return False.

        An activation that may run waits, unrefused, until the robot stands
        still (_wholebody_ready).
        """
        refusal = self._wholebody_start_refusal()
        if refusal is None and not self._wholebody_ready():
            return False
        if refusal is None:
            child = self._children["wholebody"]
            try:
                self._start_child("wholebody", child, obs, command)
                self._remember_learned_targets(command)
                return True
            except Exception as exc:  # noqa: BLE001 - learned child only
                refusal = self._describe_failure("whole-body start", exc)
                child.refuse(refusal)
        self._fallback_reason = refusal
        return False

    def _step_wholebody_mode(self, obs: Observation, command: MotorCommand, requested: str) -> None:
        if self._selected_name == "wholebody":
            child = self._selected
            if requested not in ("wholebody", "pico_teleop"):
                child.request_stop(f"{requested} requested", float(obs.robot_state.time_s))
            try:
                child.step(obs, command)
            except Exception as exc:  # noqa: BLE001 - learned child only
                reason = self._describe_failure("whole-body inference", exc)
                child.refuse(reason)
                self._fallback_reason = reason
                print(f"Whole-body tracking failed: {reason}; standing", flush=True)
                self._start_standing(obs, command, step_now=True)
                return
            self._remember_learned_targets(command)
            if child.handoff_ready:
                # Back at HOME: the standing policy takes the robot.
                self._fallback_reason = child.stop_reason
                self._start_standing(obs, command, step_now=False)
            return
        if self._try_start_wholebody(obs, command):
            return
        if self._selected is not self._standing_child()[1]:
            self._start_standing(obs, command, step_now=True)
            return
        try:
            self._selected.step(obs, command)
        except Exception as exc:  # noqa: BLE001 - learned child only
            if self._selected_name == "hold":
                raise
            self._fallback_from_exception("inference", exc, obs, command, step_now=True)

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        self._poll_reload()
        self._notice_replacement()
        if self._selected is None:
            self.state = MoveState.INACTIVE
            self._fallback_latched = False
            self._last_pico_targets = None
            self._last_balance_only = False
            return
        if self._selected.state != MoveState.STOPPING:
            self._selected.state = MoveState.STOPPING
        self._selected.on_stop(obs, command)
        self._sync_stop_state()

    def _sync_stop_state(self) -> None:
        if self._selected is not None and self._selected.state == MoveState.INACTIVE:
            self._claim.active = False
            self._selected = None
            self._selected_name = None
            self._fallback_latched = False
            self._last_pico_targets = None
            self._last_balance_only = False
            self._standing_ticks = 0
            self.state = MoveState.INACTIVE
        else:
            self.state = MoveState.STOPPING

    def on_safety_resume(self, obs: Observation) -> None:
        self._last_pico_targets = None
        self._standing_ticks = 0
        if self._selected is not None:
            self._selected.on_safety_resume(obs)
