# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Select PICO locomotion and hold position if its actor is unavailable."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path

from constants import KP_DEFAULT, KP_HARDWARE_NEUTRAL, MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from moves.pico_hybrid import AGENT_NAME as PICO_AGENT_NAME
from moves.pico_hybrid import PicoHybridMove
from observer import Observation

POLICY_NAMES = frozenset({"walk", "pico_teleop"})

# A later contract can inject another parser/Move without duplicating fallback logic.
LearnedMoveFactory = Callable[[ControllerProtocol | None, Path], Move]
PolicyFingerprint = tuple[int, int, int, int]


def _default_learned_move_factory(
    controller: ControllerProtocol | None,
    policy_path: Path,
) -> Move:
    return PicoHybridMove(controller=controller, policy_path=policy_path)


class _HoldPositionMove(Move):
    """Keep the last complete body goal without running the old walk actor."""

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
        # Set when this hold takes over from GetupMove, which leaves the 18
        # policy joints at the soft learned-policy gain (KP_RL). A static hold
        # at that gain cannot carry the robot and slowly topples it.
        self._restore_gain_pending = False

    def arm_gain_restore_after_getup(self) -> None:
        self._restore_gain_pending = True

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
        command.target_angles.update(self._held_targets)
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        _ = obs
        if self._restore_gain_pending:
            # One tick after the handoff: GetupMove.on_stop has already written
            # its KP_RL gain and the held (measured) goals have been sent. Re-seed
            # the goals first, then raise the gain, so P900 never acts on an old
            # saturated get-up target.
            self._restore_gain_pending = False
            if self._controller is not None:
                ids = [MOTOR_TO_ID[name] for name in OBSERVATION_DOF_ORDER]
                self._controller.sync_write_goal_position(
                    ids, [self._held_targets[name] for name in OBSERVATION_DOF_ORDER]
                )
                self._controller.sync_write_kp(ids, [KP_HARDWARE_NEUTRAL] * len(ids))
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
                ids = [MOTOR_TO_ID[name] for name in OBSERVATION_DOF_ORDER]
                self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
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
        legacy_move: Move | None = None,
        pico_move: Move | None = None,
        learned_move_factory: LearnedMoveFactory | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._pico_policy_path = Path(pico_policy_path)
        self._learned_move_factory = (
            learned_move_factory or _default_learned_move_factory
        )
        self._children: dict[str, Move] = {
            "walk": legacy_move or _HoldPositionMove(controller=controller),
        }
        if pico_move is not None:
            self._children["pico_teleop"] = pico_move

        self._selected_name: str | None = None
        self._selected: Move | None = None
        self._fallback_latched = False
        self._fallback_reason: str | None = None
        self._last_pico_targets: dict[str, float] | None = None
        self._last_balance_only = False
        self._started_from_getup = False

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

    def preload(self) -> None:
        # The fixed-position hold is always available. An injected legacy move
        # retains its existing preload behavior for custom clients and tests.
        self._children["walk"].preload()

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

    def _start_legacy(
        self,
        obs: Observation,
        command: MotorCommand,
        *,
        step_now: bool,
    ) -> None:
        previous = self._selected
        legacy = self._children["walk"]
        if isinstance(legacy, _HoldPositionMove):
            legacy.seed_targets(
                self._last_pico_targets
                if self._selected_name == "pico_teleop"
                else None
            )
        if previous is not None and previous is not legacy:
            previous.state = MoveState.INACTIVE
        self._start_child("walk", legacy, obs, command)
        if step_now:
            legacy.step(obs, command)

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
        # Parsing may finish during this legacy activation; adoption waits for the
        # next trigger release/press boundary.
        if self._policy_fingerprint() is not None:
            self.request_learned_reload(force=True)
        # A missing file must not overwrite the more useful runtime failure above.
        self._fallback_reason = reason
        self._start_legacy(obs, command, step_now=step_now)

    def can_balance(self, user_input) -> bool:
        """Whether on_start for ``user_input`` would select a balancing child.

        Mirrors on_start's choice: a loaded learned policy when it is
        requested and its tracking is not degraded, otherwise the legacy
        child (the static _HoldPositionMove on the PICO runtime). A finished
        background reload is adopted first, exactly as on_start would.
        """
        requested = user_input.locomotion_policy
        if requested not in POLICY_NAMES:
            return False
        self._poll_reload()
        if requested == "pico_teleop" and not user_input.learned_policy_degraded:
            learned = self._children.get("pico_teleop")
            if learned is not None:
                return learned.can_balance(user_input)
        return self._children["walk"].can_balance(user_input)

    def seed_next_start_from_getup(self) -> None:
        """Called by the scheduler every tick while get-up owns the robot."""
        self._started_from_getup = True
        legacy = self._children["walk"]
        seed = getattr(legacy, "seed_next_start_from_getup", None)
        if callable(seed):
            seed()

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        from_getup = self._started_from_getup
        self._started_from_getup = False
        self._poll_reload()
        self._notice_replacement()
        self._fallback_latched = False
        self._last_pico_targets = None
        self._last_balance_only = bool(obs.user_input.balance_only)
        requested = self._requested_policy(obs)
        child = self._children.get(requested)

        if requested == "pico_teleop" and child is None:
            reason = self._fallback_reason or (
                f"learned policy is unavailable at {self._pico_policy_path}"
            )
            self._record_fallback(reason)
            self._start_legacy(obs, command, step_now=False)
            self._arm_hold_gain_restore(from_getup)
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
        self._arm_hold_gain_restore(from_getup)

    def _arm_hold_gain_restore(self, from_getup: bool) -> None:
        if from_getup and isinstance(self._selected, _HoldPositionMove):
            self._selected.arm_gain_restore_after_getup()

    def step(self, obs: Observation, command: MotorCommand) -> None:
        self._poll_reload()
        if self._selected is None or self._selected_name is None:
            raise RuntimeError("locomotion selector is active without a child")

        balance_only = bool(obs.user_input.balance_only)
        if balance_only and not self._last_balance_only:
            # A trigger release permits one retry, not one retry per balance tick.
            self._fallback_latched = False
        self._last_balance_only = balance_only

        requested = self._requested_policy(obs)
        if self._selected_name == "pico_teleop" and requested != "pico_teleop":
            if obs.user_input.learned_policy_degraded:
                # Optional tracking failed. Retain trigger/stick input and latch the
                # baseline so tracking recovery cannot switch policies mid-stride.
                self._record_fallback("body tracking degraded")
            else:
                # Preserve support for an explicit wire-policy change from a
                # legacy/custom client.  The current PICO bridge never takes
                # this branch: it always requests pico_teleop and uses only the
                # left-trigger active_moves state as its momentary enable.
                self._fallback_reason = "learned policy deselected on the wire"
                self._fallback_latched = False
            self._start_legacy(obs, command, step_now=True)
            return

        if (
            self._selected_name == "pico_teleop"
            and obs.user_input.learned_policy_degraded
        ):
            # Optional tracking failed. Retain trigger/stick input and immediately
            # use the baseline; do not switch back during this activation.
            self._record_fallback("body tracking degraded")
            self._start_legacy(obs, command, step_now=True)
            return

        if self._selected_name in ("walk", "hold"):
            if obs.user_input.learned_policy_degraded and not self._fallback_latched:
                # The activation may already be on legacy when a learned-policy
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
            self._selected = None
            self._selected_name = None
            self._fallback_latched = False
            self._last_pico_targets = None
            self._last_balance_only = False
            self.state = MoveState.INACTIVE
        else:
            self.state = MoveState.STOPPING

    def on_safety_resume(self, obs: Observation) -> None:
        self._last_pico_targets = None
        if self._selected is not None:
            self._selected.on_safety_resume(obs)
