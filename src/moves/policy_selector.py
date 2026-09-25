# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Resilient PICO selection with the proven walk policy as a hot fallback."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from moves.pico_hybrid import AGENT_NAME as PICO_AGENT_NAME
from moves.pico_hybrid import PicoHybridMove
from moves.walk import WalkMove
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


class PolicySelectableWalkMove(Move):
    """Contain optional learned-policy faults behind the proven walk policy.

    A missing/rejected model or an exception from learned-policy start/inference
    falls back to ``walk`` in the same scheduler cycle using the same trigger and
    joystick observation. Once degraded, an activation stays on ``walk`` until the
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
            "walk": legacy_move or WalkMove(controller=controller),
        }
        if pico_move is not None:
            self._children["pico_teleop"] = pico_move

        self._selected_name: str | None = None
        self._selected: Move | None = None
        self._fallback_latched = False
        self._fallback_reason: str | None = None

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
        print(f"PICO teleop fallback: {reason}; continuing with walk", flush=True)

    def _build_learned_move(self) -> Move:
        child = self._learned_move_factory(
            self._controller,
            self._pico_policy_path,
        )
        child.preload()
        return child

    def preload(self) -> None:
        # The baseline is mandatory. Its failure cannot be hidden because no
        # locomotion fallback would remain.
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
        self._selected_name = name
        self._selected = child
        self.state = MoveState.ACTIVE

    def _start_legacy(
        self,
        obs: Observation,
        command: MotorCommand,
        *,
        step_now: bool,
    ) -> None:
        previous = self._selected
        legacy = self._children["walk"]
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

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        self._poll_reload()
        self._notice_replacement()
        self._fallback_latched = False
        requested = self._requested_policy(obs)
        child = self._children.get(requested)

        if requested == "pico_teleop" and child is None:
            reason = self._fallback_reason or (
                f"learned policy is unavailable at {self._pico_policy_path}"
            )
            self._record_fallback(reason)
            self._start_legacy(obs, command, step_now=False)
            return

        assert child is not None
        try:
            self._start_child(requested, child, obs, command)
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

        if self._selected_name == "walk":
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
                        self._start_child("pico_teleop", learned, obs, command)
                        self._selected.step(obs, command)
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
            self.state = MoveState.INACTIVE
        else:
            self.state = MoveState.STOPPING

    def on_safety_resume(self, obs: Observation) -> None:
        if self._selected is not None:
            self._selected.on_safety_resume(obs)
