# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Safe PICO-button selection between the legacy and hybrid locomotion policies."""

from __future__ import annotations

from pathlib import Path

from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from moves.pico_hybrid import AGENT_NAME as PICO_AGENT_NAME, PicoHybridMove
from moves.walk import WalkMove
from observer import Observation


POLICY_NAMES = frozenset({"walk", "pico_teleop"})


class PolicySelectableWalkMove(Move):
    """Expose one scheduler move while keeping low-level policy ownership exclusive.

    The PICO's left X button changes ``UserInput.locomotion_policy`` only while
    the trigger is released.  Consequently the current child first completes its
    smooth return to neutral.  A subsequent trigger press starts exactly one child.
    This wrapper still detects an unexpected in-motion mode change and initiates
    the same stop path rather than allowing two policies to write the 18 joints.
    """

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        pico_policy_path: str | Path = Path("src/agents") / PICO_AGENT_NAME,
        *,
        legacy_move: Move | None = None,
        pico_move: Move | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._pico_policy_path = Path(pico_policy_path)
        self._children: dict[str, Move] = {
            "walk": legacy_move or WalkMove(controller=controller),
        }
        if pico_move is not None:
            self._children["pico_teleop"] = pico_move
        self._selected_name: str | None = None
        self._selected: Move | None = None

    def preload(self) -> None:
        # Loading ONNX can take longer than one 20 ms control tick.  Do it before
        # Scheduler.run(), but keep ordinary walking usable on installations that
        # have not yet copied a validated PICO policy artifact.
        if "pico_teleop" not in self._children and self._pico_policy_path.is_file():
            self._children["pico_teleop"] = PicoHybridMove(
                controller=self._controller,
                policy_path=self._pico_policy_path,
            )
        for child in self._children.values():
            child.preload()

    @staticmethod
    def _requested_policy(obs: Observation) -> str:
        name = obs.user_input.locomotion_policy
        if name not in POLICY_NAMES:
            raise RuntimeError(f"unsupported locomotion policy: {name!r}")
        return name

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        requested = self._requested_policy(obs)
        child = self._children.get(requested)
        if child is None:
            raise RuntimeError(
                f"PICO requested {requested!r}, but validated policy "
                f"{self._pico_policy_path} is not installed"
            )
        self._selected_name = requested
        self._selected = child
        child.state = MoveState.STARTING
        child.on_start(obs, command)
        if child.state != MoveState.ACTIVE:
            raise RuntimeError(f"locomotion child {requested!r} failed to start")
        self.state = MoveState.ACTIVE

    def step(self, obs: Observation, command: MotorCommand) -> None:
        if self._selected is None or self._selected_name is None:
            raise RuntimeError("locomotion selector is active without a child")
        if self._requested_policy(obs) != self._selected_name:
            # This should already be prevented by NetworkInputSource.  Stop anyway
            # so a future input implementation cannot hot-swap joint owners.
            self.state = MoveState.STOPPING
            self._selected.state = MoveState.STOPPING
            self._selected.on_stop(obs, command)
            self._sync_stop_state()
            return
        self._selected.step(obs, command)

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        if self._selected is None:
            self.state = MoveState.INACTIVE
            return
        if self._selected.state != MoveState.STOPPING:
            self._selected.state = MoveState.STOPPING
        self._selected.on_stop(obs, command)
        self._sync_stop_state()

    def _sync_stop_state(self) -> None:
        if self._selected is not None and self._selected.state == MoveState.INACTIVE:
            self._selected = None
            self._selected_name = None
            self.state = MoveState.INACTIVE
        else:
            self.state = MoveState.STOPPING

    def on_safety_resume(self, obs: Observation) -> None:
        if self._selected is not None:
            self._selected.on_safety_resume(obs)
