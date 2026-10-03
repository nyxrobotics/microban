"""Post-get-up standing balance: Scheduler keeps GetupMove on until a walk
move that can balance is requested (see Scheduler._walk_can_take_over)."""

import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scheduler as scheduler_module
from constants import MOTOR_TO_ID, NEUTRAL_POSE
from input.input_source import UserInput
from moves.move import Move, MoveState
from moves.policy_selector import PolicySelectableWalkMove, _HoldPositionMove
from observer import RobotState
from scheduler import GETUP_AUTO_TIMEOUT_S, Scheduler

HZ = 50.0
UPRIGHT = [0.0, 0.0, -1.0]
FALLEN = [1.0, 0.0, 0.0]


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def perf_counter(self):
        return self.t

    def sleep(self, seconds):
        if seconds > 0:
            self.t += seconds


class FakeController:
    def __init__(self):
        self.imu_valid = True
        self.kp_writes = []
        self.torque_writes = []

    def get_imu_status(self):
        return {"valid": self.imu_valid, "age_s": 0.0}

    def sync_write_goal_position(self, ids, positions):
        pass

    def sync_write_kp(self, ids, gains):
        self.kp_writes.append((list(ids), list(gains)))

    def sync_write_torque_enable(self, ids, values):
        self.torque_writes.append((list(ids), list(values)))

    def shutdown(self):
        pass


class FakeObserver:
    last_position_ms = 0.0
    last_velocity_ms = 0.0

    def __init__(self):
        self.gravity = list(UPRIGHT)

    def read_state(self, _dt):
        return RobotState(
            gyro=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
            body_quat=[1.0, 0.0, 0.0, 0.0],
            projected_gravity=list(self.gravity),
            motor_positions=dict(NEUTRAL_POSE),
            motor_velocities={name: 0.0 for name in MOTOR_TO_ID},
        )


class FakeGetup(Move):
    def __init__(self):
        super().__init__()
        self.model_ready = True
        self.policy_faulted = False
        self.armed_steps = 0
        self.starts = 0
        self.velocities = []

    def on_start(self, obs, command):
        self.starts += 1
        super().on_start(obs, command)

    def on_safety_resume(self, _obs):
        # Same contract as GetupMove: restart through on_start.
        if self.state != MoveState.INACTIVE:
            self.state = MoveState.STARTING

    def step(self, obs, _command):
        self.velocities.append(dict(obs.user_input.velocity))
        if obs.user_input.getup_armed:
            self.armed_steps += 1


class FakeWalk(Move):
    def __init__(self, balances):
        super().__init__()
        self.balances = balances
        self.steps = 0

    def can_balance(self, user_input):
        return self.balances

    def step(self, _obs, _command):
        self.steps += 1


class CountingMove(Move):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def on_start(self, obs, command):
        self.calls += 1
        super().on_start(obs, command)

    def step(self, _obs, _command):
        self.calls += 1


class ScriptedInput:
    """``script(tick, inhibited) -> (gravity, UserInput)``; ``hook(tick)``
    runs before each read with the scheduler state left by the previous tick."""

    def __init__(self, harness, script, ticks):
        self.h = harness
        self.script = script
        self.ticks = ticks
        self.tick = 0
        self.inhibited = False
        self.inhibits = []

    def start(self):
        pass

    def stop(self):
        pass

    def set_motion_inhibited(self, inhibited):
        self.inhibited = inhibited
        self.inhibits.append(inhibited)

    def read(self):
        tick = self.tick
        self.tick += 1
        self.h.snapshot(tick)
        gravity, user_input = self.script(tick, self.inhibited)
        self.h.observer.gravity = gravity
        if tick + 1 >= self.ticks:
            self.h.scheduler.stop_flag_path.write_text("stop\n")
        return user_input


class Harness:
    def __init__(self, walk, *, power=False, imu_shutdown_after_s=0.75, extra_moves=None):
        self.controller = FakeController()
        self.observer = FakeObserver()
        self.getup = FakeGetup()
        self.walk = walk
        self.rows = []
        self.on_tick = None
        self._tmp = tempfile.TemporaryDirectory()
        moves = {"getup": self.getup} if walk is None else {"walk": walk, "getup": self.getup}
        moves = {**(extra_moves or {}), **moves}
        self.scheduler = Scheduler(
            frequency_hz=HZ,
            controller=self.controller,
            stop_flag_path=str(Path(self._tmp.name) / "stop"),
            moves=moves,
            hardware_power_control=power,
            imu_shutdown_after_s=imu_shutdown_after_s,
        )
        self.scheduler.observer = self.observer
        self.output = ""

    def snapshot(self, tick):
        if self.on_tick is not None:
            self.on_tick(tick)
        s = self.scheduler
        self.rows.append(dict(
            tick=tick,
            override=s._getup_active_override,
            balancing=s._getup_balancing,
            failed=s._getup_auto_failed,
            started=s._getup_auto_started_s,
            getup=self.getup.state,
            walk=None if self.walk is None else self.walk.state,
            walk_steps=None if self.walk is None else getattr(self.walk, "steps", None),
            armed_steps=self.getup.armed_steps,
        ))

    def run(self, script, ticks):
        source = ScriptedInput(self, script, ticks)
        self.scheduler.input_source = source
        self.source = source
        clock = FakeClock()
        out = io.StringIO()
        with mock.patch.object(scheduler_module, "time", clock), contextlib.redirect_stdout(out):
            self.scheduler.loop_start_time = clock.t
            self.scheduler.run()
        self.output = out.getvalue()
        self._tmp.cleanup()
        return self.rows


def sec(seconds):
    return int(round(seconds * HZ))


def fall_then_stand(fall_ticks=sec(1.0), *, moves=frozenset({"walk"}), velocity=None,
                    after=None):
    """Upright 0.2 s, fallen ``fall_ticks``, then upright."""

    def script(tick, _inhibited):
        start = sec(0.2)
        gravity = FALLEN if start <= tick < start + fall_ticks else UPRIGHT
        if after is not None:
            override = after(tick)
            if override is not None:
                return override
        return gravity, UserInput(
            active_moves=set(moves),
            velocity=dict(velocity or {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}),
        )

    return script


STAND_TICK = sec(0.2) + sec(1.0) + 20  # first tick the stand debounce is met


class StandingBalanceTest(unittest.TestCase):
    def test_keeps_balancing_after_stand_when_walk_cannot_balance(self):
        h = Harness(FakeWalk(balances=False))
        rows = h.run(fall_then_stand(), sec(6.0))
        last = rows[-1]
        self.assertTrue(last["override"])
        self.assertTrue(last["balancing"])
        self.assertEqual(last["getup"], MoveState.ACTIVE)
        self.assertEqual(h.walk.steps, rows[STAND_TICK]["walk_steps"])
        # The actor keeps running armed while balancing.
        self.assertGreater(last["armed_steps"] - rows[STAND_TICK + 2]["armed_steps"], sec(3.0))
        # Inhibited while getting up, released while balancing.
        self.assertTrue(any(h.source.inhibits))
        self.assertFalse(h.source.inhibits[-1])
        self.assertIn("keeps balancing", h.output)

    def test_keeps_balancing_when_no_walk_is_requested(self):
        h = Harness(FakeWalk(balances=True))
        rows = h.run(fall_then_stand(moves=frozenset()), sec(4.0))
        self.assertTrue(rows[-1]["balancing"])
        self.assertEqual(rows[-1]["getup"], MoveState.ACTIVE)

    def test_keeps_balancing_without_any_walk_move(self):
        h = Harness(None)
        rows = h.run(fall_then_stand(), sec(4.0))
        self.assertTrue(rows[-1]["balancing"])

    def test_hands_back_to_requested_balancing_walk(self):
        h = Harness(FakeWalk(balances=True))
        rows = h.run(fall_then_stand(), sec(4.0))
        self.assertTrue(any(row["override"] for row in rows))
        self.assertFalse(any(row["balancing"] for row in rows))
        last = rows[-1]
        self.assertFalse(last["override"])
        self.assertEqual(last["walk"], MoveState.ACTIVE)
        self.assertEqual(last["getup"], MoveState.INACTIVE)
        self.assertGreater(h.walk.steps, 0)

    def test_source_that_strips_walk_while_inhibited_still_hands_back(self):
        # GC300/PICO report no "walk" while inhibited: balancing lifts the
        # inhibit, the next snapshot shows the request and walk takes over.
        h = Harness(FakeWalk(balances=True))

        def script(tick, inhibited):
            gravity, user_input = fall_then_stand()(tick, inhibited)
            if inhibited:
                user_input.active_moves = set()
            return gravity, user_input

        rows = h.run(script, sec(4.0))
        balancing_ticks = sum(row["balancing"] for row in rows)
        self.assertGreaterEqual(balancing_ticks, 1)
        self.assertLessEqual(balancing_ticks, 2)
        self.assertFalse(rows[-1]["override"])
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)
        self.assertIn("balancing walk requested", h.output)

    def test_hand_back_waits_for_zero_velocity_command(self):
        h = Harness(FakeWalk(balances=True))
        moving_until = STAND_TICK + sec(1.0)

        def after(tick):
            if tick < moving_until:
                return None
            return UPRIGHT, UserInput(active_moves={"walk"})

        rows = h.run(
            fall_then_stand(velocity={"vx": 0.5, "vy": 0.0, "vtheta": 0.0}, after=after),
            sec(4.0),
        )
        self.assertTrue(rows[moving_until]["balancing"])
        self.assertEqual(
            rows[moving_until]["walk_steps"], rows[STAND_TICK]["walk_steps"]
        )
        self.assertFalse(rows[-1]["override"])
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)

    def test_time_limit_does_not_count_while_balancing(self):
        h = Harness(FakeWalk(balances=False))
        rows = h.run(fall_then_stand(), sec(GETUP_AUTO_TIMEOUT_S + 5.0))
        last = rows[-1]
        self.assertFalse(last["failed"])
        self.assertTrue(last["balancing"])
        self.assertIsNone(last["started"])
        self.assertNotIn("time limit", h.output)

    def test_time_limit_still_bounds_a_get_up_that_never_stands(self):
        h = Harness(FakeWalk(balances=False))
        rows = h.run(fall_then_stand(fall_ticks=10**6), sec(GETUP_AUTO_TIMEOUT_S + 2.0))
        self.assertTrue(rows[-1]["failed"])
        self.assertFalse(any(row["balancing"] for row in rows))

    def test_new_fall_while_balancing_starts_a_fresh_attempt(self):
        h = Harness(FakeWalk(balances=False))
        second_fall = sec(GETUP_AUTO_TIMEOUT_S + 2.0)

        def after(tick):
            if tick >= second_fall:
                return FALLEN, UserInput(active_moves={"walk"})
            return None

        rows = h.run(
            fall_then_stand(after=after),
            second_fall + sec(GETUP_AUTO_TIMEOUT_S + 1.0),
        )
        self.assertTrue(rows[second_fall]["balancing"])
        new_attempt = next(
            row for row in rows[second_fall:] if not row["balancing"]
        )
        # Re-entered get-up after the normal fall debounce, with a new timer.
        self.assertLessEqual(new_attempt["tick"] - second_fall, 16)
        self.assertTrue(new_attempt["override"])
        attempt_rows = [row for row in rows[second_fall:] if row["started"] is not None]
        self.assertTrue(attempt_rows)
        started = attempt_rows[0]["started"]
        self.assertGreater(started, 100.0 + second_fall / HZ)
        # Not failed one second before the fresh limit, failed after it.
        before_limit = attempt_rows[0]["tick"] + sec(GETUP_AUTO_TIMEOUT_S - 1.0)
        self.assertFalse(rows[before_limit]["failed"])
        self.assertTrue(rows[-1]["failed"])
        self.assertEqual(h.output.count("Automatic get-up attempt started"), 2)
        self.assertIn("Fall detected while balancing", h.output)

    def run_imu_glitch(self, glitch_ticks, *, power=False):
        h = Harness(FakeWalk(balances=False), power=power, imu_shutdown_after_s=5.0)
        unsafe_from = STAND_TICK + sec(1.0)
        unsafe_to = unsafe_from + glitch_ticks

        def on_tick(tick):
            h.controller.imu_valid = not (unsafe_from <= tick < unsafe_to)

        def script(tick, inhibited):
            gravity, user_input = fall_then_stand()(tick, inhibited)
            if power:
                user_input.torque_enabled = True
                user_input.policy_enabled = tick >= 3  # A, then R3 on
            return gravity, user_input

        h.on_tick = on_tick
        rows = h.run(script, unsafe_to + sec(2.0))
        return h, rows, unsafe_from, unsafe_to

    def assert_balance_survives_imu_glitch(self, h, rows, unsafe_from, unsafe_to):
        self.assertTrue(rows[unsafe_from]["balancing"])
        # Balancing state is kept through the outage (safety hold freezes goals).
        for row in rows[unsafe_from:]:
            self.assertTrue(row["override"])
            self.assertTrue(row["balancing"])
            self.assertIsNone(row["started"])
        self.assertNotIn("released", h.output)
        self.assertIn("unsafe IMU", h.output)
        # Get-up restarts through on_safety_resume -> on_start and keeps
        # balancing armed; the non-balancing walk never takes the legs.
        self.assertEqual(h.getup.starts, 2)
        self.assertEqual(rows[-1]["getup"], MoveState.ACTIVE)
        self.assertGreater(rows[-1]["armed_steps"], rows[unsafe_to + 2]["armed_steps"])
        self.assertEqual(h.walk.steps, rows[STAND_TICK]["walk_steps"])
        self.assertFalse(rows[-1]["failed"])

    def test_one_tick_imu_glitch_keeps_balancing(self):
        self.assert_balance_survives_imu_glitch(*self.run_imu_glitch(1))

    def test_imu_outage_keeps_balancing_with_power_gate(self):
        self.assert_balance_survives_imu_glitch(*self.run_imu_glitch(sec(0.5), power=True))

    def test_actor_fault_releases_balance_and_latches_fault_hold(self):
        h = Harness(FakeWalk(balances=False))
        fault_at = STAND_TICK + sec(1.0)

        def on_tick(tick):
            if tick == fault_at:
                h.getup.policy_faulted = True

        h.on_tick = on_tick
        rows = h.run(fall_then_stand(), fault_at + sec(1.0))
        self.assertTrue(rows[fault_at]["balancing"])
        self.assertTrue(rows[-1]["failed"])
        self.assertFalse(rows[-1]["balancing"])
        self.assertFalse(rows[-1]["override"])
        # Latched fault hold: no dispatch, walk never resumes.
        self.assertEqual(h.walk.steps, rows[STAND_TICK]["walk_steps"])
        self.assertIn("released: get-up actor unavailable", h.output)


class OvercurrentCutoffTest(unittest.TestCase):
    def test_getup_cutoff_only_while_getting_up(self):
        h = Harness(FakeWalk(balances=False))
        cutoffs = []
        check = h.scheduler._check_overcurrent

        def spy(state, targets, cutoff):
            cutoffs.append((len(h.rows) - 1, cutoff))
            return check(state, targets, cutoff)

        h.scheduler._check_overcurrent = spy
        second_fall = STAND_TICK + sec(3.0)

        def after(tick):
            if second_fall <= tick < second_fall + sec(1.0):
                return FALLEN, UserInput(active_moves={"walk"})
            return None

        rows = h.run(fall_then_stand(after=after), second_fall + sec(4.0))
        by_tick = dict(cutoffs)
        getting_up = [
            by_tick[row["tick"]] for row in rows
            if row["override"] and not row["balancing"] and row["tick"] in by_tick
        ]
        balancing = [
            by_tick[row["tick"]] for row in rows
            if row["balancing"] and row["tick"] in by_tick
        ]
        self.assertTrue(getting_up and balancing)
        # rows[t + 1] holds the state left by tick t, which chose that tick's
        # cutoff. Balancing runs under the normal limit once the attempt's
        # transient balancing goals have left the proxy's delay history.
        tail = h.scheduler._cmd_history.maxlen
        settled = {
            c for t, c in cutoffs
            if t >= 2 * tail and t + 1 < len(rows)
            and all(rows[j]["balancing"] for j in range(t - 2 * tail + 1, t + 2))
        }
        self.assertEqual(settled, {scheduler_module.OVERCURRENT_CUTOFF_A})
        # The first `tail` balancing ticks still write the get-up actor's
        # transient goals and the next `tail` still pair them in the history,
        # so the get-up limit holds for exactly 2 * tail balancing ticks.
        first_balancing = next(t for t, _ in cutoffs if rows[t + 1]["balancing"])
        self.assertEqual(
            [by_tick[t] for t in range(first_balancing, first_balancing + 2 * tail + 1)],
            [scheduler_module.OVERCURRENT_CUTOFF_A_GETUP] * (2 * tail)
            + [scheduler_module.OVERCURRENT_CUTOFF_A],
        )
        self.assertIn(scheduler_module.OVERCURRENT_CUTOFF_A_GETUP,
                      {c for t, c in cutoffs if t > second_fall + 16})
        # Both get-up attempts ran under the get-up limit.
        attempts = [t for t, c in cutoffs if c == scheduler_module.OVERCURRENT_CUTOFF_A_GETUP]
        self.assertTrue(any(t < STAND_TICK for t in attempts))
        self.assertTrue(any(t > second_fall for t in attempts))

    def test_fall_debounce_hold_uses_getup_cutoff(self):
        # The 15-tick fall debounce is part of the fall transient: the
        # delay-aligned current proxy still pairs the walk actor's falling
        # goals with the snapped hold (14.7 A estimated in the runtime sim,
        # against the normal 15 A limit).
        h = Harness(FakeWalk(balances=False))
        cutoffs = {}
        check = h.scheduler._check_overcurrent

        def spy(state, targets, cutoff):
            cutoffs[len(h.rows) - 1] = cutoff
            return check(state, targets, cutoff)

        h.scheduler._check_overcurrent = spy
        h.run(fall_then_stand(), STAND_TICK + sec(1.0))
        fall_start = sec(0.2)
        before_fall = {c for t, c in cutoffs.items() if t < fall_start}
        debounce = {
            c for t, c in cutoffs.items()
            if fall_start + 1 <= t < fall_start + h.scheduler._fall_debounce_ticks
        }
        self.assertEqual(before_fall, {scheduler_module.OVERCURRENT_CUTOFF_A})
        self.assertEqual(debounce, {scheduler_module.OVERCURRENT_CUTOFF_A_GETUP})


class SaturatingGetup(FakeGetup):
    """Get-up actor whose goals sit at the servo range, as the unclipped
    actor's do while it stands up: ~0.91 A per joint in the current proxy."""

    def step(self, obs, command):
        super().step(obs, command)
        for name in MOTOR_TO_ID:
            command.target_angles[name] = NEUTRAL_POSE[name] + 3.14159


class SaturatingWalk(FakeWalk):
    """Balancing walk that holds the measured pose (no proxy current) until
    ``saturate`` is set, then commands the servo range on every joint."""

    def __init__(self):
        super().__init__(balances=True)
        self.saturate = False

    def step(self, obs, command):
        super().step(obs, command)
        if self.saturate:
            for name in MOTOR_TO_ID:
                command.target_angles[name] = NEUTRAL_POSE[name] - 3.14159


class HandBackOvercurrentTest(unittest.TestCase):
    """After get-up hands back, the delay-aligned current proxy still pairs
    the get-up's last (saturated) goals with the feedback for the length of
    its command history; the get-up limit must cover exactly that window."""

    def run_hand_back(self, saturate_walk_after=None, ticks=STAND_TICK + sec(2.0)):
        h = Harness(SaturatingWalk())
        h.getup = SaturatingGetup()
        h.scheduler.registered_moves["getup"] = h.getup
        checks = []
        check = h.scheduler._check_overcurrent
        hand_back = []
        got_up = []

        def spy(state, targets, cutoff):
            tick = len(h.rows) - 1
            if h.getup.state == MoveState.ACTIVE:
                got_up.append(tick)
            elif got_up and not hand_back and h.walk.state == MoveState.ACTIVE:
                hand_back.append(tick)
            if (
                saturate_walk_after is not None and hand_back
                and tick >= hand_back[0] + saturate_walk_after
            ):
                h.walk.saturate = True
            checks.append(dict(
                tick=tick,
                cutoff=cutoff,
                current=h.scheduler._estimate_total_current(state, targets),
                getup=h.getup.state,
            ))
            return check(state, targets, cutoff)

        h.scheduler._check_overcurrent = spy
        rows = h.run(fall_then_stand(), ticks)
        self.assertTrue(hand_back, "walk never took over")
        return h, rows, checks, hand_back[0]

    def test_hand_back_does_not_trip_on_get_up_goals_in_history(self):
        h, rows, checks, hand_back = self.run_hand_back()
        tail = h.scheduler._cmd_history.maxlen
        after = [c for c in checks if c["tick"] >= hand_back]
        # The proxy still reads the get-up's saturated goals for `tail`
        # ticks: above the normal limit (the old trip), below the get-up one.
        stale = after[:tail]
        for c in stale:
            self.assertGreaterEqual(c["current"], scheduler_module.OVERCURRENT_CUTOFF_A)
            self.assertLess(c["current"], scheduler_module.OVERCURRENT_CUTOFF_A_GETUP)
            self.assertEqual(c["cutoff"], scheduler_module.OVERCURRENT_CUTOFF_A_GETUP)
        # Then the history holds only walk goals and the normal limit is back.
        for c in after[tail:]:
            self.assertLess(c["current"], 1.0)
            self.assertEqual(c["cutoff"], scheduler_module.OVERCURRENT_CUTOFF_A)
        self.assertGreater(len(after), tail + sec(1.0))
        self.assertNotIn("Overcurrent", h.output)
        self.assertEqual(len(rows), STAND_TICK + sec(2.0))
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)

    def test_sustained_walking_overcurrent_still_trips_at_normal_limit(self):
        tail = scheduler_module.OVERCURRENT_PROXY_DELAY_TICKS + 1
        start = tail + 10  # well after the get-up goals left the history
        h, rows, checks, hand_back = self.run_hand_back(saturate_walk_after=start)
        self.assertIn(
            f"threshold {scheduler_module.OVERCURRENT_CUTOFF_A:.2f} A", h.output
        )
        last = checks[-1]
        self.assertEqual(last["cutoff"], scheduler_module.OVERCURRENT_CUTOFF_A)
        self.assertLess(last["current"], scheduler_module.OVERCURRENT_CUTOFF_A_GETUP)
        # Caught once the saturated goal reaches the delay-aligned history
        # slot plus the debounce: no later than with a walk-only history.
        self.assertLessEqual(
            last["tick"] - (hand_back + start),
            tail + scheduler_module.OVERCURRENT_DEBOUNCE_TICKS,
        )
        self.assertLess(len(rows), STAND_TICK + sec(2.0))


def tilted(degrees):
    rad = math.radians(degrees)
    return [math.sin(rad), 0.0, -math.cos(rad)]


SETTLE_TICKS = scheduler_module.GETUP_HANDBACK_SETTLE_TICKS
GETUP_A = scheduler_module.OVERCURRENT_CUTOFF_A_GETUP
NORMAL_A = scheduler_module.OVERCURRENT_CUTOFF_A
LEANING = tilted(18.0)  # standing (< ~25.8 deg) but not settled (>= 12 deg)
SETTLED = tilted(5.0)
STAND_START = sec(0.2) + sec(1.0)  # first upright tick after the fall


def fall_then_lean(tilt_for_tick, *, moves=frozenset({"walk"})):
    """Upright 0.2 s, fallen 1 s, then ``tilt_for_tick(tick)`` gravity."""

    def script(tick, _inhibited):
        if tick < sec(0.2):
            gravity = UPRIGHT
        elif tick < STAND_START:
            gravity = FALLEN
        else:
            gravity = tilt_for_tick(tick)
        return gravity, UserInput(
            active_moves=set(moves),
            velocity={"vx": 0.0, "vy": 0.0, "vtheta": 0.0},
        )

    return script


class SettledHandBackTest(unittest.TestCase):
    """Hand-back to walk waits for tilt < GETUP_HANDBACK_SETTLE_TILT_DEG for
    GETUP_HANDBACK_SETTLE_TICKS consecutive ticks; get-up keeps balancing."""

    def test_settle_constants(self):
        self.assertEqual(scheduler_module.GETUP_HANDBACK_SETTLE_TILT_DEG, 12.0)
        self.assertEqual(SETTLE_TICKS, 10)
        self.assertLess(LEANING[2], -0.9)  # stand debounce counts
        self.assertGreater(LEANING[2], -math.cos(math.radians(12.0)))

    def test_hand_back_after_settle_window(self):
        h = Harness(FakeWalk(balances=True))
        settle_at = STAND_START + sec(2.0)
        rows = h.run(
            fall_then_lean(lambda t: LEANING if t < settle_at else SETTLED),
            settle_at + sec(2.0),
        )
        # The direct release at the stand debounce is gated: still leaning,
        # so the get-up actor balances instead of handing back.
        self.assertTrue(rows[STAND_TICK + 1]["balancing"])
        self.assertEqual(rows[STAND_TICK + 1]["getup"], MoveState.ACTIVE)
        # Gravity scripted at tick t is observed at tick t + 1 (as for
        # STAND_TICK), and rows[t + 1] shows the state left by tick t: the
        # SETTLE_TICKS-th settled observation releases.
        hand_back = settle_at + SETTLE_TICKS
        for row in rows[STAND_TICK + 1:hand_back + 1]:
            self.assertTrue(row["balancing"], row["tick"])
        self.assertEqual(rows[hand_back]["walk_steps"], rows[STAND_TICK]["walk_steps"])
        self.assertFalse(rows[hand_back + 1]["override"])
        self.assertFalse(rows[hand_back + 1]["balancing"])
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)
        self.assertGreater(h.walk.steps, 0)
        self.assertEqual(h.output.count("hand-back to walk deferred"), 1)
        self.assertIn("now 18.0 deg", h.output)
        self.assertIn("released: balancing walk requested", h.output)

    def test_unsettled_tilt_keeps_deferring(self):
        # Dips below the settle angle, but never for SETTLE_TICKS in a row.
        h = Harness(FakeWalk(balances=True))
        period = SETTLE_TICKS - 1
        rows = h.run(
            fall_then_lean(lambda t: SETTLED if (t // period) % 2 else LEANING),
            STAND_START + sec(4.0),
        )
        self.assertTrue(rows[-1]["override"])
        self.assertTrue(rows[-1]["balancing"])
        self.assertEqual(rows[-1]["getup"], MoveState.ACTIVE)
        self.assertEqual(h.walk.steps, rows[STAND_TICK]["walk_steps"])
        self.assertFalse(rows[-1]["failed"])
        self.assertIsNone(rows[-1]["started"])
        self.assertEqual(h.output.count("hand-back to walk deferred"), 1)
        self.assertNotIn("released", h.output)

    def test_already_settled_at_stand_hands_back_directly(self):
        h = Harness(FakeWalk(balances=True))
        rows = h.run(fall_then_lean(lambda t: SETTLED), STAND_START + sec(2.0))
        self.assertFalse(any(row["balancing"] for row in rows))
        self.assertFalse(rows[STAND_TICK + 1]["override"])
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)
        self.assertNotIn("deferred", h.output)

    def test_fall_during_deferred_wait_starts_new_attempt(self):
        h = Harness(FakeWalk(balances=True))
        refall = STAND_START + sec(2.0)
        restand = refall + sec(1.0)

        def tilt(t):
            if t < refall:
                return LEANING
            if t < restand:
                return FALLEN
            return SETTLED

        rows = h.run(fall_then_lean(tilt), restand + sec(2.0))
        self.assertTrue(rows[refall]["balancing"])
        new_attempt = next(row for row in rows[refall:] if not row["balancing"])
        self.assertLessEqual(new_attempt["tick"] - refall, 16)
        self.assertTrue(new_attempt["override"])
        self.assertIsNotNone(rows[restand]["started"])
        self.assertIn("Fall detected while balancing", h.output)
        self.assertEqual(h.output.count("Automatic get-up attempt started"), 2)
        # Upright and settled after the second attempt: direct hand-back.
        self.assertFalse(rows[-1]["override"])
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)

    def test_imu_outage_restarts_settle_window(self):
        h = Harness(FakeWalk(balances=True), imu_shutdown_after_s=5.0)
        settle_at = STAND_START + sec(2.0)
        glitch = settle_at + SETTLE_TICKS - 2

        def on_tick(tick):
            h.controller.imu_valid = tick != glitch

        h.on_tick = on_tick
        rows = h.run(
            fall_then_lean(lambda t: LEANING if t < settle_at else SETTLED),
            settle_at + sec(2.0),
        )
        # Without the glitch, rows[settle_at + SETTLE_TICKS + 1] is released.
        self.assertTrue(rows[settle_at + SETTLE_TICKS + 1]["balancing"])
        self.assertTrue(rows[glitch + SETTLE_TICKS]["balancing"])
        self.assertFalse(rows[glitch + SETTLE_TICKS + 1]["override"])
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)

    def run_cutoffs(self, settle_at, *, moves=frozenset({"walk"}), tilt=None):
        h = Harness(FakeWalk(balances=True))
        cutoffs = {}
        check = h.scheduler._check_overcurrent

        def spy(state, targets, cutoff):
            cutoffs[len(h.rows) - 1] = cutoff
            return check(state, targets, cutoff)

        h.scheduler._check_overcurrent = spy
        rows = h.run(
            fall_then_lean(
                tilt or (lambda t: LEANING if t < settle_at else SETTLED), moves=moves
            ),
            settle_at + sec(2.0),
        )
        tail = h.scheduler._cmd_history.maxlen
        first_balancing = next(t for t in sorted(cutoffs) if rows[t + 1]["balancing"])
        self.assertEqual(first_balancing, STAND_TICK)
        # The attempt itself ran under the get-up limit.
        self.assertTrue(any(
            c == GETUP_A for t, c in cutoffs.items() if sec(0.2) < t < STAND_TICK
        ))
        return h, rows, cutoffs, tail

    def test_unsettled_balance_keeps_getup_cutoff_until_hand_back(self):
        # The get-up actor is still pulling the trunk upright: get-up limit
        # through the deferred wait, then the usual tail after the hand-back.
        settle_at = STAND_START + sec(0.6)
        h, rows, cutoffs, tail = self.run_cutoffs(settle_at)
        hand_back = settle_at + SETTLE_TICKS
        self.assertTrue(rows[hand_back]["balancing"])
        self.assertFalse(rows[hand_back + 1]["override"])
        self.assertEqual(
            [cutoffs[t] for t in range(STAND_TICK, hand_back + tail + 1)],
            [GETUP_A] * (hand_back + tail - STAND_TICK) + [NORMAL_A],
        )
        self.assertEqual({c for t, c in cutoffs.items() if t >= hand_back + tail}, {NORMAL_A})

    def test_settled_balance_without_walk_returns_to_normal_cutoff(self):
        settle_at = STAND_START + sec(0.6)
        relean = settle_at + sec(0.8)

        def tilt(t):
            return SETTLED if settle_at <= t < relean else LEANING

        h, rows, cutoffs, tail = self.run_cutoffs(settle_at, moves=frozenset(), tilt=tilt)
        settled = settle_at + SETTLE_TICKS  # first tick with a settled stance
        self.assertTrue(rows[-1]["balancing"])
        self.assertEqual(
            [cutoffs[t] for t in range(STAND_TICK, settled + 2 * tail + 1)],
            [GETUP_A] * (settled + 2 * tail - STAND_TICK) + [NORMAL_A],
        )
        self.assertEqual({c for t, c in cutoffs.items() if t >= settled + 2 * tail}, {NORMAL_A})
        # A later lean does not re-arm the get-up limit once settled.
        self.assertTrue(any(t > relean + sec(1.0) for t in cutoffs))
        self.assertNotIn("deferred", h.output)

    def test_unsettled_getup_cutoff_is_bounded(self):
        cap = scheduler_module.GETUP_SETTLE_CUTOFF_MAX_TICKS
        settle_at = STAND_START + sec(4.0)
        h, rows, cutoffs, tail = self.run_cutoffs(settle_at)
        self.assertGreater(settle_at - STAND_TICK, cap + 2 * tail + sec(1.0))
        self.assertEqual(
            [cutoffs[t] for t in range(STAND_TICK, STAND_TICK + cap + 2 * tail + 1)],
            [GETUP_A] * (cap + 2 * tail) + [NORMAL_A],
        )
        # The rest of the deferred balance and the hand-back after it run
        # under the normal limit: no get-up goals are left in the history.
        hand_back = settle_at + SETTLE_TICKS
        self.assertFalse(rows[hand_back + 1]["override"])
        self.assertEqual(
            {c for t, c in cutoffs.items() if t >= STAND_TICK + cap + 2 * tail},
            {NORMAL_A},
        )


class GatelessSourceReleaseTest(unittest.TestCase):
    """Keyboard/MuJoCo viewer: no B/A/R3 gate, so a move toggle releases."""

    def run_toggle(self, power):
        h = Harness(FakeWalk(balances=False), power=power,
                    extra_moves={"head": CountingMove()})
        toggle_at = STAND_TICK + sec(1.0)

        def script(tick, inhibited):
            gravity, user_input = fall_then_stand()(tick, inhibited)
            if tick >= toggle_at:
                user_input.active_moves = {"walk", "head"}
            if power:
                user_input.torque_enabled = True
                user_input.policy_enabled = tick >= 3
            return gravity, user_input

        rows = h.run(script, toggle_at + sec(1.0))
        self.assertTrue(rows[toggle_at]["balancing"])
        return h, rows, toggle_at

    def test_move_toggle_releases_balance_without_power_gate(self):
        h, rows, toggle_at = self.run_toggle(power=False)
        self.assertFalse(rows[toggle_at + 1]["balancing"])
        self.assertFalse(rows[-1]["override"])
        self.assertEqual(rows[-1]["getup"], MoveState.INACTIVE)
        self.assertEqual(rows[-1]["walk"], MoveState.ACTIVE)
        self.assertGreater(h.scheduler.registered_moves["head"].calls, 0)
        self.assertIn("released: operator changed active moves", h.output)

    def test_move_toggle_does_not_release_with_power_gate(self):
        h, rows, _ = self.run_toggle(power=True)
        self.assertTrue(rows[-1]["balancing"])
        self.assertEqual(rows[-1]["getup"], MoveState.ACTIVE)
        self.assertEqual(h.scheduler.registered_moves["head"].calls, 0)
        self.assertNotIn("released", h.output)


class OperatorIsolationTest(unittest.TestCase):
    def test_operator_input_does_not_reach_moves_while_balancing(self):
        squat = CountingMove()
        h = Harness(FakeWalk(balances=False), power=True, extra_moves={"squat": squat})

        def script(tick, inhibited):
            gravity, user_input = fall_then_stand()(tick, inhibited)
            user_input.torque_enabled = True
            user_input.policy_enabled = tick >= 3
            if tick >= STAND_TICK + 1:
                user_input.active_moves = {"walk", "squat"}
                user_input.velocity = {"vx": 0.4, "vy": -0.2, "vtheta": 0.3}
            return gravity, user_input

        rows = h.run(script, STAND_TICK + sec(2.0))
        self.assertTrue(rows[-1]["balancing"])
        self.assertEqual(squat.calls, 0)
        self.assertEqual(h.walk.steps, rows[STAND_TICK]["walk_steps"])
        balancing_velocities = h.getup.velocities[-sec(1.5):]
        self.assertTrue(balancing_velocities)
        for velocity in balancing_velocities:
            self.assertEqual(velocity, {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})


class HardwareGateReleaseTest(unittest.TestCase):
    """B/A/R3 gate (PICO/GC300): R3 off or B ends standing balance."""

    def gate_script(self, gate_at):
        def script(tick, _inhibited):
            gravity, user_input = fall_then_stand()(tick, _inhibited)
            torque, policy = gate_at(tick)
            user_input.torque_enabled = torque
            user_input.policy_enabled = policy
            return gravity, user_input

        return script

    def run_gate(self, late_gate):
        h = Harness(FakeWalk(balances=False), power=True)
        release_at = STAND_TICK + sec(1.0)

        def gate_at(tick):
            if tick < 3:
                return True, False  # A
            if tick < release_at:
                return True, True  # R3 on
            return late_gate

        rows = h.run(self.gate_script(gate_at), release_at + sec(1.0))
        self.assertTrue(rows[release_at]["balancing"])
        self.assertFalse(rows[release_at + 2]["balancing"])
        self.assertFalse(rows[-1]["override"])
        self.assertEqual(rows[-1]["getup"], MoveState.INACTIVE)
        return h

    def test_r3_off_releases_balance_to_neutral_gate(self):
        h = self.run_gate((True, False))
        self.assertIn("released: policy output withheld", h.output)

    def test_b_releases_balance_to_limp(self):
        h = self.run_gate((False, False))
        self.assertIn("released: policy output withheld", h.output)
        self.assertEqual(h.controller.torque_writes[-1][1], [False] * len(MOTOR_TO_ID))


class _BalancingChild(Move):
    def can_balance(self, user_input):
        return True

    def step(self, _obs, _command):
        pass


class CanBalanceTest(unittest.TestCase):
    missing = Path(tempfile.gettempdir()) / "missing_pico_policy_for_balance.onnx"

    def test_hold_and_base_move_cannot_balance(self):
        self.assertFalse(_HoldPositionMove(controller=None).can_balance(UserInput()))
        self.assertFalse(FakeGetup().can_balance(UserInput()))

    def test_selector_without_policy_falls_back_to_hold(self):
        move = PolicySelectableWalkMove(controller=None, pico_policy_path=self.missing)
        move.preload()
        self.assertFalse(move.can_balance(UserInput(locomotion_policy="pico_teleop")))
        self.assertFalse(move.can_balance(UserInput(locomotion_policy="walk")))

    def test_selector_with_loaded_policy_balances_only_when_selected(self):
        move = PolicySelectableWalkMove(
            controller=None, pico_policy_path=self.missing, pico_move=_BalancingChild()
        )
        self.assertTrue(move.can_balance(UserInput(locomotion_policy="pico_teleop")))
        self.assertFalse(move.can_balance(UserInput(locomotion_policy="walk")))
        self.assertFalse(move.can_balance(
            UserInput(locomotion_policy="pico_teleop", learned_policy_degraded=True)
        ))
        self.assertFalse(move.can_balance(UserInput(locomotion_policy="bogus")))

    def test_walk_move_balances(self):
        from moves.walk import WalkMove
        from policy_fixtures import WALK_POLICY_FIXTURE

        walk = WalkMove(controller=None, policy_path=WALK_POLICY_FIXTURE)
        self.assertTrue(walk.can_balance(UserInput()))


if __name__ == "__main__":
    unittest.main()
