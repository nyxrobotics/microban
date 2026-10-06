# PICO teleop degraded-operation contract

The contract-v12 PICO actor provides locomotion and R3 balance. The robot's
`walk` move name routes through `PolicySelectableWalkMove`; the old
`src/agents/walk.onnx` actor is not run as the production fallback. If the PICO
actor cannot run, the selector holds its last complete joint goals. At startup,
before a PICO goal exists, it holds the measured pose. The pinned old actor is
still checked by the v12 deployment validator as an artifact dependency.

| Condition | Same-cycle result | Recovery |
|---|---|---|
| Camera stale/missing, calibration/FOV/IPD invalid | View may fall back to passthrough; locomotion is unchanged | Camera can recover independently |
| Optional body/hand targets missing, malformed, stale, or outside the wire envelope | Use zero offsets for missing feet and mark missing hands inactive; preserve left trigger and joystick velocity | Valid tracking can resume without restarting the gait |
| Head tracking unavailable | Ignore that head pose; keep buttons, sticks, arms, and locomotion independent | Head tracking resumes when a valid pose arrives |
| PICO ONNX absent, rejected, or unable to start | Hold the last complete joint goals, or the measured pose if none were produced | Atomic replacement is loaded in the background and can be selected on a later activation |
| PICO inference fails or produces an unusable target | Hold the last complete PICO goals in the faulting control cycle; do not start the old walk actor | Keep the hold latched until locomotion is released and activated again |
| Left X / primary button changes | No locomotion effect; bridge keeps requesting `pico_teleop` | Left trigger enables walking; R3 enables zero-velocity balance |
| PICO controller samples or PC-to-robot UDP packets stop | Replay the last authenticated operator state, including motion, policy and torque state | Fresh input replaces that state when it arrives; a sender-authority change requires a new released-trigger snapshot |
| IMU becomes unavailable | Keep the previous motor goals and torque while policy output is inhibited | Valid IMU input allows the normal motion gate to resume |
| Robot falls in policy mode (R3 on) | After 0.3 s tilted beyond 60 deg, walking is inhibited and the get-up actor (contract v5, +-pi servo range; a v4 +-1.57 rad model is refused) runs (up within ~2-3 s in sim; 20 s limit per attempt) | Once upright for 0.4 s, the get-up actor keeps balancing the robot until a walk move that can itself balance is requested at zero velocity and the trunk has stayed within 12 deg of upright for 0.2 s (the hand-back waits for that settled stance); R3 off, B or an actor fault also end it. A new fall starts a fresh attempt |
| Get-up attempt exceeds 20 s or the actor output is non-finite | Latched fault: hold the measured pose at P900 and return slowly toward neutral | R3 off then on (or B) clears the fault and allows a new attempt |

After an automatic get-up, tracking does not resume on its own while the
operator keeps holding the triggers: the get-up actor keeps balancing. Release
the left trigger once after the robot is upright to hand the legs back to the
PICO policy (it takes over at zero velocity), and the right trigger for the
arm and head tracking; a release while the robot is still getting up does not
count. Measured in the runtime MuJoCo sim on 2026-10-02 with the PICO policy
loaded, tracking resumed 0.5-3.8 s after the release. When
`src/agents/pico_teleop.onnx` fails its contract or startup self-test
(docs/policies.md), the legs fall back to the static position hold, which
cannot balance: the get-up actor then keeps the robot standing until R3 is
switched off, and a policy package of the current contract must be installed
to restore tracking.

An explicit B press turns torque off. A changes to the neutral-return mode; R3
selects the PICO policy for balance even with the left trigger released. Missing
tracking data and packet gaps do not synthesize A or B events. Before the first
authenticated operator snapshot, the hardware gate stays in its startup state.
The A/B servo-bus command is a broadcast sync-write. Because that packet has no
per-servo acknowledgement, the runtime reads the torque-enable registers before
resuming normal control. It retries IDs whose state differs from the command
for up to about one second and reports any that remain unconfirmed. A starts the
neutral return for confirmed joints while continuing ON retries for the others.
OFF cancels pending ON retries first; unresolved IDs also receive slower
background checks. The limp loop does not send goal positions. A seeds measured
goals before enabling torque when feedback is available. If an initial position
reply is missing, it seeds that joint directly to neutral and enables it without
waiting for a position read; that joint may move directly to neutral. During the
A return, subsequent position reply failures do not stop feedforward neutral
goals. During policy output, a joint with a missing position reply is resent its
previous goal until feedback returns.

The learned implementation is constructed through `LearnedMoveFactory` in
`moves/policy_selector.py`. The current v12 parser and its embedded source
identity checks remain authoritative for policy admission.

The focused non-hardware regression command is:

```bash
PYTHONPATH=src uv run --with pytest python -m pytest -q \
  tests/test_policy_selector.py tests/test_network_input.py
```

These tests do not open the motor bus or launch the physical gateway. They cover
selector fallback holding, optional-tracker isolation, packet-to-selector
activation, atomic-file reload, and replay of authenticated operator input.
