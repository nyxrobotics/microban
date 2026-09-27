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
| Robot fall with the current get-up artifact | Fall detection inhibits walking; the unavailable get-up actor returns toward neutral | Automatic get-up requires a newly trained, accepted v2 artifact |

An explicit B press turns torque off. A changes to the neutral-return mode; R3
selects the PICO policy for balance even with the left trigger released. Missing
tracking data and packet gaps do not synthesize A or B events. Before the first
authenticated operator snapshot, the hardware gate stays in its startup state.

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
