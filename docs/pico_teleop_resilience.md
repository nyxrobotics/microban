# PICO teleop degraded-operation contract

Basic joystick locomotion is the availability baseline. `src/agents/walk.onnx` is
the pinned actor; `pico_teleop.onnx`, body trackers and the stereo view are optional
enhancements. The robot runtime must not turn an enhancement fault into a stop.

| Condition | Same-cycle result | Recovery |
|---|---|---|
| Camera stale/missing, calibration/FOV/IPD invalid | View may fall back to passthrough; locomotion is unchanged | Camera can recover independently |
| Body target missing, malformed, stale, out of envelope | Keep left trigger and all three stick axes; clear targets and use `walk` | Legacy remains latched until trigger release; next press may retry learned policy |
| Learned ONNX absent or contract/load rejected | Start `walk` with the same observation | Atomically replace the file; background validation makes it eligible on a later activation |
| Learned start/inference exception or non-finite/unsafe output | Discard the learned tick, then call legacy `on_start` and `step` in that control cycle | Reason remains in `fallback_reason`; reload is attempted without process restart and adoption waits for a later activation |
| Left X / primary button changes | No locomotion effect; bridge keeps requesting `pico_teleop` | Left trigger remains the only momentary enable |
| Controller/network packets stop | Watchdog returns neutral/HOME and disarms | After reconnect, release the left trigger once, then press to move |
| Robot-side IMU/fall/current safety interlock | Existing robot safety behavior remains authoritative | Follow that interlock's explicit recovery procedure |

The learned implementation is constructed through `LearnedMoveFactory` in
`moves/policy_selector.py`. When the next policy contract is finalized, inject its
loader there; do not relax the current parser or guess future metadata fields.

Run the non-hardware regression suite from the repository root:

```bash
PYTHONPATH=src uv run --with pytest python -m pytest -q \
  tests/test_policy_selector.py tests/test_network_input.py
```

These tests never open the motor bus or launch the physical gateway. They cover
same-cycle inference fallback, command preservation, tracker degradation latching,
trigger-only packet-to-selector activation, atomic-file reload and network
reconnect deadman rules.
