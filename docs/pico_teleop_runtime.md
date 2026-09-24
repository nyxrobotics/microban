# PICO 4 Ultra hybrid-policy runtime

The robot runtime keeps the existing walking policy as the startup mode. With
network control active, press the **left-controller X button while the left
trigger is released** to toggle between:

- `walk`: the existing velocity walking policy;
- `pico_teleop`: the independently trained 83-observation/18-action policy.

This is a controller state, not an environment-variable switch. The bridge
places `locomotion_policy` in every complete UDP snapshot. A mode change while
the left trigger is held is rejected, walking is disarmed, and another released
snapshot is required before a later trigger press can move the robot. The two
policies never own the same 18 joints in the same scheduler tick.

The other controls are unchanged:

| Control | Result |
|---|---|
| left stick | forward/backward and lateral velocity |
| right stick X | body yaw rate |
| left trigger (held) | enable the selected locomotion policy |
| right trigger (held) | slew camera-head yaw to trunk-forward |
| HMD orientation | camera-head yaw/roll/pitch |
| left grip (held) | show calibrated robot stereo view; otherwise passthrough |
| left X, trigger released | toggle `walk` / `pico_teleop` |

## Install and validate a trained policy

Export from `mjlab_microban`, copy the resulting file as
`src/agents/pico_teleop.onnx`, then validate it without touching hardware:

```bash
cd /home/kanade/Git-projects/mjlab_microban
uv run --locked python -m mjlab_microban.scripts.export_teleop_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<run>/model_<iteration>.pt \
  --output /tmp/pico_teleop.onnx

cd /home/kanade/Git-projects/microban
cp /tmp/pico_teleop.onnx src/agents/pico_teleop.onnx
PYTHONPATH=src .venv/bin/python tools/validate_pico_policy.py
```

The validator and runtime both fail closed unless the model has exactly one
`[1,83]` input and one `[1,18]` output and its metadata agrees on observation
order, all 21 encoder defaults, the 18 action joint names/order, body-frame gyro,
50 Hz rate, action scale/soft limits, target coordinate frames and training
bounds. A legacy `walk.onnx` therefore cannot be selected accidentally.

Run the normal network entry point; no policy-selection environment variable is
needed:

```bash
make teleop-run HOST=microban
```

If `pico_teleop.onnx` is absent, normal walking remains available, but pressing
X to request `pico_teleop` stops the control loop and disables torque rather than
falling back silently.

## Body-target reference

Training defines hand/foot commands as offsets from an episode-reset reference
in the robot trunk frame (`+X` forward, `+Y` left, `+Z` up). The PICO bridge must
capture the matching body-tracker reference at explicit policy enable/reset and
keep it fixed. It must never send the absolute human pelvis-to-limb positions.
Stale/jumping tracking clears the calibration, removes `walk`, zeros velocity
and sends no hand/foot target. Walking stays disarmed until the left trigger is
released and a fresh reference can be established again. The robot receiver
independently requires `body_target_contract: "microban_pico_offsets_v1"`,
`body_target_safety_margin: 0.8`, complete left/right foot and hand pairs, and
the same live bounds as the bridge: hands `[-0.064, 0.064] m` on every axis,
feet `[-0.024, 0.024] m` on X/Y and `[0, 0.040] m` on Z. A missing, malformed or
out-of-range value immediately removes `walk`, zeros velocity, clears both
target pairs and requires a valid released-trigger snapshot before rearming.

The runtime clips received offsets to the ONNX-recorded training support before
inference. This is a last safety boundary, not a substitute for bridge-side
freshness, calibration, scaling and range checks.

## Required supported-robot acceptance

Do not use the learned policy free-standing immediately. In this order:

1. Inspect the policy in simulation and confirm it can stand, walk, turn, return
   home and track bounded hand/foot commands.
2. Put Microban in a rigid support harness with feet clear of the floor.
3. With torque disabled, rotate the trunk by hand about positive roll, pitch and
   yaw and record BMI088 gyro signs. Confirm the transform based on
   `IMU_MOUNT_QUAT` produces Microban body-frame `+X/+Y/+Z` angular velocity.
   The constant is WXYZ `sensor -> body`; the MJCF IMU site, orientation
   conversion and gyro conversion share this convention and a contract test.
4. With the harness fitted, enable the policy with zero velocity and zero body
   targets. Verify every encoder/action name and direction at low gain/current.
5. Test one axis and small target at a time, then trigger-release return to home,
   tracking timeout and X-mode switch. If a get-up move is installed locally,
   also verify its scheduler ownership transition.
6. Only after those pass, test feet on the floor at reduced command ranges.

The mounting quaternion exists in software, but its gyro-axis/sign mapping has
not yet been physically certified on this robot. That hardware check is the
remaining gate between a trained artifact and safe free-standing deployment.
