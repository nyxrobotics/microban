# PICO 4 Ultra hybrid-policy runtime

The robot runtime keeps the existing walking policy as the startup mode. With
network control active, the **current state of the left-controller X button**
selects the policy momentarily:

- X released: `walk`, the existing velocity walking policy;
- X held: `pico_teleop`, the independently trained
  83-observation/18-action policy.

This is a controller state, not an environment-variable switch. The bridge
places `locomotion_policy` in every complete UDP snapshot. Changing X state while
the left trigger is held immediately disarms walking, and a released-trigger
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
| left X (held) | select `pico_teleop`; releasing X selects `walk` |

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
bounds. The learned-policy training contract must be version `5`, while the
independent observation schema remains version `2`, with previous-action
semantics exactly
`effective_action_after_absolute_target_soft_clip_in_raw_delta_coordinates`.
Missing/unversioned models and v1 raw-action-feedback models are rejected even
though their tensor shapes are also `[1,83] -> [1,18]`.

V5 additionally requires the exact distribution declaration
`diagonal_normal_latent_with_per_joint_asymmetric_zero_anchored_arctan_bijection_v1`,
a `0.05` actor target-guard ratio and a `0.0001 rad` maximum default-interior
epsilon. Four strict JSON metadata arrays carry the unrounded 18-joint raw soft
limits and guarded actor limits. The runtime independently re-derives all four
from its compiled-in `NEUTRAL_POSE`, action scale and physical soft limits,
using the same float32 boundary arithmetic as training, and requires an exact
or four-float32-epsilon-tight value match (at most `4.7684e-7 rad` with the
current unit action scale). That narrow allowance covers only the operation
ordering difference between MuJoCo/MjLab limit resolution and the runtime's
already-compiled soft limits; three-decimal metadata remains far outside it.
Every actor interval must strictly contain zero and be strictly inside its
corresponding raw soft interval. Missing, rounded, non-finite or materially
tampered vectors therefore fail closed.

They also require the parity-gate version-1 deterministic PyTorch-to-ONNX
record: `onnx_parity_verified=true`, its fixed runtime/corpus/tolerances, a
canonical `model_N.pt` checkpoint filename, iteration `N`, completed-update
count `N+1`, and a lowercase 64-hex checkpoint SHA-256. The validator prints
that checkpoint identity for the deployment record. Legacy or manually
exported artifacts without this gate record, including `walk.onnx`, are
rejected.

After each inference, the runtime converts every raw action to
`default_joint_pos + raw_action * scale`, clips that absolute target to the
compiled-in soft limits, commands the clipped value, then maps it back with
`(clipped_target - default_joint_pos) / scale`. Only that effective delta is
stored in the next observation. This exactly matches v5 training and prevents
unbounded network output from feeding back while a servo target is saturated.
The v5 actor bijection should keep normal inference inside its narrower guarded
interval; the wider compiled-in absolute soft clip remains an independent final
defense and is not removed or widened by model metadata.

V5 must be trained from a clean run. An unversioned v1 checkpoint may be
inspected only with the simulator evaluator's explicit diagnostic flag; it
cannot be resumed, exported as v5, accepted by this runtime, or copied into
`src/agents` as a deployable policy. Versioned-v2, v3 and v4 checkpoints are
rejected even for that diagnostic path because their tensor widths do not prove
v5 bounded-actor training semantics. The diagnostic pre-update artifact
`model_pristine.pt` (iteration `-1`) is also rejected: deployment accepts only
a parity-gated canonical `model_N.pt` with non-negative `N`.

After metadata validation, the offline validator also runs the same 16 fixed
neutral, lower-bound, upper-bound, midpoint and seed-`20260924` finite inputs
through the installed ONNX Runtime provider as part of model loading. Every
result must be exactly `[1,18]`, finite and strictly inside all 18 guarded actor
bounds; equality with either endpoint is a failure. This is reported as
`onnxruntime_compatibility_smoke`. It checks that the deployment runtime can
load and execute the bounded graph, but it is not another PyTorch/ONNX
numerical-parity result because the validator does not carry expected PyTorch
outputs. Run it in Microban's actual Python/ONNX Runtime environment before
enabling torque, and record the provider names from its JSON output.

The provenance record is traceability and internal-consistency metadata, not a
cryptographic signature of the ONNX itself. Only copy artifacts produced by the
checked-in exporter from a trusted training workspace.

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
independently requires
`body_target_contract: "microban_pico_offsets_v2_both_feet_stationary"`,
`body_target_safety_margin: 0.8`, complete left/right foot and hand pairs, and
the same live bounds as the bridge: hands `[-0.064, 0.064] m` on every axis,
single-foot offsets `[-0.024, 0.024] m` on X/Y and `[0, 0.040] m` on Z. The
bridge must project a support foot at Z `<= 0.0025 m` to exact XYZ zero; the
receiver rejects any non-zero vector left in that band. Thus the exported zero
lower bound represents only the exact-zero inactive command; an active
single-foot Z is `(0.0025, 0.040] m`, matching v4 training from the floor
boundary upward. After that projection, if both foot offsets are active, each is
limited to X/Y `[-0.008, 0.008] m` and active Z `(0.0025, 0.016] m`, and `vx`,
`vy` and `vtheta` must all be exactly zero. A missing, legacy-v1, malformed,
out-of-range or moving-both-feet snapshot immediately removes `walk`, zeros
velocity, clears both target pairs and requires a valid released-trigger
snapshot before rearming.

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
