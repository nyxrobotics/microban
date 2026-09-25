# Legacy contract-v10 PICO hybrid-policy runtime

> This file is retained as the contract-v10 metadata reference. New deployments
> must use [the contract-v12 runtime and deployment procedure](pico_teleop_v12_runtime.md).
> Do not use the v10 export commands below for the current Microban policy.

The current PICO bridge always requests `pico_teleop`; the left-controller X
(WebXR `primary_button`) is parsed for protocol compatibility but is ignored by
both bridge mappers. The **left trigger is the only momentary locomotion and
tracking enable**: released sends no `walk`/`hmd_head` activation, and held sends
both with `locomotion_policy=pico_teleop`. There is no controller policy toggle
and no environment-variable switch. The receiver may internally downgrade a
learned request to the pinned `walk` actor after a tracker/policy fault.

The learned policy is optional. A missing/rejected ONNX, load/start/inference
exception, non-finite output, or invalid body-target snapshot immediately uses the
pinned `walk.onnx` actor with the current joystick command. A tracker/policy fault
latches `walk` for the rest of that left-trigger activation; recovery is considered
only after release and the next press, so it cannot surprise-switch mid-stride.

The other controls are unchanged:

| Control | Result |
|---|---|
| left stick | forward/backward and lateral velocity |
| right stick X | body yaw rate |
| left trigger (held) | enable PICO locomotion, HMD and available body/hand tracking |
| right trigger (held) | enable absolute HMD-origin controller arm tracking |
| right grip (held) | slew camera-head yaw to trunk-forward |
| HMD orientation | camera-head yaw/roll/pitch |
| left grip (held) | show calibrated robot stereo view; otherwise passthrough |
| left X | no locomotion-policy function |

## Install and validate a trained policy

Export from `mjlab_microban` to a temporary file, validate it without touching
hardware, then atomically install it as `src/agents/pico_teleop.onnx`:

```bash
cd /home/kanade/Git-projects/mjlab_microban
uv run --locked python -m mjlab_microban.scripts.export_teleop_onnx \
  --checkpoint logs/rsl_rl/mjlab_microban_teleop/<run>/model_14999.pt \
  --acceptance-receipt artifacts/teleop_v10_gates/<run>_boundary_15000_gate.json \
  --require-final-acceptance \
  --output /tmp/pico_teleop.onnx

cd /home/kanade/Git-projects/microban
PYTHONPATH=src .venv/bin/python tools/validate_pico_policy.py \
  /tmp/pico_teleop.onnx
install -m 0644 /tmp/pico_teleop.onnx src/agents/.pico_teleop.onnx.new
mv -f -- src/agents/.pico_teleop.onnx.new src/agents/pico_teleop.onnx
```

The validator must pass against the file in `/tmp` before installation. The
final rename stays within `src/agents`, so it atomically replaces any installed
policy rather than exposing a partially copied ONNX file.

The validator and runtime both fail closed unless the model has exactly one
`[1,83]` input and one `[1,18]` output and its metadata agrees on observation
order, all 21 encoder defaults, the 18 action joint names/order, body-frame gyro,
50 Hz rate, action scale/soft limits, target coordinate frames and training
bounds. The learned-policy training contract must be version `10`, the training
provenance schema must be version `2`, and the independent observation schema
remains version `2`, with previous-action semantics exactly
`effective_action_after_absolute_target_soft_clip_in_raw_delta_coordinates`.
The actor initialization must be
`full_state_v9_model1499_to_v10_fixed_lr_v1`, and the recipe revision must be
`v10_v9_model1499_full_state_migration_fixed_lr_pico_curriculum_v1`.
Missing, unversioned and contract-v1 through contract-v9 models are rejected
even when their tensor shapes are also `[1,83] -> [1,18]`.

Contract v10 requires the exact distribution declaration
`diagonal_normal_ppo_latent_stored_exactly_then_per_joint_asymmetric_zero_anchored_arctan_environment_transform_with_operational_envelope_v1`.
PPO stores and scores its finite latent directly; only the action sent to the
environment is passed through the asymmetric, zero-anchored arctangent. The
deterministic ONNX graph includes that physical-action transform, so the robot
must consume its output directly and must not apply a second transform.

The metadata also fixes the `0.05` actor target-guard ratio, `0.0001 rad`
maximum default-interior epsilon, latent scale multiplier `1024`, latent
absolute cap `32`, mean fraction `3/8`, minimum-standard-deviation cap/divisor
`0.025`/`64`, and maximum-standard-deviation cap/divisor `1.0`/`16`. Every
scalar must be present, numeric, finite and exactly equal to the compiled v10
contract. The runtime independently derives the per-joint operational latent,
mean, standard-deviation and deterministic `T(mean)` output envelopes in
float32 and fails at startup if their nesting or ten-sigma margin is invalid.
The runtime comparison permits one outward float32 ULP at each `T(mean)` edge
to cover the last-bit difference between PyTorch, NumPy and ONNX Runtime
transcendental kernels; the wider guarded actor limit remains strict.

Four strict JSON metadata arrays carry the unrounded 18-joint raw soft limits
and guarded actor limits. The runtime independently re-derives all four from
its compiled-in `NEUTRAL_POSE`, action scale and physical soft limits, using the
same float32 boundary arithmetic as training, and requires an exact or
four-float32-epsilon-tight value match (at most `4.7684e-7 rad` with the current
unit action scale). That narrow allowance covers only the operation-ordering
difference between MuJoCo/MjLab limit resolution and the runtime's
already-compiled soft limits; three-decimal metadata remains far outside it.
Every actor interval must strictly contain zero and be strictly inside its
corresponding raw soft interval. Missing, rounded, non-finite or materially
tampered values therefore fail closed.

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
stored in the next observation. This exactly matches v10 training and prevents
unbounded network output from feeding back while a servo target is saturated.
Every live ONNX output must be inside both the guarded actor interval and the
narrower deterministic `T(mean)` envelope. The resulting absolute target must
also be strictly inside the robot's compiled soft limit. A violation rejects
the complete tick before any of its 18 targets are written; it is not silently
saturated. The absolute clip remains in the matched observation calculation,
but it is a no-op for every valid v10 output and is never widened by metadata.

Contract v10 permits one migration chain: a full-state transfer from the exact
accepted contract-v9 `model_1499.pt`. Every later canonical v10 stage must
inherit the same migration ledger. Its eight flattened ONNX metadata fields are
fixed as follows:

- `migration_source_checkpoint_sha256` =
  `de8b6139872179679a16d72f3007f6d96cf65c97fa88565841eaa5f89511a65f`
  and `migration_source_checkpoint_iteration` = `1499`;
- `migration_source_training_provenance_sha256` =
  `f09f5580f03d3e38deef4916db7aea3bd8b1f683dc02a75079d24dd17923fce9`;
- `migration_source_tree_sha256` =
  `61a9fc7b1fe10436c0f033f89710e33e9e5470716d94794f110731c18e7d792a`;
- `migration_source_gate_sha256` =
  `acb2e39411155d70aed2b18561a243bd8ad20eef56e0942d2dafbb4f96f39b7c`;
- `migration_state_transfer` =
  `actor_critic_optimizer_moments_iteration_common_step_v1`;
- `migration_source_optimizer_learning_rate` =
  `7.593750000000002e-05`, and `training_fixed_learning_rate` = `1e-5`.

That lineage also remains bound to the inherited safe-velocity checkpoint at
iteration `500`, SHA-256
`416a8b16f7f7980822e4e1df81ffaf9515bc18a246e6fc257405a2c46ceece93`,
and its acceptance receipt SHA-256
`e68701b11774dd30c8e45a2fd89614a2e4423a9486d01a0d936f0fa6fb760492`.
This records the required transfer of the actor, critic, optimizer moments,
iteration and common-step state before the optimizer rate is fixed at `1e-5`;
it is not an actor-only warm start.

Deployment accepts only `canonical_v10_stage` provenance for the final
`10000->15000` stage and its parity-gated `model_14999.pt` checkpoint
(iteration `14999`, `15000` completed updates). It also requires a schema-3
acceptance receipt with status `pass`, boundary `15000`, evaluator revision
`microban_teleop_deterministic_evaluator_v10_1`, acceptance revision
`microban_teleop_acceptance_v10_1`, and exactly three nominal plus three
moving-HMD reports. The receipt must bind the checkpoint and training-
provenance hashes, plus the recipe revision recorded in the ONNX. Diagnostic,
canary, intermediate,
`model_pristine.pt` and direct v9 exports remain non-deployable. Without
`--acceptance-receipt`, the exporter records `deployment_accepted=false`, which
the runtime rejects. Keep `--require-final-acceptance` in the deployment command
so a missing receipt fails the export instead of producing that diagnostic
artifact.

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

If `pico_teleop.onnx` is absent or rejected, holding the left trigger continues
with normal walking and records the fallback reason in
`PolicySelectableWalkMove.fallback_reason`.
An atomically replaced file is parsed in a background thread and becomes eligible
on a later trigger activation; the process does not need to restart. Construction
is behind the injected `LearnedMoveFactory`, so a future policy-contract parser can
be added without changing the fallback state machine.

Camera transport is deliberately outside this decision. A stale/missing frame,
invalid calibration/FOV/IPD, or passthrough failure may degrade the HMD view, but
does not remove `walk`, zero the joystick, or affect policy inference. Only loss of
the controller/network command stream invokes the existing watchdog: it returns a
neutral `UserInput`, releases locomotion, and requires a fresh trigger release
before reconnection can move again.

## Body-target reference

Training defines hand/foot commands as offsets from an episode-reset reference
in the robot trunk frame (`+X` forward, `+Y` left, `+Z` up). The PICO bridge must
capture the matching body-tracker reference at explicit policy enable/reset and
keep it fixed. It must never send the absolute human pelvis-to-limb positions.
Stale/jumping tracking clears the learned target calibration and sends no trusted
hand/foot target. The robot receiver then keeps the held trigger and joystick but
downgrades that activation to the legacy walking actor. A fresh learned-policy
activation is considered after the left trigger is released. The robot receiver
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
out-of-range or moving-both-feet snapshot clears both target pairs, marks the
learned channel degraded and preserves the packet's `walk` deadman plus velocity
for the legacy actor. It never turns an optional tracker fault into loss of basic
joystick locomotion.

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
5. Test one axis and small target at a time, then trigger-release return to home
   and tracking timeout fallback. If a get-up move is installed locally,
   also verify its scheduler ownership transition.
6. Only after those pass, test feet on the floor at reduced command ranges.

The mounting quaternion exists in software, but its gyro-axis/sign mapping has
not yet been physically certified on this robot. That hardware check is the
remaining gate between a trained artifact and safe free-standing deployment.
