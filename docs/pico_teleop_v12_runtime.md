# Contract-v12 PICO policy runtime

The robot accepts contract-v12 only as an optional learned locomotion policy.
The pinned `walk.onnx` policy remains the availability baseline. A missing
artifact, rejected metadata, ONNX load error, or learned-policy start error
makes `PolicySelectableWalkMove` select and start `walk` in that activation's
start cycle; the scheduler first calls `walk.step()` on its next control cycle.
By contrast, an error during an active learned-policy step (including a
malformed observation, non-finite inference result/target, or finite-amplitude
guard violation) starts **and steps** `walk` in the faulting control cycle. The
fallback remains latched until the left-trigger locomotion activation is fully
released, so a repaired tracker or hot-reloaded file cannot switch the gait
mid-stride.

## 実機の全関節ゲート（右手 B / A / R3）

`MICROBAN_INPUT=network` の実機起動は必ず全関節トルクOFF・policy
OFFから始まる。右手Bはいつでも全21関節のトルクをOFFにする。通信が
途切れた場合は直前のサーボ目標とトルク状態を保持し、新しい動作目標を送らない。右手AはトルクをONにする前に
現在角を全servoのgoalへ書き、古いgoalへの跳ねを防いだうえで、policy
出力を使わず0.5 rad/s以下で全関節を `NEUTRAL_POSE` へ移す。

右スティック押し込み（R3）は、Aの状態を一度受信した後に限り、通常の
policy出力と初期姿勢復帰をトグルする。左トリガーを離していても、R3で
policyを有効にすると既存の歩行policyへ速度ゼロを渡して立位を保つ。
左トリガーを押すとPICO teleop policyに切り替わる。policy有効中にAを
押した場合もpolicyを止め、同じ低速初期姿勢復帰へ入る。R3はトルクOFF
中には無視される。B、A、R3の処理は個別moveより上位のschedulerが所有
するため、脚だけでなく腕・首を含む全関節へ一貫して適用される。

## Physical action semantics

Contract v12 is intentionally different from contract v10:

- the actor input is the exact 83-value teleop schema;
- ONNX emits the deterministic mean of the normalized, unbounded legacy
  Gaussian actor;
- each of the 18 outputs is used as the raw joint-position action:
  `target = Microban training default + raw * training scale`;
- no bounded-distribution transform, raw-action clamp, or effective-action
  reconstruction is applied to the policy state;
- the exact float32 raw output becomes observation columns `48:66` on the next
  tick, even if its derived physical target saturates;
- separately, the actuator-facing `MotorCommand` is the continuous clamp of the
  finite derived absolute target to the compiled Microban soft limits.

The runtime validates all 18 values and derived targets for finiteness before
writing any target. It also applies the final-evaluation finite-amplitude guard
described below. A finite target outside a soft limit is not an exception or a
stop: that joint commands the nearest limit while all joints and the raw actor
recurrence continue in the same tick. A numerical failure or an escape from the
separate authenticated gross-amplitude guard remains an availability event
handled by the legacy walk fallback.

The v10 exporter supplies soft-limit metadata and the existing parser requires
it to match the robot constants. The current v12 exporter does not supply those
fields because its learned action contract is `action_clip_semantics=none`. If a
v12 artifact does contain both soft-limit vectors, the parser requires the same
exact match; if neither is present, the robot uses its compiled vectors. In both
cases construction fails before ONNX inference unless every bound/default/scale
and the global inactive neutral form a finite, ordered 18-joint clamp contract.

The allowed **5-degree measured-joint overshoot** is only an offline simulator
stage-gate tolerance: the evaluator may observe a simulated measured joint up to
5 degrees beyond a soft limit. It is not a physical-robot encoder acceptance
tolerance and grants no runtime permission. It never expands a commanded soft
limit: the physical runtime clamps every actuator target to the compiled limit,
without `soft_limit +/- 5 degrees` headroom.

## Trigger-only selection

Both current bridge paths always serialize
`locomotion_policy="pico_teleop"`. The left X/WebXR primary state is ignored.
Holding the left trigger puts `walk` in `active_moves` and exposes
the learned-policy body targets. R3 also keeps the baseline `walk` actor active
at zero velocity while the left trigger is released. The right trigger puts
`hmd_head` in `active_moves` for neck tracking:
every valid PICO frame keeps `pico_arms` active, with
`arm_tracking_enabled=true` and a paired bounded joint target while held, or
`arm_tracking_enabled=false` and the exact authenticated PICO arm HOME while
released. The right grip supplies `head_yaw_front` upstream and is not inferred
from either trigger by the robot. `tests/test_policy_selector.py`,
`tests/test_network_input.py`, and `tests/test_pico_arms.py` cover these paths.

The right trigger is only an enable gate, never a pose clutch. The bridge
recomputes each hand every frame from the current HMD-origin-to-controller
absolute vector in the current HMD axes; it does not capture or subtract the
controller pose at press or release. The robot authenticates the independent
direct-overlay contract
`microban_hmd_absolute_arm_fk_live_box_pitch100_roll120_elbow110_v1`: shoulder
pitch `-100..+100 degrees`, left/right shoulder roll `+10..+120` /
`-120..-10 degrees`, and elbow `-110..0 degrees`. This does not widen or alter
the learned v12 actor's original narrow Cartesian hand-observation contract.

Release is also the explicit re-arm boundary. After startup, transport timeout,
motion inhibition, tracker/contract degradation, or learned-policy fault, do not
resume from a trigger that was already held: send and receive a released-trigger
snapshot, then press again. The new pressed snapshot must be fresh and must
contain the exact v12 target contract plus complete feet and hands expressed
from the bridge's fixed policy-session calibration origin. Calibration is a
bridge-side prerequisite: the robot validates freshness, contract, completeness
and envelopes, but cannot prove the physical origin was calibrated correctly.
Do not re-arm with an uncalibrated/reused absolute pose. A stale packet,
incomplete target pair, or target outside the wire envelope cannot re-arm v12;
the receiver keeps the joystick-triggered legacy walk path and the selector
keeps that fallback latched until another release/press boundary.

The arm deadman has its own release-to-rearm latch. A new/reconnected session
cannot begin with the right trigger held. During a brief controller-tracking
dropout the bridge repeats its last bounded arm target, so the robot neither
requires another press nor uses a partially invalid side. An expired transport
snapshot, malformed/out-of-range paired target, safety inhibition, or session
loss removes `pico_arms`, clears the cached target, and returns those six joints
to global neutral. A subsequent valid released frame commands the robot-local
PICO arm HOME and rearms the next press. The overlay is registered after
`walk`, so it replaces only shoulder pitch/roll and elbow commands while the
learned or fallback actor continues to own the legs.

## Training HOME versus inactive neutral

These are intentionally distinct software conventions:

- `NEUTRAL_POSE` in the physical runtime has shoulder pitch `+10 degrees`. Git
  blame traces it to Microban commit `f27a9e29` (2026-05-22, *Neutral pose and
  motor signs*). That commit records no measurement provenance. The robot MJCF
  contains only the shoulder-pitch joint range (`-pi..+pi`) and no HOME/keyframe
  value, so XML is not authority for `+10 degrees`.
- MjLab commit `0119357e` changed both training shoulder pitches from
  `+10 degrees` to `0 degrees`; current `HOME_FRAME`, PICO contract defaults and
  the pinned `walk.onnx` `default_joint_pos` metadata all use `0 degrees`.

The PICO actor therefore continues to use its local 0-degree training HOME.
Left-trigger locomotion release interpolates all 18 policy joints to the
unchanged global `NEUTRAL_POSE`, including shoulder pitch `+10 degrees`, before
declaring the walk move inactive. Independently, a valid right-trigger release
keeps the arm overlay active and commands its authenticated arm HOME (shoulder
pitch `0 degrees`, roll `+10/-10 degrees`, elbow `-20 degrees`) after the walk
output. Only loss of the PICO arm session returns those six joints to global
neutral. Thus neither release path creates the former one-tick unowned target
jump.

## Deployment admission

The canonical staged route is
`601 -> 3000 -> 3100 -> 7000 -> 7100 -> 10000 -> 10100 -> 15000` completed
updates. The 100-update boundaries are mandatory activation canaries: 3,100
checks the first staged activation, 7,100 checks HMD/hand activation, and 10,100
checks full-body/foot activation. Every boundary checkpoint, every activation
canary, every interrupted checkpoint, and every preview is simulator-only. The
physical parser accepts only `model_14999.pt` at 15,000 completed updates, with
`deployment_accepted=true` and a passing canonical final-stage gate.

The ONNX metadata must bind the final checkpoint to all of the following:

- the pinned legacy velocity checkpoint SHA-256 and iteration;
- the pinned raw-action 9-by-300 probe SHA-256;
- the exact 63-to-83 semantic column map, 20 teleop-only columns, actor
  topology, frozen normalizer and frozen legacy-tensor contract;
- recipe
  `legacy_velocity_model14999_staged_mask_reachable_fk_elbow_minus10_raw_actions_v5`
  and gradient
  schedule `freeze_extra_to7000_then_hmd_hand_to10000_then_all_v1`; the retired
  pre-FK recipes are never accepted by the robot;
- bootstrap mapping
  `normalized_legacy_velocity_63_to_teleop83_reachable_fk_elbow_minus10_v4`,
  physical target-position normalization, and the complete `hand_target_fk`
  JSON contract (joint box, HOME, reachable AABB, scale, wire/runtime limits and
  named evaluator points);
- the final nine-scenario locomotion report and schema-v2 stage-gate identities;
  the tracking profile must be either canonical
  `full_body_reachable_performance_perturbation_v2` or the explicitly
  lineage-bound deadline-final profile
  `deadline_full_body_hand_rms35mm_p95_70mm_foot_rms50mm_p95_80mm_perturbation_v2`;
  every other
  profile is rejected;
- the tracking report's exact 18-joint v12, pinned-source and learned-minus-
  source raw-action extrema, plus the versioned runtime guard derived from
  those hash-bound values;
- a 64-sample full-83-column PyTorch/ONNX parity check (the 20 new columns must
  not be zeroed), plus the independent 10,000-sample zero-extra legacy parity;
- the exact observation term order, 21 observation joints, 18 action joints,
  defaults, scale, frames, units, body-target limits, raw previous-action and
  no-clip semantics.

`tools/validate_pico_policy.py` performs this parser check and a fixed 16-input
ONNX Runtime CPU smoke without opening motor or network interfaces. The same
command also authenticates the installed fallback as SHA-256
`10c58a63c66337669c3d4c588732d541a6a07eea3291c0401f79893c7f60f15d`,
requires its graph to be float32 `obs[1,63] -> actions[1,18]`, and runs a
separate fixed 16-input smoke with `CPUExecutionProvider` only:

```bash
PYTHONPATH=src uv run --locked python tools/validate_pico_policy.py \
  /path/to/final-pico-teleop-v12.onnx
```

For v12, the learned-policy load smoke checks the fixed output shape, float32
finiteness and the authenticated finite-amplitude guard on every fixed sample.
The validator reports that guard and an explicit `walk_fallback` record containing
the fallback path, digest, tensor contract, providers and smoke result.
For v12 it also hashes the validator, learned-policy contract parser, selector,
walk runtime, direct-arm runtime and contract, network input parser and data
contract, production entrypoint, scheduler, configuration, `uv.lock`, and
fallback ONNX; all 13 hashes must equal the identities embedded by the packager.
This binds both the six-joint overlay and its ordering after the learned walk
move. The validator reports that complete identity and rehashes it after both
CPU smokes so a source changed during admission is rejected.
The production `PicoHybridMove` performs the same embedded-identity check at
load and repeats it after its fixed CPU smoke. Therefore the `teleop-run`
rsync cannot start a learned actor against different runtime bytes; the
selector rejects that actor and retains the authenticated `walk.onnx` fallback.
Contract-v10 validation remains a separate branch with its existing bounded-
action and effective-action checks unchanged.

### Finite-amplitude fallback guard

Finiteness alone cannot classify an abnormally large finite actor result. A
soft-limit or MJCF hard-limit **reject** is not compatible with the source policy:
its accepted 9-by-300 probe contains hypothetical raw targets as much as
2.327371 rad beyond a soft limit while the simulated measured joints remain
within every soft limit. The actuator-facing continuous clamp handles those
finite soft-limit excursions without rejecting the tick; the evidence-derived
guard below remains only a gross numeric-anomaly detector.

The final tracking report therefore records, over every acceptance scenario and
step, per-joint minima, maxima and absolute maxima for the v12 raw action, the
pinned source actor on the same shared 63 observations, and their delta. The
deployment packager copies those values and their report SHA-256 into ONNX
metadata. For each joint it must calculate exactly:

```text
guard_absmax = max(v12_absmax, source_absmax + delta_absmax) * 6.0
```

The factor `6.0` is the deployed versioned deadline engineering margin for a gross finite-
anomaly detector, not a learned-action clamp or a claim about a joint's safe
physical range. The runtime recomputes the formula from the metadata evidence,
requires exact agreement and accepts equality at the boundary. If
`abs(raw_action)` exceeds its joint's guard, it raises before writing any of the
18 targets or updating the previous-action recurrence. `PolicySelectableWalkMove`
then starts and steps the pinned legacy walk policy in that same control cycle
and latches the fallback until trigger release.

The deployment ONNX uses these exact metadata keys for that contract:

```text
v12_raw_action_envelope_schema_version=1
v12_raw_action_joint_names_json
v12_raw_action_{min,max,absmax}_json
v12_source_raw_action_{min,max,absmax}_json
v12_learned_source_delta_{min,max,absmax}_json
runtime_raw_action_guard_formula=max(v12_absmax,source_absmax+delta_absmax)*multiplier
runtime_raw_action_guard_multiplier=6.0
runtime_raw_action_guard_absmax_json
runtime_raw_action_guard_semantics=finite_float32_then_per_joint_absmax_else_same_cycle_legacy_fallback_v1
```

`v12_tracking_report_sha256` binds the evidence source, and
`v12_raw_action_joint_names_json` must exactly equal the 18-action policy order.

The final gate must review the resulting 18 numeric bounds before packaging.
Missing/malformed evidence, a changed joint order, inconsistent extrema,
float32 overflow, a formula/factor/semantics mismatch, or a recomputed-bound
mismatch rejects the ONNX at load time.

After packaging, produce a concrete human-review record from the same robot
validator. This command fails unless names and bounds both have exactly 18
entries, prints one named bound per line, and records the validator JSON, the
reviewed table, and all four file digests. Reviewing this table is a release
action; these local files do not replace the hash-bound canonical stage-gate
receipt.

```bash
review_dir=artifacts/pico_teleop_release_review
mkdir -p "$review_dir"
PYTHONPATH=src uv run --locked python tools/validate_pico_policy.py \
  src/agents/pico_teleop.onnx | tee "$review_dir/validator.json"
PYTHONPATH=src uv run --locked python - "$review_dir/validator.json" <<'PY' \
  | tee "$review_dir/runtime_raw_action_guard.tsv"
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
names = report["action_joint_names"]
bounds = report["v12_raw_action_guard"]["absolute_maximum"]
if len(names) != 18 or len(bounds) != 18:
    raise SystemExit(f"expected 18 named bounds, got {len(names)} and {len(bounds)}")
for index, (name, bound) in enumerate(zip(names, bounds, strict=True)):
    print(f"{index:02d}\t{name}\t{float(bound):.9g}")
PY
sha256sum src/agents/pico_teleop.onnx \
  src/agents/walk.onnx \
  "$review_dir/validator.json" \
  "$review_dir/runtime_raw_action_guard.tsv" \
  | tee "$review_dir/SHA256SUMS"
```

Metadata attachment should happen in a final deployment-packaging step after
the sidecar stage gate exists. Attaching metadata changes the ONNX file digest,
so the packager must then rerun ONNX checker, the full-83-column parity check,
and the deployment runtime validator on the final file before atomic install.

The adjacent `mjlab_microban_v8j` repository now provides that exact final-only
path. After the run has reached 15,000 completed updates and its current stage
gate passes, use:

```bash
cd ../mjlab_microban_v8j
scripts/evaluate_microban_teleop_v12_stage.sh <run-name> 14999
scripts/export_microban_teleop_v12_deployment.sh \
  <run-name> ../microban/src/agents/pico_teleop.onnx --force
```

The publisher accepts no intermediate or diagnostic mode. It runs this
repository's `tools/validate_pico_policy.py` with `CPUExecutionProvider` against
the complete temporary artifact and the pinned repository fallback, then
atomically replaces `pico_teleop.onnx` only after both parser/runtime smokes
pass. A failed export or validator keeps the previously installed policy
unchanged.

## Workstation and Raspberry Pi preflight

The repository tracks `uv.lock`; do not regenerate it implicitly on the Pi.
Run this release preflight only from a clean, committed worktree: the following
must print nothing. `make teleop-validate` rsyncs the current filesystem, not a
named commit, and does not itself reject unrelated modified or untracked files.

```bash
git status --porcelain=v1 --untracked-files=all
```

After the final exporter above installs `src/agents/pico_teleop.onnx`, run the
same learned-policy parser and both fixed 16-input ONNX Runtime CPU smokes on the
workstation and on the Pi before opening the motor bus:

```bash
cd ../microban
make teleop-validate HOST=microban
```

`teleop-validate` performs these fail-fast steps in order:

1. validate the installed learned ONNX and pinned `walk.onnx` with the
   workstation checkout;
2. rsync the checkout, including the pinned `uv.lock`, to `microban`;
3. run `uv sync --frozen` on the Pi;
4. validate both installed paths with the Pi's CPU runtime.

The target does not start `src/main.py` and does not access the motor bus. Only
after it passes should the control loop be launched:

```bash
make teleop-run HOST=microban
```

The direct controller-arm overlay does not wait for a newly trained locomotion
artifact. With the PICO app open in the foreground, start the pinned
XRoboToolkit service and live bridge from the sibling `microban_teleop`
checkout in two additional terminals:

```bash
./scripts/run_xrobot_service_local.sh
./scripts/run_twist2_local.sh teleop --send --robot microban
```

Hold the right trigger to command both arms, whether standing or walking;
release it to command the exact PICO arm HOME. The left trigger remains the
independent locomotion deadman. These commands do not replace the supported-
robot physical acceptance checks required before unrestricted operation.

This preflight's scope is artifact admission, CPU load, and two sets of 16 fixed
inference samples. It does not test UDP freshness/session re-arm, live PICO
calibration, tracker loss, MuJoCo dynamics, physical motor commands, or a fall.
It also does not deliberately fault the final ONNX on the Pi. The offline
selector regression injects a learned-policy inference failure while using the
real `WalkMove` and pinned `walk.onnx`; it proves that the faulting cycle starts
and runs the fallback and produces 18 finite targets without opening the motor
bus. This remains an offline regression, not a hardware integration test.

Likewise, `make teleop-sim` by itself only starts the robot repository's MuJoCo
process with a network input socket. It neither generates calibrated v12 body
targets nor supplies a trigger activation, so that command alone does not
exercise `pico_teleop` selection and is not v12 body-target validation. Use the
canonical training-repository stage/canary evaluators for checkpoint validation;
use an explicitly paired PICO bridge plus simulator only as a separate live
integration observation.

If either validator fails, do not bypass it. The learned policy remains
optional: removing or withholding `src/agents/pico_teleop.onnx` leaves the
pinned `walk.onnx` fallback available under the same left-trigger control, but
an absent or altered fallback is now a failed deployment preflight.
