# Installed policies (forward-lean-home branch)

This branch runs the forward-lean HOME (trunk 10 deg forward, COM over the
soles; `src/constants.py` NEUTRAL_POSE / HOME_ROOT_QUAT_WXYZ). Every policy
here must be trained and exported at that HOME.

| file | status |
| --- | --- |
| `getup.onnx` | installed: forward-lean get-up contract v6, SHA-256 `ce6cdc0489b451b32a35c3a791830ca123cadb308f3b2f4eaf1019c3721678bf` (mjlab_microban lean stage-5 `model_21495`) |
| `walk.onnx` | **not installed**: the forward-lean walking policy (walk contract `v4_forward_lean_home_servo_range`) is still being trained |
| `pico_teleop.onnx` | **not installed**: the forward-lean contract-v12 PICO policy (recipe `..._receiver_box_hands_v17` or the pose-release `..._receiver_box_hands_active_hand_arm_pose_release_v18`) has not been trained |

The centered upright-HOME `walk.onnx` (walk contract v3) and `pico_teleop.onnx`
(centered pose-release v12, SHA-256 `ce343503...`) live on
`feature/neck-roll-pitch-camera`. They were removed here on purpose so that the
runtime fails closed (missing file) instead of running a policy trained at
another HOME. Walking, GC300, the PICO walk fallback and PICO teleop do not
start until the lean models are installed.

When installing the lean models:

1. Install the lean `walk.onnx` exported by mjlab_microban
   `export_walk_onnx.py` (contract v4; `WalkMove` checks it on load).
2. Pin the lean walking source in `src/moves/pico_hybrid.py`
   (`EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256`, `..._ITERATION`,
   `EXPECTED_V12_LEGACY_PROBE_SHA256`; currently unmatchable `"0" * 64`
   placeholders). The v12 package's walk-fallback identity
   (`microban_walk_fallback_onnx_sha256`) hashes the installed `walk.onnx`, so
   the PICO package must be built against the installed lean `walk.onnx`.
3. Package the lean PICO policy with the forward-lean packager (v7) against
   this repository, install it as `pico_teleop.onnx` and run
   `make teleop-validate` (tools/validate_pico_policy.py).
