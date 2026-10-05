# Installed policies (forward-lean-home branch)

This branch runs the forward-lean HOME (trunk 10 deg forward, COM over the
soles; `src/constants.py` NEUTRAL_POSE / HOME_ROOT_QUAT_WXYZ). Every policy
here must be trained and exported at that HOME.

| file | status |
| --- | --- |
| `getup.onnx` | installed: forward-lean get-up contract v6, SHA-256 `ce6cdc0489b451b32a35c3a791830ca123cadb308f3b2f4eaf1019c3721678bf` (mjlab_microban lean stage-5 `model_21495`) |
| `walk.onnx` | installed: forward-lean walk contract `v4_forward_lean_home_servo_range`, SHA-256 `b33cd9ea7dbebbfe4543c0bb616a54d9ba713ded1ffdda0891b79dad09e2c1d2` (mjlab_microban `forward-lean-v2` `artifacts/walk_v4_forward_lean_home_servo_cont2_29000.onnx`, run `2026-10-05_06-14-02_lean_walk_cont2` `model_29000`, checkpoint SHA-256 `a7c28c8abaf038d85c9c773bdaa2cc6bbe3c9bb2af6135622a4949c38a0401ff`); pinned as `EXPECTED_WALK_FALLBACK_SHA256` in `tools/validate_pico_policy.py` |
| `pico_teleop.onnx` | **not installed**: the forward-lean contract-v12 PICO policy (recipe `..._receiver_box_hands_v17` or the pose-release `..._receiver_box_hands_active_hand_arm_pose_release_v18`) has no passing build (2026-10-06): the fresh pose-release chain `lean_v12_pr_*` passed its 10000 and 10100 gates, but every 10100 -> 15000 retrain (seeds 42, 42, 43, 44) and all six final rescues failed the unchanged 14999 final gate (mixed_forward_left twist/falls), so nothing was packaged. A changed 10100 -> 15000 training recipe is needed before a lean PICO package can exist |

The centered upright-HOME `walk.onnx` (walk contract v3) and `pico_teleop.onnx`
(centered pose-release v12, SHA-256 `ce343503...`) live on
`feature/neck-roll-pitch-camera`. They were removed here on purpose so that the
runtime fails closed (missing file) instead of running a policy trained at
another HOME. The lean `walk.onnx` is installed (walking, GC300 and get-up
recovery run); PICO teleop does not start until the lean `pico_teleop.onnx` is
installed.

When installing the lean models:

1. Done (2026-10-05): the lean `walk.onnx` exported by mjlab_microban
   `export_walk_onnx.py` (contract v4; `WalkMove` checks it on load).
2. Done (2026-10-05): the lean walking source is pinned in
   `src/moves/pico_hybrid.py` (`EXPECTED_V12_LEGACY_SOURCE_CHECKPOINT_SHA256`
   `a7c28c8a...`, `..._ITERATION` 29000, `EXPECTED_V12_LEGACY_PROBE_SHA256`
   `ebcf4554...`, the fresh pose-release chain's probe receipt). The v12
   package's walk-fallback identity
   (`microban_walk_fallback_onnx_sha256`) hashes the installed `walk.onnx`, so
   the PICO package must be built against the installed lean `walk.onnx`.
3. Package the lean PICO policy with the forward-lean packager (v7) against
   this repository, install it as `pico_teleop.onnx` and run
   `make teleop-validate` (tools/validate_pico_policy.py).
