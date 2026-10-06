# Installed policies (forward-lean-home branch)

This branch runs the forward-lean HOME (trunk 10 deg forward, COM over the
soles; `src/constants.py` NEUTRAL_POSE / HOME_ROOT_QUAT_WXYZ). Every policy
here must be trained and exported at that HOME.

| file | status |
| --- | --- |
| `getup.onnx` | installed: forward-lean get-up contract v6, SHA-256 `ce6cdc0489b451b32a35c3a791830ca123cadb308f3b2f4eaf1019c3721678bf` (mjlab_microban lean stage-5 `model_21495`) |
| `walk.onnx` | installed: forward-lean walk contract `v4_forward_lean_home_servo_range`, SHA-256 `b33cd9ea7dbebbfe4543c0bb616a54d9ba713ded1ffdda0891b79dad09e2c1d2` (mjlab_microban `forward-lean-v2` `artifacts/walk_v4_forward_lean_home_servo_cont2_29000.onnx`, run `2026-10-05_06-14-02_lean_walk_cont2` `model_29000`, checkpoint SHA-256 `a7c28c8abaf038d85c9c773bdaa2cc6bbe3c9bb2af6135622a4949c38a0401ff`); pinned as `EXPECTED_WALK_FALLBACK_SHA256` in `tools/validate_pico_policy.py` |
| `pico_teleop.onnx` | **暫定版（ユーザー承認の waiver、ゲート合格版ではない）**: SHA-256 `ed656a2a479dc1d457d7370a4e237c68048197f064cc843dd1df9d9681a31f9e`、mjlab_microban `forward-lean-v2` run `2026-10-06_01-05-06_lean_v12_pr_10100_to15000` `model_14999`（checkpoint SHA-256 `b41a6cb2...`）。既知の制限: 斜め左前＋左旋回で左手を伸ばすと左方向にほとんど進まない（-0.017 m/s、必要値 0.02 m/s）。修正版モデルができ次第置き換える。詳細と削除手順は `docs/pico_teleop_v12_runtime.md` の「暫定版 PICO モデル」節 |

The centered upright-HOME `walk.onnx` (walk contract v3) and `pico_teleop.onnx`
(centered pose-release v12, SHA-256 `ce343503...`) live on
`feature/neck-roll-pitch-camera`. They were removed here on purpose so that the
runtime fails closed (missing file) instead of running a policy trained at
another HOME. The lean `walk.onnx` is installed (walking, GC300 and get-up
recovery run). 2026-10-06 から PICO teleop は上表の暫定版 `pico_teleop.onnx` で動きます。

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
   済（2026-10-06、暫定版）: ユーザー承認の waiver 付きパッケージを導入し、
   `tools/validate_pico_policy.py` が合格。修正版ができたら同じ手順で置き換える。
