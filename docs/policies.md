# 学習したモデル（歩行・起き上がり）の契約と導入

実機は学習リポジトリ（mjlab_microban）で学習した 2 つのモデルを使う。

| モデル | ファイル | 使う動作 |
|---|---|---|
| 歩行 | `src/agents/walk.onnx` | `WalkMove`（キーボード・ゲームパッド） |
| 起き上がり | `src/agents/getup.onnx` | `GetupMove`（転倒後の起き上がり、立位の保持） |

どちらも `src/policy_contract.py` の契約 `microban-policy-1` だけで照合する。
学習側との取り決めの全文は学習リポジトリ側と共有している契約の文書（metadata のキーの一覧）にあり、ここでは実機側の要点を書く。

## 契約
- 版: metadata の `microban_policy_contract` が `microban-policy-1`。観測・行動・目標の意味を変えたら、両方のリポジトリで手で上げる。
- 種類とレシピ: `microban_policy_kind`（walk / getup）と `microban_recipe`。受け付けるレシピ id は `policy_contract.RECIPES` に 1 つずつ。
- HOME: `home_pose` の全精度のスタンプと `default_joint_pos` が `config/home_pose.yaml` と 1e-6 以内で一致すること。別の HOME で学習したモデルは拒否する。
- 目標の式: どのモデルも `target = clip(HOME + raw * 1.0, -π, +π)`（18 関節、ソフトウェアのクリップなし、±π はサーボの 1 回転の範囲）。前回の行動の観測は生の出力。
- 観測の並び: `observation_schema_json`（walk 63、getup 60）と `observation_joint_names`。
- 合格: `gate_status=pass` と判定レポートの sha256。チェックポイントの名前と sha256。

実機側のソースのハッシュは使わない。互換性は契約の版と次の起動時の自己テストで確かめる。

## 起動時の自己テスト
各モデルには、書き出したチェックポイントの評価のロールアウトから取った観測 8〜64 行と、それに対する学習時の actor の出力が入っている。
実機は読み込むときにそれを ONNX Runtime に通し、行ごとに `max|ORT − 記録| ≤ 1e-4 + 1e-5 × max|記録|` を確かめる。
記録の観測は、重力のノルムが 1、関節角が可動範囲 + 5° の内側、関節速度が 12.1 rad/s 以下であること。

## manifest と導入
導入の道具（学習リポジトリの `scripts/retrain_all_for_home.py`）は次だけを書き、実機の .py もテストも書き換えない。

- `src/agents/walk.onnx`、`getup.onnx`
- `src/agents/manifest.json`: 契約、HOME の tag、学習側の commit、試運転かどうか、各ファイルの sha256 とチェックポイントの sha256
- `config/home_pose.yaml`（schema 2）

実機はモデルを読むたびに、ファイルの sha256・チェックポイントの sha256・HOME の tag を manifest と照合する。
試運転のパッケージ（`dry_run`）は環境変数 `MICROBAN_ALLOW_DRYRUN_POLICY=1` のときだけ受け付ける（導入の検証とテストの実行のためだけ）。

検証（モーターのバスは開かない）:

```bash
PYTHONPATH=src uv run --locked python tools/validate_policies.py src/agents
PYTHONPATH=src uv run --locked --with pytest python -m pytest -q tests
make teleop-validate HOST=microban   # 作業機と Raspberry Pi の両方で validate_policies を実行
```

`make teleop-validate` は作業ツリーをそのまま rsync するので、`git status --porcelain=v1 --untracked-files=all` が空の状態で実行する。

契約に合わないモデルの扱い:
- 歩行: `WalkMove` が例外を出し、そのランタイムは起動しない。
- 起き上がり: `GetupMove.model_ready` が False になり、転倒時はゆっくり初期姿勢へ戻るだけになる。

## サーボのゲイン
- 学習したモデル（歩行・起き上がり）が動いている間は、頭と首を含む全 21 関節を P125（`KP_RL`。学習の BAM と同じ値）。
- 学習したモデルを使わない静止（A の初期姿勢、起き上がり後に歩行が始まらないままの静止、起き上がりモデルが使えないとき、学習した動作が終わって HOME に戻ったあと）は全関節 P900（`KP_HARDWARE_NEUTRAL`）。
- `RobotController.sync_write_kp` は前回そのサーボに送った値と違うものだけを送る。起動直後は値が分からないので、`main.py` の最初の書き込み（全関節 P900）は必ず全サーボに届く。サーボが通信から外れて戻ったとき（再起動で RAM のゲインが戻った可能性がある）は、最後に指定した値をそのサーボに送り直す。
