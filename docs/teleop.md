# PICO 4 Ultra network teleoperation

PICO側のPC bridgeは隣の `microban_teleop` リポジトリにあります。このリポジトリにはロボット側の
UDP入力、歩行deadman、HMD首3軸制御を置いています。

## 起動

最初は実機ではなくMuJoCoで確認します。

```bash
make teleop-sim
```

カメラ校正と支持治具上のacceptanceが完了した後だけ、実機を起動します。

```bash
make teleop-run
```

これはSSH接続元IPだけを許可して、UDP port 5555でfull-snapshot JSONを受信します。手動起動なら:

```bash
MICROBAN_INPUT=network \
MICROBAN_NETWORK_ALLOWED_IP=192.168.8.177 \
PYTHONPATH=src .venv/bin/python src/main.py
```

環境変数:

| 変数 | 既定値 | 内容 |
|---|---:|---|
| `MICROBAN_NETWORK_PORT` | `5555` | UDP受信port |
| `MICROBAN_NETWORK_ALLOWED_IP` | 未指定 | 指定時、このIPv4送信元だけ許可 |

## robot-side safety

- packetは部分更新ではなく完全snapshot。省略fieldはneutral。
- version/session/sequenceを確認し、同一sessionの古いpacketを破棄。
- 起動・新session・許可する送信元の変更の後、一度`walk`なしpacketを受けるまで歩行を禁止。
- 受信が途切れても、最後に受けた操作を保つ。止めるのはB（トルクOFF）のpacketだけ。
- malformed JSON、NaN/Inf、未知moveを破棄し、受信threadを継続。
- `pico_teleop`は固定contract tag、安全余裕0.8、左右の手足pair、80% target範囲をrobot側でも検証。
  違反packetは速度・targetを即時clearし、左triggerのreleaseまで再armしない。
- IMUがinvalid、100 ms超のstale、または非finiteになった瞬間から、直前の関節目標とトルクを保持し、
  ポリシー出力を止める。
- IMU異常後は、左triggerのreleaseを再観測するまで歩行を再許可しない。
- 歩行解除時、18 policy軸を0.8秒のsmoothstepで初期姿勢へ戻してからKPを戻す。
- HMD首指令はyaw/roll/pitchのjoint limitとslew limitをrobot側でも適用。

