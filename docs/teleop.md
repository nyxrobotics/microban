# PICO 4 Ultra network teleoperation

PICO/WebXR側の実装は隣の `microban_teleop` リポジトリにあります。このリポジトリにはロボット側の
UDP入力、歩行deadman、HMD首3軸制御、カメラstream serviceを置いています。

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
| `MICROBAN_NETWORK_STALE_S` | `0.3` | command watchdog |
| `MICROBAN_NETWORK_ALLOWED_IP` | 未指定 | 指定時、このIPv4送信元だけ許可 |

## robot-side safety

- packetは部分更新ではなく完全snapshot。省略fieldはneutral。
- version/session/sequenceを確認し、同一sessionの古いpacketを破棄。
- 起動・timeout・新session後、一度`walk`なしpacketを受けるまで歩行を禁止。
- malformed JSON、NaN/Inf、未知moveを破棄し、受信threadを継続。
- 300 ms無通信で速度ゼロ・全move解除。
- IMUがinvalid、100 ms超のstale、または非finiteになった瞬間から全21軸を実測角で保持。
  750 ms継続時はcontrol loopを終了して全motorをtorque-off。
- IMU異常後は、左triggerのreleaseを再観測するまで歩行を再許可しない。
- 歩行解除時、18 policy軸を0.8秒のsmoothstepで初期姿勢へ戻してからKPを戻す。
- HMD首指令はyaw/roll/pitchのjoint limitとslew limitをrobot側でも適用。

## カメラ

USBステレオカメラ接続後:

```bash
make camera-stream-enable
```

systemd unitは`http://microban:8080/stream`でMJPEGを公開します。設定はロボット側の
`/etc/default/microban-camera`です。カメラ未接続の状態でこのinstall targetを実行しないでください。

詳細は `../microban_teleop/docs/pico4ultra_webxr.md` を参照してください。
