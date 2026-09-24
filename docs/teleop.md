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
- `pico_teleop`は固定contract tag、安全余裕0.8、左右の手足pair、80% target範囲をrobot側でも検証。
  違反packetは速度・targetを即時clearし、左triggerのreleaseまで再armしない。
- 300 ms無通信で速度ゼロ・全move解除。
- IMUがinvalid、100 ms超のstale、または非finiteになった瞬間から全21軸を実測角で保持。
  750 ms継続時はcontrol loopを終了して全motorをtorque-off。
- IMU異常後は、左triggerのreleaseを再観測するまで歩行を再許可しない。
- 歩行解除時、18 policy軸を0.8秒のsmoothstepで初期姿勢へ戻してからKPを戻す。
- HMD首指令はyaw/roll/pitchのjoint limitとslew limitをrobot側でも適用。

## カメラ

USBステレオカメラ接続後、2通りの配信方法があります。どちらも同じUSBカメラを排他利用するので
同時には動かせません。**デフォルトはH264/UDP(ハードウェアエンコード)を使ってください**。MJPEG/HTTP
はブラウザで手軽に確認したい場合向けに残しています。

### H264/UDP (`camera-stream-udp`, デフォルト・低遅延・高fps)

```bash
sudo apt-get install -y ffmpeg  # ロボット側、初回のみ
make camera-stream-udp
```

Piの`camera-stream-enable`サービスは事前に止めてください(`ssh microban sudo systemctl stop microban-camera`)。
実行したマシンのIPを自動検出し(`teleop-run`と同じ`SSH_CONNECTION`の仕組み)、GPUハードウェアH264エンコード
(`/dev/video11`, bcm2835-codec-encode)でエンコードしたMJPEG→H264をUDP/MPEG-TSでそのマシンへ直接送ります。
1280x480で実測ほぼフル60fps(MJPEG/HTTPの32fpsに対して大幅に高い)。受信・表示は同じマシンから別ターミナルで:

```bash
make camera-view-udp
```

止めるときは必ずCtrl+C(SIGINT)で。`kill -9`で強制終了すると`/dev/video11`(GPUのH264エンコーダ)が
再初期化できなくなることがあり、その場合はPiの再起動が必要です。

### MJPEG/HTTP (`camera-stream-enable`, ブラウザで手軽に見たいとき用)

```bash
make camera-stream-enable
```

systemd unitは`http://microban:8080/stream`でMJPEGを公開します。設定はロボット側の
`/etc/default/microban-camera`です。カメラ未接続の状態でこのinstall targetを実行しないでください。
ブラウザ等どこからでも見れますが、Pi Zero 2Wの2.4GHz WiFi帯域に対してMJPEGは1フレームが大きすぎて、
1280x480でも実測32fps程度が上限です。

### PICO校正済み表示向けTLS latest-snapshot

連続`/stream`はclientがWi-Fi帯域より遅いと古いJPEGがsocket queueへ溜まります。校正済みPICO表示では
uStreamerの最新1枚だけを取得し、撮影時刻headerとJPEGをTLS 1.3で一緒に認証するproxyを使います。
初回だけrobot上で秘密鍵を生成し、公開certificateをPCへcopyします。

```bash
make camera-stream-tls-provision
make camera-stream-tls
```

秘密鍵はrobotの`~/.config/microban-camera-tls/server.key`から出ません。PC側certificateは既定で
`~/.config/microban-teleop/microban-camera.crt`です。`camera-stream-tls`はforegroundで動き、Ctrl+Cで
終了します。既存uStreamerは同時に必要です。proxyが転送するのは`/snapshot`だけで、V4L2 dequeueから
response header生成までのuStreamer monotonic timingとrobot wall clockを改変せず保持します。

詳細は `../microban_teleop/docs/pico4ultra_webxr.md` を参照してください。
