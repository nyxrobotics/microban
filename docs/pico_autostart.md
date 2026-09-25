# PICO実機テレオペのロボット自動起動

ロボット側はsystemd system serviceで、PICO network runtimeとcamera latest-snapshot mTLS proxyを
自動起動できます。既存のgamepad headless serviceとは競合させず、送信元PCを1つのIPv4 addressへ固定します。

## 初回設定

PC側のIPv4 addressを確認し、ロボットへ同期済みのrepositoryで次を実行します。

```bash
cd /home/user/microban
sudo bash systemd/configure-pico-services.sh install --allowed-ip 192.168.8.177
sudo bash systemd/configure-pico-services.sh enable
```

`install`はunitを検証・配置するだけでprocessを起動しません。cameraを使わないcontrol-only構成では
`--without-camera`を付けます。camera付き構成は既存`microban-camera.service`と、事前に
`systemd/provision-camera-tls.sh`で生成したrobot側identityを要求します。

## 管理

```bash
sudo bash systemd/configure-pico-services.sh status
sudo bash systemd/configure-pico-services.sh health
sudo bash systemd/configure-pico-services.sh logs
sudo bash systemd/configure-pico-services.sh restart
sudo bash systemd/configure-pico-services.sh disable
sudo bash systemd/configure-pico-services.sh enable
sudo bash systemd/configure-pico-services.sh uninstall
```

`enable`は競合する既存の`microban-gamepad.service`を停止・無効化してからPICO runtimeを起動します。これにより、
両方がboot対象になった場合の起動順依存を避けます。PICO側の`disable`だけではgamepad serviceを再有効化しません。
gamepadへ戻す場合はPICO側を`disable`した後、既存のgamepad用設定手順で明示的に有効化してください。

`status`は設定した構成のsystemd状態を表示し、`health`はenable/active、旧gamepadとの競合、UDP 5555、
camera構成時のHTTP 8080/TLS 8443 listenerまで検査して、未準備なら非zeroで終了します。cameraを使わない
`--without-camera`構成では、存在しないcamera unitを異常扱いしません。runtimeとTLS proxyは一時的な
network、camera、serial障害が長時間続いても再起動上限で停止せず、復旧を待ち続けます。

PICO runtimeは`MICROBAN_SERIAL_HOLD_LAST_ON_ERROR=1`で動きます。21台のどのサーボでも位置・速度などの
同期読取に失敗した周期は、最後の正常な観測値と送信済み目標を保持し、新しい動作目標を送信しません。
送信でエラーが出た場合も動作計算を止め、直前の送信済み目標を再送して通信復旧を待ちます。欠けた読取値や
非有限値も通信エラーとして扱います。Aボタンの初回目標送信など、hardware gate内の送信が失敗した場合も
次の周期で再試行します。連続して0.3秒以上エラーが続いた後は、復旧時に歩行トリガーの離し直しが必要です。
通信エラーだけを理由に全関節のトルクを切る処理は行いません。

`disable`はruntimeを先に停止し、その終了処理と独立したall-joint torque-off helperを実行してからcamera proxyを
停止します。`uninstall`は既存のuStreamer service、camera証明書・秘密鍵、学習済みONNXを削除しません。
gamepad headless unitとはsystemd上でも競合させています。手動runtimeのPIDが既にmotor busを所有している場合も、
serviceのpre-start helperはそのsessionへ割り込んでtorqueを変更せず、起動を失敗させます。

## 起動時の状態遷移

serviceは`MICROBAN_INPUT=network`と`MICROBAN_START_TORQUE_OFF=1`を固定し、通常runtimeの前にも独立した
all-joint torque-offを実行します。したがってboot、service restart、PC/PICO再接続だけではtorqueが入りません。

1. 起動直後: 全関節torque OFF、policy OFF。
2. PICO右手A: torque ON、policyはOFFのまま、全関節を低速で初期姿勢へ移動。
3. 右stick押し込み: 初期姿勢待機とpolicy有効をtoggle。
4. PICO右手B: 即時に全関節torque OFF、policy OFF。

UDPやPICO入力が途切れた場合は、直前の全関節目標とトルク状態を保持して新しい動作目標の送信を止めます。
PC bridgeが送る`torque_enabled=false`だけではトルクを切らず、右手Bを示す
`torque_off_requested=true`を受けたときだけトルクを切ります。再接続後の歩行はトリガーを一度離してから
再開します。長いPICO接続断後はAと右stick押し込みを改めて操作します。

これは物理非常停止の代替ではありません。実機確認は支持治具を使い、別の人が電源へ手を届かせた状態で行って
ください。
