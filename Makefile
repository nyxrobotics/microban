.PHONY: sync setup run teleop-run teleop-validate camera-stream-enable camera-stream-udp camera-view-udp camera-stream-tls-provision camera-stream-tls stop shutdown voltage imu sim teleop-sim viewer gamepad-headless-enable gamepad-headless-disable

HOST ?= microban
ID ?=
CAMERA_UDP_PORT ?= 5000
CAMERA_UDP_DEVICE ?= /dev/v4l/by-id/usb-3D_USB_Camera_3D_USB_Camera_01.00.00-video-index0
CAMERA_UDP_RESOLUTION ?= 1280x480
CAMERA_UDP_FPS ?= 60
CAMERA_UDP_BITRATE ?= 4M
CAMERA_TLS_PORT ?= 8443
CAMERA_TLS_CA ?= $(HOME)/.config/microban-teleop/microban-camera.crt
CAMERA_TLS_CLIENT_CERT ?= $(HOME)/.config/microban-teleop/microban-camera-client.crt
CAMERA_TLS_CLIENT_KEY ?= $(HOME)/.config/microban-teleop/microban-camera-client.key

sync:
	rsync -avz \
		--exclude='.git' \
		--exclude='.venv' \
		--exclude='__pycache__' \
		--exclude='cad' \
		--exclude='docs' \
		--exclude='logs' \
		--exclude='src/debug' \
		--exclude='src/sim' \
		--exclude='src/model/mjcf' \
		./ $(HOST):microban

setup: sync
	ssh $(HOST) "bash -l -c 'cd microban && uv sync --frozen'"

sim:
	PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50

viewer:
	PYTHONPATH=src uv run src/sim/viewer_main.py --hz 25

run: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/main.py'"

# External-PC VR control: robot receives the deadman-gated UDP command stream.
teleop-run: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && MICROBAN_INPUT=network MICROBAN_NETWORK_ALLOWED_IP=\$${SSH_CONNECTION%% *} PYTHONPATH=src .venv/bin/python src/main.py'"

# Validate the final PICO policy with the exact local runtime, sync the same
# checkout and lockfile, materialize the Pi environment, then repeat the same
# CPU-only parser/inference smoke on the Pi. This never opens the motor bus.
teleop-validate:
	PYTHONPATH=src uv run --locked python tools/validate_pico_policy.py src/agents/pico_teleop.onnx
	$(MAKE) setup HOST=$(HOST)
	ssh $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python tools/validate_pico_policy.py src/agents/pico_teleop.onnx'"

teleop-sim:
	PYTHONPATH=src uv run --group sim src/sim/sim_main.py --hz 50 --input network

# Default camera stream: GPU-hardware H264 encode (bcm2835-codec /dev/video11) over
# UDP/MPEG-TS to whichever machine runs this target (same SSH_CONNECTION trick as
# teleop-run). Needs ffmpeg on the Pi (sudo apt-get install -y ffmpeg) and exclusive access
# to the camera, so stop camera-stream-enable first (they share the same USB device).
# Measured ~60fps at 1280x480 vs ~32fps for camera-stream-enable's MJPEG/HTTP stream, because
# H264 needs far less bandwidth per frame over the robot's 2.4GHz WiFi link. Always stop this
# with Ctrl+C (SIGINT), not kill -9: force-killing it can wedge /dev/video11 and require
# rebooting the Pi to recover.
camera-stream-udp: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && ffmpeg -f v4l2 -input_format mjpeg -video_size $(CAMERA_UDP_RESOLUTION) -framerate $(CAMERA_UDP_FPS) -i $(CAMERA_UDP_DEVICE) -vf format=yuv420p -c:v h264_v4l2m2m -b:v $(CAMERA_UDP_BITRATE) -f mpegts udp://\$${SSH_CONNECTION%% *}:$(CAMERA_UDP_PORT)?pkt_size=1316'"

# Alternative: plain MJPEG/HTTP relay (no hardware encode), viewable from any browser at
# http://microban:8080/stream. Lower fps (~32 at 1280x480) than camera-stream-udp; kept for
# quick ad-hoc checks.
camera-stream-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-camera-stream.sh'"

# View the camera-stream-udp output on this machine.
camera-view-udp:
	ffplay -fflags nobuffer -flags low_delay "udp://@:$(CAMERA_UDP_PORT)?pkt_size=1316"

# Generate mutually pinned identities. The robot server key remains on the
# robot; the PC client key remains on the PC. Only public certificates cross SSH.
camera-stream-tls-provision: sync
	ssh $(HOST) "bash -l -c 'cd microban && bash systemd/provision-camera-tls.sh'"
	MICROBAN_CAMERA_CLIENT_CERT="$(CAMERA_TLS_CLIENT_CERT)" MICROBAN_CAMERA_CLIENT_KEY="$(CAMERA_TLS_CLIENT_KEY)" bash systemd/provision-camera-client-tls.sh
	mkdir -p "$(dir $(CAMERA_TLS_CA))"
	scp "$(HOST):.config/microban-camera-tls/server.crt" "$(CAMERA_TLS_CA)"
	chmod 0644 "$(CAMERA_TLS_CA)"
	scp "$(CAMERA_TLS_CLIENT_CERT)" "$(HOST):.config/microban-camera-tls/client.crt.new"
	ssh $(HOST) "chmod 0644 .config/microban-camera-tls/client.crt.new && mv .config/microban-camera-tls/client.crt.new .config/microban-camera-tls/client.crt"
	chmod 0600 "$(CAMERA_TLS_CLIENT_KEY)"

# Latest-only mutually authenticated stereo snapshots. Unlike /stream, a
# client that is slower than the camera cannot accumulate an MJPEG frame queue.
camera-stream-tls: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && exec python3 tools/camera_snapshot_tls_proxy.py --cert ~/.config/microban-camera-tls/server.crt --key ~/.config/microban-camera-tls/server.key --client-cert ~/.config/microban-camera-tls/client.crt --port $(CAMERA_TLS_PORT)'"

stop:
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/stop.py'"

voltage: sync
	ssh $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/voltage.py $(ID)'"

imu: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && PYTHONPATH=src .venv/bin/python src/imu.py'"

shutdown:
	ssh -tt $(HOST) "sudo shutdown -h now"

# Opt-in headless mode: a service launches the control loop when START is held 2s on
# the gamepad (no SSH needed); B stops it. See docs/usage.md.
gamepad-headless-enable: sync
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh'"

gamepad-headless-disable:
	ssh -tt $(HOST) "bash -l -c 'cd microban && sudo bash systemd/install-gamepad-daemon.sh --uninstall'"
