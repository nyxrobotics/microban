#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo bash systemd/install-camera-stream.sh" >&2
    exit 1
fi

apt-get update
apt-get install -y ustreamer

CAMERA_DEVICE="/dev/video0"
for candidate in /dev/v4l/by-id/*-video-index0; do
    if [[ -e "$candidate" ]]; then
        CAMERA_DEVICE="$candidate"
        break
    fi
done

USER_NAME=${SUDO_USER:-user}
sed "s/^User=.*/User=${USER_NAME}/" systemd/microban-camera.service \
    > /etc/systemd/system/microban-camera.service
chmod 0644 /etc/systemd/system/microban-camera.service
install -d -m 0755 /etc/default
if [[ ! -e /etc/default/microban-camera ]]; then
    printf '%s\n' \
        "CAMERA_DEVICE=${CAMERA_DEVICE}" \
        "CAMERA_FORMAT=MJPEG" \
        "CAMERA_RESOLUTION=3200x1200" \
        "CAMERA_FPS=30" \
        "CAMERA_PORT=8080" \
        > /etc/default/microban-camera
fi

systemctl daemon-reload
systemctl enable --now microban-camera.service
systemctl --no-pager --full status microban-camera.service || true

echo "Camera stream: http://$(hostname):8080/stream"
echo "If startup failed, inspect supported modes with: v4l2-ctl --list-formats-ext -d ${CAMERA_DEVICE}"
