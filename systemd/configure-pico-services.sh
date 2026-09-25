#!/usr/bin/env bash
# Install and manage fail-closed PICO hardware services on the robot.
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd -- "${script_dir}/.." && pwd)"
readonly runtime_unit="microban-pico-runtime.service"
readonly camera_unit="microban-camera-tls.service"
readonly conflicting_unit="microban-gamepad.service"
readonly runtime_unit_path="/etc/systemd/system/${runtime_unit}"
readonly camera_unit_path="/etc/systemd/system/${camera_unit}"
readonly runtime_env="/etc/default/microban-pico-runtime"
readonly camera_env="/etc/default/microban-camera-tls"
readonly state_file="/etc/default/microban-pico-services"

usage() {
  cat <<'EOF'
usage:
  sudo bash systemd/configure-pico-services.sh install --allowed-ip IPV4 [options]
  sudo bash systemd/configure-pico-services.sh enable|disable|restart|status|logs|uninstall

install options:
  --allowed-ip IPV4       Only this PC may send control snapshots (required)
  --port PORT             Robot UDP control port (default: 5555)
  --service-user USER     Runtime account (default: invoking sudo user)
  --without-camera        Do not install/enable the camera mTLS proxy

`install` validates and writes units but never starts them. `enable` starts the
services and enables boot startup. The runtime always starts all-joint torque
OFF; only a fresh PICO A-button command may enable torque.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_root() {
  (( EUID == 0 )) || die "run with sudo"
}

safe_path() {
  [[ "$1" =~ ^/[A-Za-z0-9._/@+,-]+$ ]] || die "path contains unsupported characters: $1"
}

render_unit() {
  local source=$1
  local destination=$2
  local service_user=$3
  local service_home=$4
  sed \
    -e "s|@REPO_ROOT@|${repo_root}|g" \
    -e "s|@SERVICE_USER@|${service_user}|g" \
    -e "s|@SERVICE_HOME@|${service_home}|g" \
    "${source}" > "${destination}"
  chmod 0644 "${destination}"
}

action=${1:-}
[[ -n "${action}" ]] || { usage >&2; exit 2; }
shift
require_root

case "${action}" in
  install)
    allowed_ip=
    port=5555
    service_user=${SUDO_USER:-user}
    with_camera=1
    while (( $# > 0 )); do
      case "$1" in
        --allowed-ip) (( $# >= 2 )) || die "--allowed-ip needs a value"; allowed_ip=$2; shift 2 ;;
        --port) (( $# >= 2 )) || die "--port needs a value"; port=$2; shift 2 ;;
        --service-user) (( $# >= 2 )) || die "--service-user needs a value"; service_user=$2; shift 2 ;;
        --without-camera) with_camera=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown install option: $1" ;;
      esac
    done
    [[ -n "${allowed_ip}" ]] || die "--allowed-ip is required (fail-closed sender allowlist)"
    /usr/bin/python3 - "${allowed_ip}" <<'PY' || exit 1
import ipaddress
import sys
try:
    value = ipaddress.ip_address(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"ERROR: invalid --allowed-ip: {exc}")
if value.version != 4 or str(value) != sys.argv[1]:
    raise SystemExit("ERROR: --allowed-ip must be one canonical IPv4 address")
PY
    [[ "${port}" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || die "invalid UDP port"
    [[ "${service_user}" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]] || die "invalid service user"
    getent passwd "${service_user}" >/dev/null || die "service user does not exist: ${service_user}"
    service_home=$(getent passwd "${service_user}" | cut -d: -f6)
    safe_path "${repo_root}"
    safe_path "${service_home}"
    [[ -x "${repo_root}/.venv/bin/python" ]] || die "robot venv is missing; run 'uv sync --frozen'"
    [[ -f "${repo_root}/src/agents/pico_teleop.onnx" && ! -L "${repo_root}/src/agents/pico_teleop.onnx" ]] || die "validated pico_teleop.onnx is missing"
    if (( with_camera == 1 )); then
      systemctl cat microban-camera.service >/dev/null 2>&1 || die "install microban-camera.service first"
      for required in server.crt server.key client.crt; do
        path="${service_home}/.config/microban-camera-tls/${required}"
        [[ -f "${path}" && ! -L "${path}" ]] || die "camera TLS identity is missing: ${path}"
      done
      key_mode=$(stat -c '%a' -- "${service_home}/.config/microban-camera-tls/server.key")
      [[ "${key_mode}" == "600" || "${key_mode}" == "400" ]] || die "camera TLS server key must be mode 0600 or 0400"
    fi

    install -d -m 0755 /etc/systemd/system /etc/default
    temp_runtime=$(mktemp /etc/default/.microban-pico-runtime.XXXXXX)
    temp_camera=$(mktemp /etc/default/.microban-camera-tls.XXXXXX)
    temp_state=$(mktemp /etc/default/.microban-pico-services.XXXXXX)
    trap 'rm -f -- "${temp_runtime:-}" "${temp_camera:-}" "${temp_state:-}"' EXIT
    chmod 0644 "${temp_runtime}" "${temp_camera}" "${temp_state}"
    {
      printf 'MICROBAN_NETWORK_ALLOWED_IP=%s\n' "${allowed_ip}"
      printf 'MICROBAN_NETWORK_PORT=%s\n' "${port}"
      printf 'MICROBAN_NETWORK_STALE_S=0.3\n'
    } > "${temp_runtime}"
    {
      printf 'MICROBAN_CAMERA_TLS_PORT=8443\n'
      printf 'MICROBAN_CAMERA_HTTP_PORT=8080\n'
    } > "${temp_camera}"
    printf 'MICROBAN_CAMERA_TLS_ENABLED=%s\n' "${with_camera}" > "${temp_state}"
    mv -f -- "${temp_runtime}" "${runtime_env}"
    mv -f -- "${temp_camera}" "${camera_env}"
    mv -f -- "${temp_state}" "${state_file}"
    trap - EXIT
    render_unit "${repo_root}/systemd/${runtime_unit}.in" "${runtime_unit_path}" "${service_user}" "${service_home}"
    if (( with_camera == 1 )); then
      render_unit "${repo_root}/systemd/${camera_unit}.in" "${camera_unit_path}" "${service_user}" "${service_home}"
    else
      systemctl disable --now "${camera_unit}" 2>/dev/null || true
      rm -f -- "${camera_unit_path}"
    fi
    systemctl daemon-reload
    echo "Installed disabled robot services. Start them with: sudo bash $0 enable"
    ;;
  enable)
    (( $# == 0 )) || die "enable takes no options"
    [[ -f "${runtime_unit_path}" && -f "${runtime_env}" && -f "${state_file}" ]] || die "run '$0 install ...' first"
    units=("${runtime_unit}")
    if grep -qx 'MICROBAN_CAMERA_TLS_ENABLED=1' "${state_file}"; then
      [[ -f "${camera_unit_path}" ]] || die "camera service configuration is incomplete"
      units=("${camera_unit}" "${runtime_unit}")
    fi
    # Do not leave both mutually exclusive controllers enabled for the next
    # boot.  Merely declaring Conflicts= prevents concurrent execution, but
    # does not define which controller wins when both are enabled.
    systemctl disable --now "${conflicting_unit}" 2>/dev/null || true
    systemctl enable --now "${units[@]}"
    ;;
  disable)
    (( $# == 0 )) || die "disable takes no options"
    systemctl disable --now "${runtime_unit}" 2>/dev/null || true
    systemctl disable --now "${camera_unit}" 2>/dev/null || true
    ;;
  restart)
    (( $# == 0 )) || die "restart takes no options"
    systemctl restart "${runtime_unit}"
    if [[ -f "${state_file}" ]] && grep -qx 'MICROBAN_CAMERA_TLS_ENABLED=1' "${state_file}"; then
      systemctl restart "${camera_unit}"
    fi
    ;;
  status)
    (( $# == 0 )) || die "status takes no options"
    systemctl --no-pager --full status "${runtime_unit}" "${camera_unit}"
    ;;
  logs)
    (( $# == 0 )) || die "logs takes no options"
    journalctl -u "${runtime_unit}" -u "${camera_unit}" -f
    ;;
  uninstall)
    (( $# == 0 )) || die "uninstall takes no options"
    systemctl disable --now "${runtime_unit}" "${camera_unit}" 2>/dev/null || true
    rm -f -- "${runtime_unit_path}" "${camera_unit_path}" "${runtime_env}" "${camera_env}" "${state_file}"
    systemctl daemon-reload
    echo "Removed PICO runtime units/config. Camera stream service and TLS identities were preserved."
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
