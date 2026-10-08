#!/usr/bin/env bash
# Install and switch the robot-local GC300 runtime without using the old
# headless gamepad daemon (which blocks the robot's Wi-Fi).
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd -- "${script_dir}/.." && pwd)"
readonly unit_name="microban-gc300-runtime.service"
readonly pico_unit="microban-pico-runtime.service"
readonly gamepad_daemon_unit="microban-gamepad.service"
readonly unit_path="/etc/systemd/system/${unit_name}"

usage() {
  cat <<'EOF'
usage: sudo bash systemd/configure-gc300-runtime.sh install|enable|pico|disable|status

install  Render the unit for this checkout; do not start it.
enable   Select GC300 control now and at boot, stopping PICO control first.
pico     Restore PICO control now and at boot, stopping GC300 control first.
disable  Stop GC300 control and disable its boot startup.
status   Show the GC300 and PICO runtime states.
EOF
}

stop_if_installed() {
  local unit=$1
  if systemctl cat "${unit}" >/dev/null 2>&1; then
    systemctl disable --now "${unit}"
  fi
}

(( EUID == 0 )) || { echo "Run with sudo" >&2; exit 1; }

case "${1:-}" in
  install)
    (( $# == 1 )) || { usage >&2; exit 2; }
    [[ -x "${repo_root}/.venv/bin/python" ]] || {
      echo "Robot virtual environment is missing: ${repo_root}/.venv/bin/python" >&2
      exit 1
    }
    [[ -f "${repo_root}/src/gc300_main.py" && -f "${repo_root}/src/input/gc300_input.py" ]] || {
      echo "GC300 runtime source is missing" >&2
      exit 1
    }
    service_user="$(stat -c '%U' -- "${repo_root}")"
    [[ -n "${service_user}" && "${service_user}" != root && "${service_user}" != UNKNOWN ]] || {
      echo "Repository must be owned by the unprivileged runtime account" >&2
      exit 1
    }
    [[ "${repo_root}" =~ ^/[A-Za-z0-9._/@+,-]+$ ]] || {
      echo "Repository path contains unsupported characters" >&2
      exit 1
    }
    install -d -m 0755 /etc/systemd/system
    temp_unit="$(mktemp /etc/systemd/system/.microban-gc300-runtime.XXXXXX)"
    trap 'rm -f -- "${temp_unit:-}"' EXIT
    sed \
      -e "s|@REPO_ROOT@|${repo_root}|g" \
      -e "s|@SERVICE_USER@|${service_user}|g" \
      "${script_dir}/${unit_name}.in" > "${temp_unit}"
    chmod 0644 "${temp_unit}"
    mv -f -- "${temp_unit}" "${unit_path}"
    trap - EXIT
    systemctl daemon-reload
    echo "Installed ${unit_name}; run '$0 enable' to select GC300 control."
    ;;
  enable)
    (( $# == 1 )) || { usage >&2; exit 2; }
    [[ -f "${unit_path}" ]] || { echo "Run '$0 install' first" >&2; exit 1; }
    stop_if_installed "${pico_unit}"
    stop_if_installed "${gamepad_daemon_unit}"
    systemctl enable --now "${unit_name}"
    ;;
  pico)
    (( $# == 1 )) || { usage >&2; exit 2; }
    systemctl cat "${pico_unit}" >/dev/null
    stop_if_installed "${unit_name}"
    stop_if_installed "${gamepad_daemon_unit}"
    bash "${script_dir}/configure-pico-services.sh" enable
    ;;
  disable)
    (( $# == 1 )) || { usage >&2; exit 2; }
    stop_if_installed "${unit_name}"
    ;;
  status)
    (( $# == 1 )) || { usage >&2; exit 2; }
    systemctl --no-pager show \
      -p Id -p LoadState -p ActiveState -p UnitFileState \
      "${unit_name}" "${pico_unit}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
