#!/usr/bin/env bash
set -euo pipefail

default_dir="${XDG_CONFIG_HOME:-$HOME/.config}/microban-teleop"
cert_path="${MICROBAN_CAMERA_CLIENT_CERT:-$default_dir/microban-camera-client.crt}"
key_path="${MICROBAN_CAMERA_CLIENT_KEY:-$default_dir/microban-camera-client.key}"
config_dir="$(dirname "$cert_path")"
if [[ "$config_dir" != "$(dirname "$key_path")" ]]; then
  echo "camera TLS client certificate and key must share one private directory" >&2
  exit 1
fi

mkdir -p "$config_dir"
chmod 0700 "$config_dir"

validate_identity() {
  chmod 0600 "$key_path"
  chmod 0644 "$cert_path"
  openssl pkey -in "$key_path" -check -noout >/dev/null
  openssl x509 -in "$cert_path" -noout -checkend 0 >/dev/null
  cert_public="$({ openssl x509 -in "$cert_path" -pubkey -noout \
    | openssl pkey -pubin -outform DER; } | sha256sum | cut -d' ' -f1)"
  key_public="$({ openssl pkey -in "$key_path" -pubout -outform DER; } \
    | sha256sum | cut -d' ' -f1)"
  if [[ ! "$cert_public" =~ ^[0-9a-f]{64}$ || "$cert_public" != "$key_public" ]]; then
    echo "camera TLS client certificate and key do not match" >&2
    exit 1
  fi
}

if [[ -e "$cert_path" || -e "$key_path" ]]; then
  if [[ -f "$cert_path" && -f "$key_path" ]]; then
    validate_identity
    echo "$cert_path"
    exit 0
  fi
  echo "refusing a partial camera TLS client identity in $config_dir" >&2
  exit 1
fi

umask 077
temp_dir="$(mktemp -d "$config_dir/.camera-client-tls.XXXXXX")"
cleanup() {
  rm -rf -- "$temp_dir"
}
trap cleanup EXIT

openssl req -x509 -newkey ed25519 -nodes -days 825 \
  -subj "/CN=microban-camera-gateway" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=clientAuth" \
  -keyout "$temp_dir/client.key" \
  -out "$temp_dir/client.crt"

install -m 0600 "$temp_dir/client.key" "$key_path"
install -m 0644 "$temp_dir/client.crt" "$cert_path"
validate_identity
echo "$cert_path"
