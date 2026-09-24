#!/usr/bin/env bash
set -euo pipefail

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/microban-camera-tls"
cert_path="$config_dir/server.crt"
key_path="$config_dir/server.key"
host_name="$(hostname)"

mkdir -p "$config_dir"
chmod 0700 "$config_dir"

if [[ -e "$cert_path" || -e "$key_path" ]]; then
  if [[ -f "$cert_path" && -f "$key_path" ]]; then
    chmod 0600 "$key_path"
    chmod 0644 "$cert_path"
    echo "$cert_path"
    exit 0
  fi
  echo "refusing a partial camera TLS identity in $config_dir" >&2
  exit 1
fi

openssl req -x509 -newkey ed25519 -nodes -days 825 \
  -subj "/CN=$host_name" \
  -addext "subjectAltName=DNS:$host_name,DNS:$host_name.lan" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=serverAuth" \
  -keyout "$key_path" \
  -out "$cert_path"
chmod 0600 "$key_path"
chmod 0644 "$cert_path"
echo "$cert_path"
