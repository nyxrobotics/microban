#!/usr/bin/env bash
set -euo pipefail

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/microban-camera-tls"
cert_path="$config_dir/server.crt"
key_path="$config_dir/server.key"
host_name="$(hostname)"

umask 077
mkdir -p "$config_dir"
chmod 0700 "$config_dir"

validate_identity() {
  chmod 0600 "$key_path"
  chmod 0644 "$cert_path"
  openssl pkey -in "$key_path" -check -noout >/dev/null
  openssl x509 -in "$cert_path" -noout -checkend 0 >/dev/null
  openssl verify -CAfile "$cert_path" -verify_hostname "$host_name" \
    "$cert_path" >/dev/null
  cert_public="$({ openssl x509 -in "$cert_path" -pubkey -noout \
    | openssl pkey -pubin -outform DER; } | sha256sum | cut -d' ' -f1)"
  key_public="$({ openssl pkey -in "$key_path" -pubout -outform DER; } \
    | sha256sum | cut -d' ' -f1)"
  if [[ ! "$cert_public" =~ ^[0-9a-f]{64}$ || "$cert_public" != "$key_public" ]]; then
    echo "camera TLS server certificate and key do not match" >&2
    exit 1
  fi
}

if [[ -e "$cert_path" || -e "$key_path" ]]; then
  if [[ -f "$cert_path" && -f "$key_path" ]]; then
    validate_identity
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
validate_identity
echo "$cert_path"
