#!/usr/bin/python3
"""Update only the robot's single allowed PICO UDP sender address."""

import ipaddress
import os
from pathlib import Path
import stat
import sys
import tempfile


RUNTIME_ENV = Path("/etc/default/microban-pico-runtime")


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"check", "sync"}:
        raise ValueError("usage: microban-pico-allowed-ip check|sync CANONICAL_IPV4")
    action, expected = sys.argv[1:]
    address = ipaddress.IPv4Address(expected)
    if (
        str(address) != expected
        or address.is_unspecified
        or address.is_loopback
        or address.is_multicast
        or expected == "255.255.255.255"
    ):
        raise ValueError("expected one usable canonical IPv4 address")

    metadata = RUNTIME_ENV.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0 or metadata.st_mode & 0o022:
        raise ValueError("robot runtime environment file must be root-owned and not writable by others")
    if metadata.st_size > 8192:
        raise ValueError("robot runtime environment file is too large")
    lines = RUNTIME_ENV.read_text(encoding="utf-8").splitlines(keepends=True)
    indices = [index for index, line in enumerate(lines) if line.startswith("MICROBAN_NETWORK_ALLOWED_IP=")]
    if len(indices) != 1:
        raise ValueError("expected exactly one robot sender allowlist entry")
    current = lines[indices[0]].removeprefix("MICROBAN_NETWORK_ALLOWED_IP=").strip()
    if current == expected:
        print("MATCH")
        return 0
    if action == "check":
        raise ValueError(f"robot allows {current}, PC source is {expected}")

    lines[indices[0]] = f"MICROBAN_NETWORK_ALLOWED_IP={expected}\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".microban-pico-runtime.", dir=RUNTIME_ENV.parent)
    try:
        os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode))
        os.fchown(descriptor, metadata.st_uid, metadata.st_gid)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.writelines(lines)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, RUNTIME_ENV)
        directory = os.open(RUNTIME_ENV.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("UPDATED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
