#!/usr/bin/env python3
"""Expose uStreamer's latest JPEG snapshot through a small TLS 1.3 proxy.

The proxy deliberately forwards only ``/snapshot``.  uStreamer's continuous
MJPEG endpoint can queue frames behind a slow Wi-Fi client; requesting the
latest snapshot avoids carrying that queue into teleoperation.  The original
uStreamer timing headers are covered by TLS and are preserved verbatim so the
PC can derive the V4L2 dequeue time without inventing a new timestamp.
"""

from __future__ import annotations

import argparse
import http.client
import socket
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

MAX_JPEG_BYTES = 4 * 1024 * 1024
FORWARDED_HEADERS = (
    "X-Timestamp",
    "X-UStreamer-Online",
    "X-UStreamer-Dropped",
    "X-UStreamer-Width",
    "X-UStreamer-Height",
    "X-UStreamer-Grab-Timestamp",
    "X-UStreamer-Encode-Begin-Timestamp",
    "X-UStreamer-Encode-End-Timestamp",
    "X-UStreamer-Expose-Begin-Timestamp",
    "X-UStreamer-Expose-Cmp-Timestamp",
    "X-UStreamer-Expose-End-Timestamp",
    "X-UStreamer-Send-Timestamp",
)


class SnapshotError(RuntimeError):
    """The local uStreamer snapshot response is unusable."""


def fetch_snapshot(host: str, port: int, timeout_s: float) -> tuple[bytes, dict[str, str]]:
    """Fetch one latest-only JPEG and its exact source timing headers."""
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request("GET", "/snapshot", headers={"Connection": "close"})
        response = connection.getresponse()
        if response.status != 200:
            raise SnapshotError(f"uStreamer returned HTTP {response.status}")
        if response.headers.get_content_type().lower() != "image/jpeg":
            raise SnapshotError("uStreamer snapshot is not image/jpeg")
        raw_length = response.headers.get("Content-Length")
        if raw_length is None or not raw_length.isascii() or not raw_length.isdecimal():
            raise SnapshotError("uStreamer has no strict Content-Length")
        length = int(raw_length)
        if not 1 <= length <= MAX_JPEG_BYTES:
            raise SnapshotError("uStreamer snapshot size is outside bounds")
        body = response.read(MAX_JPEG_BYTES + 1)
        if len(body) != length:
            raise SnapshotError("uStreamer Content-Length does not match the body")
        if len(body) > MAX_JPEG_BYTES or not body.startswith(b"\xff\xd8") or not body.endswith(b"\xff\xd9"):
            raise SnapshotError("uStreamer did not return a complete bounded JPEG")

        headers: dict[str, str] = {}
        for name in FORWARDED_HEADERS:
            values = response.headers.get_all(name, failobj=[])
            if len(values) != 1:
                raise SnapshotError(f"uStreamer has no unique {name}")
            headers[name] = values[0]
        return body, headers
    finally:
        connection.close()


class SnapshotProxyHandler(BaseHTTPRequestHandler):
    """Strict single-endpoint HTTP handler used behind TLS."""

    protocol_version = "HTTP/1.1"
    server_version = "MicrobanCameraTLS/1"
    sys_version = ""
    upstream_host: ClassVar[str] = "127.0.0.1"
    upstream_port: ClassVar[int] = 8080
    upstream_timeout_s: ClassVar[float] = 1.0

    def do_GET(self) -> None:
        if self.path != "/snapshot":
            self.send_error(404)
            return
        try:
            body, headers = fetch_snapshot(
                self.upstream_host,
                self.upstream_port,
                self.upstream_timeout_s,
            )
        except (OSError, SnapshotError, http.client.HTTPException) as exc:
            self.send_error(502, explain=str(exc)[:160])
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        for name in FORWARDED_HEADERS:
            self.send_header(name, headers[name])
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_HEAD(self) -> None:
        self.send_error(405)

    def log_request(self, code: object = "-", size: object = "-") -> None:
        # A normal client polls several times per second; log only errors to
        # avoid turning stdout/SSH backpressure into camera latency.
        del code, size

    def log_message(self, format: str, *args: object) -> None:
        print(f"camera-tls {self.address_string()}: {format % args}", flush=True)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def get_request(self) -> tuple[socket.socket, object]:
        request, address = super().get_request()
        request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return request, address


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TLS 1.3 latest-snapshot proxy for Microban's local uStreamer"
    )
    parser.add_argument("--cert", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", default=8443, type=int)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", default=8080, type=int)
    parser.add_argument("--upstream-timeout-s", default=1.0, type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.upstream_port <= 65535:
        raise SystemExit("ports must be between 1 and 65535")
    if not 0.05 <= args.upstream_timeout_s <= 5.0:
        raise SystemExit("--upstream-timeout-s must be between 0.05 and 5 seconds")
    if not args.cert.is_file() or not args.key.is_file():
        raise SystemExit("TLS certificate/key file is missing")

    SnapshotProxyHandler.upstream_host = args.upstream_host
    SnapshotProxyHandler.upstream_port = args.upstream_port
    SnapshotProxyHandler.upstream_timeout_s = args.upstream_timeout_s
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(args.cert, args.key)

    server = ReusableThreadingHTTPServer((args.bind, args.port), SnapshotProxyHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(
        f"Microban latest-snapshot TLS proxy listening on {args.bind}:{args.port}; "
        f"upstream http://{args.upstream_host}:{args.upstream_port}/snapshot",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
