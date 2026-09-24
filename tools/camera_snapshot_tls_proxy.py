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
import os
import socket
import ssl
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

MAX_JPEG_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_WORKERS = 4
DEFAULT_TLS_HANDSHAKE_TIMEOUT_S = 1.0
DEFAULT_REQUEST_TIMEOUT_S = 1.0
DEFAULT_REQUEST_DEADLINE_S = 2.0
MAX_REQUESTS_PER_CONNECTION = 256
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


def fetch_snapshot(
    host: str, port: int, timeout_s: float
) -> tuple[bytes, dict[str, str]]:
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
        if (
            len(body) > MAX_JPEG_BYTES
            or not body.startswith(b"\xff\xd8")
            or not body.endswith(b"\xff\xd9")
        ):
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

    def handle_one_request(self) -> None:
        # Socket inactivity timeouts alone do not stop a slowloris which sends
        # one byte just before every timeout.  Close the connection when the
        # complete request/response transaction exceeds an absolute deadline.
        handled = getattr(self, "_requests_handled", 0) + 1
        self._requests_handled = handled
        self._force_close_after_response = handled >= MAX_REQUESTS_PER_CONNECTION
        deadline_s = self.server.request_deadline_s
        timer = threading.Timer(deadline_s, self._expire_connection)
        timer.daemon = True
        timer.start()
        try:
            super().handle_one_request()
        finally:
            timer.cancel()
        if self._force_close_after_response:
            self.close_connection = True

    def end_headers(self) -> None:
        if getattr(self, "_force_close_after_response", False):
            self.send_header("Connection", "close")
        super().end_headers()

    def _expire_connection(self) -> None:
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass

    def do_GET(self) -> None:
        if self.headers.get_all("Transfer-Encoding", failobj=[]):
            self.send_error(400, explain="request bodies are not accepted")
            return
        content_lengths = self.headers.get_all("Content-Length", failobj=[])
        if content_lengths and content_lengths != ["0"]:
            self.send_error(400, explain="request bodies are not accepted")
            return
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
    """Bounded raw TCP acceptor which performs mTLS inside worker threads."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        tls_context: ssl.SSLContext,
        max_workers: int,
        tls_handshake_timeout_s: float,
        request_timeout_s: float,
        request_deadline_s: float,
    ) -> None:
        self.tls_context = tls_context
        self.tls_handshake_timeout_s = tls_handshake_timeout_s
        self.request_timeout_s = request_timeout_s
        self.request_deadline_s = request_deadline_s
        self._worker_slots = threading.BoundedSemaphore(max_workers)
        super().__init__(server_address, handler_class)

    def get_request(self) -> tuple[socket.socket, object]:
        request, address = super().get_request()
        request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return request, address

    def process_request(self, request: socket.socket, client_address: object) -> None:
        if not self._worker_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket,
        client_address: object,
    ) -> None:
        tls_request: ssl.SSLSocket | None = None
        try:
            # Never wrap the listening socket: SSLSocket.accept() performs a
            # blocking handshake in serve_forever's sole accept thread.  A raw
            # accept followed by this bounded worker handshake keeps one
            # incomplete ClientHello from stopping every camera request.
            request.settimeout(self.tls_handshake_timeout_s)
            tls_request = self.tls_context.wrap_socket(
                request,
                server_side=True,
                do_handshake_on_connect=False,
            )
            tls_request.settimeout(self.tls_handshake_timeout_s)
            tls_request.do_handshake()
            tls_request.settimeout(self.request_timeout_s)
            self.finish_request(tls_request, client_address)
        except (OSError, TimeoutError, ssl.SSLError):
            # Authentication failures and hostile/incomplete connections are
            # expected at this boundary; do not turn them into traceback I/O.
            pass
        except Exception:  # noqa: BLE001 - stdlib server isolation boundary
            self.handle_error(tls_request or request, client_address)
        finally:
            self.shutdown_request(tls_request or request)
            self._worker_slots.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TLS 1.3 latest-snapshot proxy for Microban's local uStreamer"
    )
    parser.add_argument("--cert", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument(
        "--client-cert",
        required=True,
        type=Path,
        help="exact self-signed client certificate trusted for mTLS",
    )
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", default=8443, type=int)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", default=8080, type=int)
    parser.add_argument("--upstream-timeout-s", default=1.0, type=float)
    parser.add_argument("--max-workers", default=DEFAULT_MAX_WORKERS, type=int)
    parser.add_argument(
        "--tls-handshake-timeout-s",
        default=DEFAULT_TLS_HANDSHAKE_TIMEOUT_S,
        type=float,
    )
    parser.add_argument(
        "--request-timeout-s",
        default=DEFAULT_REQUEST_TIMEOUT_S,
        type=float,
    )
    parser.add_argument(
        "--request-deadline-s",
        default=DEFAULT_REQUEST_DEADLINE_S,
        type=float,
    )
    return parser


def require_private_key(path: Path) -> None:
    """Reject missing, linked, foreign, or group/world-readable key files."""
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SystemExit(f"TLS private key is unavailable: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("TLS private key must be a regular file")
    if metadata.st_uid != os.geteuid():
        raise SystemExit("TLS private key must be owned by the service user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise SystemExit("TLS private key permissions must be owner-only")


def build_tls_context(
    cert_path: Path,
    key_path: Path,
    client_cert_path: Path,
) -> ssl.SSLContext:
    if not cert_path.is_file() or not client_cert_path.is_file():
        raise SystemExit("TLS server/client certificate file is missing")
    require_private_key(key_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    try:
        context.load_cert_chain(cert_path, key_path)
        # The provisioner installs one self-signed client leaf.  CA:FALSE and
        # clientAuth constrain it to this exact mutually pinned client identity.
        context.load_verify_locations(cafile=client_cert_path)
    except (OSError, ssl.SSLError) as exc:
        raise SystemExit(f"cannot load camera TLS identity: {exc}") from exc
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.upstream_port <= 65535:
        raise SystemExit("ports must be between 1 and 65535")
    if not 0.05 <= args.upstream_timeout_s <= 5.0:
        raise SystemExit("--upstream-timeout-s must be between 0.05 and 5 seconds")
    if not 1 <= args.max_workers <= 32:
        raise SystemExit("--max-workers must be between 1 and 32")
    if not 0.1 <= args.tls_handshake_timeout_s <= 5.0:
        raise SystemExit("--tls-handshake-timeout-s must be between 0.1 and 5 seconds")
    if not 0.1 <= args.request_timeout_s <= 5.0:
        raise SystemExit("--request-timeout-s must be between 0.1 and 5 seconds")
    if not args.upstream_timeout_s + 0.25 <= args.request_deadline_s <= 10.0:
        raise SystemExit(
            "--request-deadline-s must be at least upstream timeout + 0.25 "
            "and at most 10 seconds"
        )
    SnapshotProxyHandler.upstream_host = args.upstream_host
    SnapshotProxyHandler.upstream_port = args.upstream_port
    SnapshotProxyHandler.upstream_timeout_s = args.upstream_timeout_s
    context = build_tls_context(args.cert, args.key, args.client_cert)

    server = ReusableThreadingHTTPServer(
        (args.bind, args.port),
        SnapshotProxyHandler,
        tls_context=context,
        max_workers=args.max_workers,
        tls_handshake_timeout_s=args.tls_handshake_timeout_s,
        request_timeout_s=args.request_timeout_s,
        request_deadline_s=args.request_deadline_s,
    )
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
