from __future__ import annotations

import http.client
import importlib.util
import io
import os
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).parents[1] / "tools" / "camera_snapshot_tls_proxy.py"
REPO_ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("camera_snapshot_tls_proxy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def source_headers() -> Message:
    headers = Message()
    headers["Content-Type"] = "image/jpeg"
    headers["Content-Length"] = "8"
    for name in proxy.FORWARDED_HEADERS:
        headers[name] = "true" if name == "X-UStreamer-Online" else "1"
    return headers


def make_identity(
    directory: Path, name: str, eku: str, *, server: bool
) -> tuple[Path, Path]:
    cert = directory / f"{name}.crt"
    key = directory / f"{name}.key"
    command = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "ed25519",
        "-nodes",
        "-days",
        "1",
        "-subj",
        f"/CN={'localhost' if server else name}",
        "-addext",
        "basicConstraints=critical,CA:FALSE",
        "-addext",
        "keyUsage=critical,digitalSignature",
        "-addext",
        f"extendedKeyUsage={eku}",
    ]
    if server:
        command.extend(("-addext", "subjectAltName=DNS:localhost"))
    command.extend(("-keyout", str(key), "-out", str(cert)))
    subprocess.run(command, check=True, capture_output=True)
    os.chmod(key, 0o600)
    return cert, key


class FakeResponse:
    status = 200

    def __init__(self, body: bytes = b"\xff\xd8data\xff\xd9") -> None:
        self.body = body
        self.headers = source_headers()

    def read(self, _limit: int) -> bytes:
        return self.body


class FakeConnection:
    response = FakeResponse()

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.arguments = (host, port, timeout)
        self.request_arguments: tuple[object, ...] | None = None
        self.closed = False

    def request(self, *args: object, **kwargs: object) -> None:
        self.request_arguments = (*args, kwargs)

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class CameraSnapshotTlsProxyTests(unittest.TestCase):
    def test_provisioners_create_idempotent_owner_only_identities(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            environment = os.environ.copy()
            environment["XDG_CONFIG_HOME"] = str(directory)
            for script_name in (
                "provision-camera-tls.sh",
                "provision-camera-client-tls.sh",
            ):
                script = REPO_ROOT / "systemd" / script_name
                for _ in range(2):
                    subprocess.run(
                        ("bash", str(script)),
                        check=True,
                        capture_output=True,
                        env=environment,
                    )

            server_dir = directory / "microban-camera-tls"
            client_dir = directory / "microban-teleop"
            self.assertEqual(server_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual(client_dir.stat().st_mode & 0o777, 0o700)
            self.assertEqual((server_dir / "server.key").stat().st_mode & 0o777, 0o600)
            self.assertEqual((server_dir / "server.crt").stat().st_mode & 0o777, 0o644)
            self.assertEqual(
                (client_dir / "microban-camera-client.key").stat().st_mode & 0o777,
                0o600,
            )
            self.assertEqual(
                (client_dir / "microban-camera-client.crt").stat().st_mode & 0o777,
                0o644,
            )

    def test_fetches_complete_jpeg_and_all_timing_headers(self) -> None:
        connection = FakeConnection("ignored", 1, 1.0)
        with patch.object(proxy.http.client, "HTTPConnection", return_value=connection):
            body, headers = proxy.fetch_snapshot("127.0.0.1", 8080, 0.5)
        self.assertEqual(body, b"\xff\xd8data\xff\xd9")
        self.assertEqual(set(headers), set(proxy.FORWARDED_HEADERS))
        self.assertEqual(
            connection.request_arguments,
            ("GET", "/snapshot", {"headers": {"Connection": "close"}}),
        )
        self.assertTrue(connection.closed)

    def test_rejects_duplicate_security_header(self) -> None:
        response = FakeResponse()
        response.headers["X-Timestamp"] = "2"
        connection = FakeConnection("ignored", 1, 1.0)
        connection.response = response
        with (
            patch.object(proxy.http.client, "HTTPConnection", return_value=connection),
            self.assertRaisesRegex(proxy.SnapshotError, "unique X-Timestamp"),
        ):
            proxy.fetch_snapshot("127.0.0.1", 8080, 0.5)

    def test_rejects_truncated_or_non_jpeg_body(self) -> None:
        for body in (b"short", b"\xff\xd8bad"):
            response = FakeResponse(body)
            response.headers.replace_header("Content-Length", str(len(body)))
            connection = FakeConnection("ignored", 1, 1.0)
            connection.response = response
            with (
                self.subTest(body=body),
                patch.object(
                    proxy.http.client, "HTTPConnection", return_value=connection
                ),
                self.assertRaises(proxy.SnapshotError),
            ):
                proxy.fetch_snapshot("127.0.0.1", 8080, 0.5)

    def test_handler_exposes_only_snapshot(self) -> None:
        handler = object.__new__(proxy.SnapshotProxyHandler)
        handler.path = "/stream"
        handler.headers = Message()
        handler.requestline = "GET /stream HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "GET"
        handler.wfile = io.BytesIO()
        handler.send_error = Mock()
        handler.do_GET()
        handler.send_error.assert_called_once_with(404)

    def test_mtls_slow_handshake_and_request_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            server_cert, server_key = make_identity(
                directory, "server", "serverAuth", server=True
            )
            client_cert, client_key = make_identity(
                directory, "client", "clientAuth", server=False
            )
            rogue_cert, rogue_key = make_identity(
                directory, "rogue", "clientAuth", server=False
            )
            context = proxy.build_tls_context(
                server_cert,
                server_key,
                client_cert,
            )
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertEqual(context.minimum_version, ssl.TLSVersion.TLSv1_3)
            self.assertEqual(context.maximum_version, ssl.TLSVersion.TLSv1_3)

            proxy.SnapshotProxyHandler.upstream_host = "127.0.0.1"
            proxy.SnapshotProxyHandler.upstream_port = 1
            proxy.SnapshotProxyHandler.upstream_timeout_s = 0.05
            server = proxy.ReusableThreadingHTTPServer(
                ("127.0.0.1", 0),
                proxy.SnapshotProxyHandler,
                tls_context=context,
                max_workers=2,
                tls_handshake_timeout_s=0.25,
                request_timeout_s=0.25,
                request_deadline_s=0.4,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            slow = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            try:
                client_context = ssl.create_default_context(cafile=str(server_cert))
                client_context.minimum_version = ssl.TLSVersion.TLSv1_3
                client_context.maximum_version = ssl.TLSVersion.TLSv1_3
                client_context.load_cert_chain(client_cert, client_key)
                connection = http.client.HTTPSConnection(
                    "localhost", port, timeout=1.0, context=client_context
                )
                started = time.monotonic()
                connection.request("GET", "/snapshot")
                response = connection.getresponse()
                self.assertEqual(response.status, 502)
                response.read()
                connection.close()
                self.assertLess(time.monotonic() - started, 0.8)

                rogue_context = ssl.create_default_context(cafile=str(server_cert))
                rogue_context.minimum_version = ssl.TLSVersion.TLSv1_3
                rogue_context.maximum_version = ssl.TLSVersion.TLSv1_3
                rogue_context.load_cert_chain(rogue_cert, rogue_key)
                rogue = http.client.HTTPSConnection(
                    "localhost", port, timeout=1.0, context=rogue_context
                )
                with self.assertRaises((OSError, ssl.SSLError)):
                    rogue.request("GET", "/snapshot")
                    rogue.getresponse()
                rogue.close()

                time.sleep(0.3)
                slow.settimeout(1.0)
                with self.assertRaises((ConnectionError, OSError)):
                    if slow.recv(1) == b"":
                        raise ConnectionError("server closed incomplete handshake")

                authenticated = client_context.wrap_socket(
                    socket.create_connection(("127.0.0.1", port), timeout=1.0),
                    server_hostname="localhost",
                )
                authenticated.settimeout(1.0)
                authenticated.sendall(b"GET /snapshot HTTP/1.1\r\nHost:")
                time.sleep(0.45)
                try:
                    final_byte = authenticated.recv(1)
                except OSError:
                    final_byte = b""
                self.assertEqual(final_byte, b"")
                authenticated.close()
            finally:
                slow.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
