from __future__ import annotations

import importlib.util
import io
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).parents[1] / "tools" / "camera_snapshot_tls_proxy.py"
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
    def test_fetches_complete_jpeg_and_all_timing_headers(self) -> None:
        connection = FakeConnection("ignored", 1, 1.0)
        with patch.object(proxy.http.client, "HTTPConnection", return_value=connection):
            body, headers = proxy.fetch_snapshot("127.0.0.1", 8080, 0.5)
        self.assertEqual(body, b"\xff\xd8data\xff\xd9")
        self.assertEqual(set(headers), set(proxy.FORWARDED_HEADERS))
        self.assertEqual(connection.request_arguments, ("GET", "/snapshot", {"headers": {"Connection": "close"}}))
        self.assertTrue(connection.closed)

    def test_rejects_duplicate_security_header(self) -> None:
        response = FakeResponse()
        response.headers["X-Timestamp"] = "2"
        connection = FakeConnection("ignored", 1, 1.0)
        connection.response = response
        with (
            patch.object(
                proxy.http.client, "HTTPConnection", return_value=connection
            ),
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
        handler.requestline = "GET /stream HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "GET"
        handler.wfile = io.BytesIO()
        handler.send_error = Mock()
        handler.do_GET()
        handler.send_error.assert_called_once_with(404)


if __name__ == "__main__":
    unittest.main()
