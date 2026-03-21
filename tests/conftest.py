"""Pytest configuration and fixtures for frontend tests."""

import atexit
import http.server
import socket
import socketserver
import sys
from typing import Any
import threading
import time
from pathlib import Path

import pytest

# Constants
TESTS_DIR: Path = Path(__file__).parent
TESTS_TEMP_DIR: Path = TESTS_DIR / ".temp"
HTTP_SERVER_PORT: int = 8765

# Module-level server reference for cleanup
_httpd: socketserver.TCPServer | None = None


def _is_port_in_use(port: int) -> bool:
	"""Check if a port is in use."""
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		return s.connect_ex(("localhost", port)) == 0


def _wait_for_port(port: int, timeout: float = 5.0) -> bool:
	"""Wait for a port to become available."""
	start: float = time.time()
	while time.time() - start < timeout:
		if _is_port_in_use(port):
			return True
		time.sleep(0.1)
	return False


def _shutdown_server() -> None:
	"""Shut down the HTTP server if running."""
	global _httpd
	if _httpd is not None:
		_httpd.shutdown()
		_httpd = None


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
	"""HTTP request handler that suppresses logging."""

	def log_message(self, format: str, *args: object) -> None:
		"""Suppress all log messages."""
		pass


class _QuietTCPServer(socketserver.TCPServer):
	"""TCPServer that silences BrokenPipeError from client disconnects."""

	allow_reuse_address = True

	def handle_error(
		self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
	) -> None:
		exc_type: type[BaseException] | None = sys.exc_info()[0]
		if exc_type is not None and issubclass(exc_type, BrokenPipeError):
			return
		super().handle_error(request, client_address)


def _make_handler_factory(
	directory: str,
) -> type[QuietHTTPRequestHandler]:
	"""Create a handler class bound to a specific directory."""

	class BoundHandler(QuietHTTPRequestHandler):
		def __init__(self, *args: object, **kwargs: object) -> None:
			super().__init__(
				*args,  # type: ignore[arg-type]
				directory=directory,
				**kwargs,  # type: ignore[arg-type]
			)

	return BoundHandler


def pytest_configure(config: pytest.Config) -> None:
	"""Start HTTP server on the controller process (not xdist workers).

	The controller outlives all workers, so the server stays up for the
	entire test session. Workers just connect to it.
	"""
	global _httpd
	if hasattr(config, "workerinput"):
		return  # xdist worker — skip

	# Ensure directory exists (server can serve before pipeline completes;
	# tests wait for pipeline via the ensure_pipeline_output fixture)
	TESTS_TEMP_DIR.mkdir(parents=True, exist_ok=True)

	try:
		_httpd = _QuietTCPServer(
			("", HTTP_SERVER_PORT),
			_make_handler_factory(str(TESTS_TEMP_DIR)),
		)
	except OSError as exc:
		pytest.exit(f"Cannot bind HTTP server to port {HTTP_SERVER_PORT}: {exc}")
	server_thread: threading.Thread = threading.Thread(
		target=_httpd.serve_forever,
		daemon=True,
	)
	server_thread.start()

	if not _wait_for_port(HTTP_SERVER_PORT):
		pytest.exit(f"HTTP server failed to start on port {HTTP_SERVER_PORT}")

	print(f"\n[conftest] HTTP server running at http://localhost:{HTTP_SERVER_PORT}")
	atexit.register(_shutdown_server)


def pytest_unconfigure(config: pytest.Config) -> None:
	"""Shut down HTTP server on session end (including KeyboardInterrupt)."""
	if not hasattr(config, "workerinput"):
		_shutdown_server()


@pytest.fixture(scope="session")
def ensure_pipeline_output() -> Path:
	"""Ensure test pipeline output exists, skip frontend tests if not.

	The pipeline is run by ``make test`` (via the ``tests/.temp/.pipeline_complete``
	prerequisite) before pytest starts. If running pytest directly without make,
	frontend tests are skipped.
	"""
	if not (TESTS_TEMP_DIR / ".pipeline_complete").exists():
		pytest.skip("Pipeline output not found. Run 'make am-pipeline-test' first.")
	return TESTS_TEMP_DIR


@pytest.fixture(scope="session")
def http_server(ensure_pipeline_output: Path) -> str:
	"""Return the base URL of the shared HTTP server.

	The server is started by the controller process in pytest_configure
	and lives for the entire session. This fixture just ensures the
	pipeline has run (so pages have data) before returning the URL.
	"""
	return f"http://localhost:{HTTP_SERVER_PORT}"
