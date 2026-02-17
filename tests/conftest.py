"""Pytest configuration and fixtures for frontend tests."""

import http.server
import socketserver
import subprocess
import threading
from pathlib import Path
from typing import Generator

import pytest

# Constants
TESTS_DIR: Path = Path(__file__).parent
TESTS_TEMP_DIR: Path = TESTS_DIR / ".temp"
HTTP_SERVER_PORT: int = 8765


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
	"""HTTP request handler that suppresses logging."""

	def log_message(self, format: str, *args: object) -> None:
		"""Suppress all log messages."""
		pass


@pytest.fixture(scope="session")
def ensure_pipeline_output() -> Path:
	"""Ensure the test pipeline has been run and output exists.

	If tests/.temp/ doesn't exist, runs `make am-pipeline-test` to generate it.
	Returns the path to the temp directory.
	"""
	if not TESTS_TEMP_DIR.exists() or not (TESTS_TEMP_DIR / "vis").exists():
		print(f"\n[fixture] Running test pipeline to generate {TESTS_TEMP_DIR}")
		result: subprocess.CompletedProcess[str] = subprocess.run(
			["make", "am-pipeline-test"],
			cwd=TESTS_DIR.parent,  # Project root
			capture_output=True,
			text=True,
			timeout=600,  # 10 minute timeout for pipeline
		)
		if result.returncode != 0:
			pytest.fail(
				f"Pipeline failed with code {result.returncode}:\n"
				f"stdout: {result.stdout}\n"
				f"stderr: {result.stderr}"
			)
	return TESTS_TEMP_DIR


@pytest.fixture(scope="session")
def http_server(ensure_pipeline_output: Path) -> Generator[str, None, None]:
	"""Start an HTTP server serving tests/.temp/ for the test session.

	Yields the base URL (e.g., 'http://localhost:8765').
	"""
	server_dir: Path = ensure_pipeline_output

	def handler_factory(*args: object, **kwargs: object) -> QuietHTTPRequestHandler:
		return QuietHTTPRequestHandler(
			*args,
			directory=str(server_dir),
			**kwargs,  # type: ignore[arg-type]
		)

	httpd: socketserver.TCPServer = socketserver.TCPServer(
		("", HTTP_SERVER_PORT),
		handler_factory,  # type: ignore[arg-type]
	)

	# Run server in background thread
	server_thread: threading.Thread = threading.Thread(
		target=httpd.serve_forever,
		daemon=True,
	)
	server_thread.start()

	base_url: str = f"http://localhost:{HTTP_SERVER_PORT}"
	print(f"\n[fixture] HTTP server running at {base_url}")

	yield base_url

	# Cleanup
	httpd.shutdown()
	print("\n[fixture] HTTP server stopped")
