"""Pytest configuration and fixtures for frontend tests."""

import http.server
import os
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path
from typing import Generator

import filelock
import pytest

# Constants
TESTS_DIR: Path = Path(__file__).parent
TESTS_TEMP_DIR: Path = TESTS_DIR / ".temp"
HTTP_SERVER_PORT: int = 8765


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


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
	"""HTTP request handler that suppresses logging."""

	def log_message(self, format: str, *args: object) -> None:
		"""Suppress all log messages."""
		pass


def _run_pipeline() -> None:
	"""Run the test pipeline, failing the test session if it errors."""
	print(f"\n[fixture] Running test pipeline to generate {TESTS_TEMP_DIR}")
	# suppress progress bars from tqdm, HuggingFace, and transformers
	# while keeping actual log/print messages intact
	quiet_env: dict[str, str] = {
		**os.environ,
		"TQDM_DISABLE": "1",
		"HF_HUB_DISABLE_PROGRESS_BARS": "1",
		"TRANSFORMERS_VERBOSITY": "error",
		"SPINNER_UPDATE_INTERVAL": "60",
	}
	# UV_NOSYNC=1 avoids nested uv lock deadlock: the outer `uv run python -m
	# pytest` holds a uv workspace lock, and without --no-sync the inner
	# `uv run python` (from make) would try to acquire the same lock.
	result: subprocess.CompletedProcess[str] = subprocess.run(
		["make", "am-pipeline-test", "UV_NOSYNC=1"],
		cwd=TESTS_DIR.parent,  # Project root
		text=True,
		timeout=600,  # 10 minute timeout for pipeline
		env=quiet_env,
	)
	if result.returncode != 0:
		pytest.fail(f"Pipeline failed with code {result.returncode}")


@pytest.fixture(scope="session")
def ensure_pipeline_output(
	tmp_path_factory: pytest.TempPathFactory,
	worker_id: str,
) -> Path:
	"""Ensure the test pipeline has been run and output exists.

	If tests/.temp/.pipeline_complete doesn't exist, runs `make am-pipeline-test`
	to generate it. Uses file locking with xdist to prevent multiple workers from
	running the pipeline concurrently (which would cause race conditions as the
	Makefile target starts with ``rm -rf tests/.temp/``).

	Returns the path to the temp directory.
	"""
	if worker_id == "master":
		# Not running with xdist - run directly
		if not (TESTS_TEMP_DIR / ".pipeline_complete").exists():
			_run_pipeline()
		return TESTS_TEMP_DIR

	# Running with xdist - coordinate via file lock
	root_tmp_dir: Path = tmp_path_factory.getbasetemp().parent
	lock_file: Path = root_tmp_dir / "pipeline.lock"

	with filelock.FileLock(str(lock_file)):
		if not (TESTS_TEMP_DIR / ".pipeline_complete").exists():
			_run_pipeline()

	return TESTS_TEMP_DIR


@pytest.fixture(scope="session")
def http_server(
	ensure_pipeline_output: Path,
	tmp_path_factory: pytest.TempPathFactory,
	worker_id: str,
) -> Generator[str, None, None]:
	"""Start an HTTP server serving tests/.temp/ for the test session.

	With xdist, coordinates so only one worker starts the server.
	Yields the base URL (e.g., 'http://localhost:8765').
	"""
	server_dir: Path = ensure_pipeline_output
	base_url: str = f"http://localhost:{HTTP_SERVER_PORT}"

	# factory wrapper loses type info for *args/**kwargs
	def handler_factory(*args: object, **kwargs: object) -> QuietHTTPRequestHandler:
		return QuietHTTPRequestHandler(
			*args,  # type: ignore[arg-type]
			directory=str(server_dir),
			**kwargs,  # type: ignore[arg-type]
		)

	if worker_id == "master":
		# Not running with xdist - start server normally
		httpd: socketserver.TCPServer = socketserver.TCPServer(
			("", HTTP_SERVER_PORT),
			handler_factory,
		)
		server_thread: threading.Thread = threading.Thread(
			target=httpd.serve_forever,
			daemon=True,
		)
		server_thread.start()
		print(f"\n[fixture] HTTP server running at {base_url}")
		yield base_url
		httpd.shutdown()
		print("\n[fixture] HTTP server stopped")
		return

	# Running with xdist - coordinate via file lock
	root_tmp_dir: Path = tmp_path_factory.getbasetemp().parent
	lock_file: Path = root_tmp_dir / "http_server.lock"

	with filelock.FileLock(str(lock_file)):
		if not _is_port_in_use(HTTP_SERVER_PORT):
			# We're the first worker - start the server
			httpd = socketserver.TCPServer(
				("", HTTP_SERVER_PORT),
				handler_factory,
			)
			server_thread = threading.Thread(
				target=httpd.serve_forever,
				daemon=True,
			)
			server_thread.start()
			print(f"\n[fixture] HTTP server started by {worker_id} at {base_url}")

	# Wait for server to be ready (in case another worker is starting it)
	if not _wait_for_port(HTTP_SERVER_PORT):
		pytest.fail(f"HTTP server failed to start on port {HTTP_SERVER_PORT}")

	yield base_url
	# Note: We don't shutdown the server here because other workers may still need it.
	# The server runs as a daemon thread and will be cleaned up when its parent process exits.
