"""Pytest configuration and fixtures for frontend tests."""

import atexit
import http.server
import os
import socket
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import filelock
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

	socketserver.TCPServer.allow_reuse_address = True
	_httpd = socketserver.TCPServer(
		("", HTTP_SERVER_PORT),
		_make_handler_factory(str(TESTS_TEMP_DIR)),
	)
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
def http_server(ensure_pipeline_output: Path) -> str:
	"""Return the base URL of the shared HTTP server.

	The server is started by the controller process in pytest_configure
	and lives for the entire session. This fixture just ensures the
	pipeline has run (so pages have data) before returning the URL.
	"""
	return f"http://localhost:{HTTP_SERVER_PORT}"
