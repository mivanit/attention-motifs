"""VRAM-aware parallel model scheduler for s1 activation generation.

Spawns subprocesses running ``python -m pattern_lens.activations`` for each
model, packing multiple small models onto the same GPU when they fit and
giving large models exclusive access.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TextIO

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledModel:
	"""A model queued for activation generation."""

	name: str
	n_params: int
	estimated_vram: int  # bytes


@dataclass
class RunningModel:
	"""A model currently being processed in a subprocess."""

	model: ScheduledModel
	process: subprocess.Popen[str]
	device: str
	cpu_cores: list[int]
	log_file: TextIO
	log_path: str
	start_time: float


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------


def estimate_vram_bytes(
	n_params: int,
	safety_factor: float = 3.0,
	dtype_bytes: int = 4,
) -> int:
	"""Estimate VRAM needed for inference with activation caching.

	The estimate accounts for model weights (``n_params * dtype_bytes``)
	multiplied by a ``safety_factor`` to cover activation cache, intermediate
	tensors, and CUDA allocator overhead.
	"""
	return int(n_params * dtype_bytes * safety_factor)


# ---------------------------------------------------------------------------
# CPU core pool
# ---------------------------------------------------------------------------


@dataclass
class CorePool:
	"""Manages allocation of CPU cores to subprocess workers."""

	available: list[int] = field(default_factory=list)

	@classmethod
	def from_system(cls, total_cores: int | None = None) -> "CorePool":
		"""Create a pool with all system cores."""
		n: int = total_cores if total_cores is not None else (os.cpu_count() or 1)
		return cls(available=list(range(n)))

	def allocate(self, n: int) -> list[int]:
		"""Allocate *n* cores from the pool. Returns fewer if not enough."""
		n_actual: int = min(n, len(self.available))
		if n_actual == 0:
			return []
		allocated: list[int] = self.available[:n_actual]
		self.available = self.available[n_actual:]
		return allocated

	def release(self, cores: list[int]) -> None:
		"""Return cores to the pool."""
		self.available.extend(cores)
		self.available.sort()

	@property
	def n_available(self) -> int:
		return len(self.available)


# ---------------------------------------------------------------------------
# GPU VRAM query
# ---------------------------------------------------------------------------


def _parse_device_index(device: str) -> int:
	"""Extract the numeric device index from a CUDA device string."""
	if ":" in device:
		return int(device.split(":")[1])
	# "cuda" without index → device 0
	return 0


def get_free_vram(device: str) -> int:
	"""Return free VRAM in bytes for the given CUDA device.

	This queries the CUDA driver directly, so it reflects memory used by
	*all* processes on the device (not just the current one).
	"""
	device_idx: int = _parse_device_index(device)
	free: int
	free, _ = torch.cuda.mem_get_info(device_idx)
	return free


def get_total_vram(device: str) -> int:
	"""Return total VRAM in bytes for the given CUDA device."""
	device_idx: int = _parse_device_index(device)
	_free: int
	total: int
	_free, total = torch.cuda.mem_get_info(device_idx)
	return total


# ---------------------------------------------------------------------------
# Subprocess builder
# ---------------------------------------------------------------------------


def _build_subprocess_cmd(
	model_name: str,
	device: str,
	save_path: str,
	prompts_path: str,
	n_samples: int,
	min_chars: int,
	max_chars: int,
	force: bool,
	cpu_cores: list[int],
	batch_size: int = 32,
) -> list[str]:
	"""Build the command list to run pattern_lens.activations for one model.

	On Linux, wraps the command with ``taskset`` for CPU affinity.
	"""
	python_cmd: list[str] = [
		sys.executable,
		"-m",
		"pattern_lens.activations",
		"--model",
		model_name,
		"--device",
		device,
		"--save-path",
		save_path,
		"--prompts",
		prompts_path,
		"--raw-prompts",
		"--min-chars",
		str(min_chars),
		"--max-chars",
		str(max_chars),
		"--n-samples",
		str(n_samples),
		"--batch-size",
		str(batch_size),
	]

	if force:
		python_cmd.append("--force")

	# on Linux, pin to specific CPU cores via taskset
	if platform.system() == "Linux" and cpu_cores:
		core_list: str = ",".join(str(c) for c in cpu_cores)
		return ["taskset", "-c", core_list, *python_cmd]

	return python_cmd


def _build_subprocess_env(n_threads: int) -> dict[str, str]:
	"""Build environment variables for the subprocess.

	Sets thread count limits so each subprocess uses a proportional share
	of CPU resources instead of spawning threads for all cores.
	"""
	env: dict[str, str] = os.environ.copy()
	thread_str: str = str(max(1, n_threads))
	env["OMP_NUM_THREADS"] = thread_str
	env["MKL_NUM_THREADS"] = thread_str
	# avoid uv workspace lock contention between subprocesses
	env["UV_NOSYNC"] = "1"
	# strip CUDA_VISIBLE_DEVICES so subprocesses can see all GPUs —
	# the scheduler passes --device explicitly to each subprocess
	env.pop("CUDA_VISIBLE_DEVICES", None)
	return env


# ---------------------------------------------------------------------------
# Log tailing / progress parsing
# ---------------------------------------------------------------------------

_ANSI_RE: re.Pattern[str] = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# matches standard tqdm output:  desc: NN%|bar| cur/total [elapsed<remaining, rate]
_TQDM_RE: re.Pattern[str] = re.compile(
	r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
	r"\s*\[(?P<elapsed>[^<]+)<(?P<remain>[^,]+),\s*(?P<rate>[^\]]+)\]"
)


def _format_elapsed(seconds: float) -> str:
	"""Format elapsed seconds as ``M:SS`` or ``H:MM:SS``."""
	total_s: int = int(seconds)
	hours: int = total_s // 3600
	minutes: int = (total_s % 3600) // 60
	secs: int = total_s % 60
	if hours > 0:
		return f"{hours}:{minutes:02d}:{secs:02d}"
	return f"{minutes}:{secs:02d}"


def _tail_log(log_path: str, max_bytes: int = 4096) -> str:
	"""Read the last meaningful line from a subprocess log file.

	Opens a separate read handle (independent of the write handle held by
	Popen), seeks near the end, and returns the last non-empty content.
	Handles tqdm's ``\\r``-delimited progress by splitting on ``\\r`` and
	taking the last segment.

	Returns ``""`` if the file is empty or unreadable.
	"""
	try:
		with open(log_path, "r", errors="replace") as f:
			f.seek(0, 2)
			size: int = f.tell()
			if size == 0:
				return ""
			f.seek(max(0, size - max_bytes))
			data: str = f.read()
	except OSError:
		return ""

	# find the last non-empty line
	lines: list[str] = data.split("\n")
	last_line: str = ""
	for line in reversed(lines):
		stripped: str = line.strip()
		if stripped:
			last_line = stripped
			break

	if not last_line:
		return ""

	# tqdm uses \r for in-place updates — take the last segment
	if "\r" in last_line:
		segments: list[str] = last_line.split("\r")
		for seg in reversed(segments):
			seg_stripped: str = seg.strip()
			if seg_stripped:
				last_line = seg_stripped
				break

	# strip ANSI escape codes
	result: str = _ANSI_RE.sub("", last_line)
	return result[:200]


def _parse_tqdm(line: str) -> tuple[int, int, str] | None:
	"""Parse a tqdm progress line.

	Returns ``(current, total, rate)`` or ``None`` if not a tqdm line.
	"""
	match: re.Match[str] | None = _TQDM_RE.search(line)
	if match is None:
		return None
	current: int = int(match.group("cur"))
	total: int = int(match.group("tot"))
	rate: str = match.group("rate").strip()
	return current, total, rate


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

# minimum cores to assign per subprocess (avoid starving workers)
_MIN_CORES_PER_PROCESS: int = 2
# how long to sleep when no model can be scheduled
_POLL_INTERVAL_SECONDS: float = 2.0


class ModelScheduler:
	"""VRAM-aware scheduler that runs model activation generation in parallel.

	Models are sorted largest-first (greedy bin-packing). The scheduler
	continuously polls GPU free VRAM and spawns subprocesses for models
	that fit, distributing CPU cores evenly across running workers.
	"""

	def __init__(
		self,
		models: list[ScheduledModel],
		devices: list[str],
		prompts_path: str,
		save_path: str,
		n_samples: int,
		min_chars: int,
		max_chars: int,
		force: bool,
		total_cpu_cores: int | None = None,
		batch_size: int = 32,
	) -> None:
		# sort largest first for greedy bin-packing
		self.pending: list[ScheduledModel] = sorted(
			models, key=lambda m: m.estimated_vram, reverse=True
		)
		self.devices: list[str] = devices
		self.prompts_path: str = prompts_path
		self.save_path: str = save_path
		self.n_samples: int = n_samples
		self.min_chars: int = min_chars
		self.max_chars: int = max_chars
		self.force: bool = force
		self.batch_size: int = batch_size

		self.running: list[RunningModel] = []
		self.completed: list[str] = []
		self.failed: list[tuple[str, int]] = []  # (model_name, exit_code)

		# track VRAM committed to running (but possibly not yet loaded) models
		self._device_committed: dict[str, int] = {d: 0 for d in devices}

		self._total_cores: int = (
			total_cpu_cores if total_cpu_cores is not None else (os.cpu_count() or 1)
		)
		self.core_pool: CorePool = CorePool.from_system(total_cpu_cores)
		self.total_models: int = len(models)

	def _log(self, msg: str) -> None:
		"""Print a scheduler status message."""
		print(f"\033[93m[scheduler] {msg}\033[m")

	def _poll_running(self) -> None:
		"""Check for completed subprocesses and collect results."""
		still_running: list[RunningModel] = []
		for rm in self.running:
			retcode: int | None = rm.process.poll()
			if retcode is None:
				still_running.append(rm)
				continue

			# process finished
			rm.log_file.close()
			self.core_pool.release(rm.cpu_cores)
			self._device_committed[rm.device] -= rm.model.estimated_vram

			if retcode == 0:
				elapsed_str: str = _format_elapsed(time.monotonic() - rm.start_time)
				self.completed.append(rm.model.name)
				self._log(
					f"completed {rm.model.name} on {rm.device} "
					f"in {elapsed_str} "
					f"({len(self.completed)}/{self.total_models} done)"
				)
			else:
				elapsed_str = _format_elapsed(time.monotonic() - rm.start_time)
				self.failed.append((rm.model.name, retcode))
				self._log(
					f"FAILED {rm.model.name} on {rm.device} "
					f"in {elapsed_str} (exit code {retcode}), "
					f"see log: {rm.log_path}"
				)

		self.running = still_running

	def _find_best_fit(self) -> tuple[ScheduledModel, str] | None:
		"""Find the largest pending model that fits on any device.

		Returns ``(model, device)`` or ``None`` if nothing fits.
		"""
		# need at least some cores for a new process
		if self.core_pool.n_available < _MIN_CORES_PER_PROCESS:
			return None

		# collect effective free VRAM per device, accounting for committed
		# VRAM from models that have been spawned but not yet loaded
		device_free: dict[str, int] = {}
		for device in self.devices:
			try:
				reported_free: int = get_free_vram(device)
				total: int = get_total_vram(device)
				committed: int = self._device_committed.get(device, 0)
				# conservative: use whichever is lower — driver-reported free,
				# or total minus what we've committed to running models
				effective_free: int = min(reported_free, total - committed)
				device_free[device] = max(0, effective_free)
			except (RuntimeError, ValueError) as e:
				self._log(f"warning: could not query VRAM for {device}: {e}")
				continue

		if not device_free:
			return None

		# pending is already sorted largest-first, so first fit = best fit
		for model in self.pending:
			for device, free in device_free.items():
				if model.estimated_vram <= free:
					return model, device

		return None

	def _spawn_model(self, model: ScheduledModel, device: str) -> None:
		"""Spawn a subprocess for a single model."""
		# allocate CPU cores: fair share based on total models remaining
		n_models_remaining: int = len(self.running) + len(self.pending)
		cores_per_process: int = min(
			max(
				_MIN_CORES_PER_PROCESS,
				self._total_cores // max(1, n_models_remaining),
			),
			self.core_pool.n_available,
		)
		cpu_cores: list[int] = self.core_pool.allocate(cores_per_process)

		cmd: list[str] = _build_subprocess_cmd(
			model_name=model.name,
			device=device,
			save_path=self.save_path,
			prompts_path=self.prompts_path,
			n_samples=self.n_samples,
			min_chars=self.min_chars,
			max_chars=self.max_chars,
			force=self.force,
			cpu_cores=cpu_cores,
			batch_size=self.batch_size,
		)
		env: dict[str, str] = _build_subprocess_env(n_threads=len(cpu_cores))

		vram_mb: float = model.estimated_vram / (1024 * 1024)
		self._log(
			f"starting {model.name} on {device} "
			f"(est. {vram_mb:.0f}MB VRAM, cores {cpu_cores})"
		)

		# log subprocess output to a file
		log_path: str = f"{self.save_path}/{model.name.replace('/', '_')}_parallel.log"
		log_file: TextIO = open(log_path, "w")  # noqa: SIM115
		try:
			process: subprocess.Popen[str] = subprocess.Popen(
				cmd,
				env=env,
				stdout=log_file,
				stderr=subprocess.STDOUT,
				text=True,
			)
		except OSError:
			log_file.close()
			self.core_pool.release(cpu_cores)
			raise

		self.pending.remove(model)
		self._device_committed[device] += model.estimated_vram
		self.running.append(
			RunningModel(
				model=model,
				process=process,
				device=device,
				cpu_cores=cpu_cores,
				log_file=log_file,
				log_path=log_path,
				start_time=time.monotonic(),
			)
		)

	def _cleanup_running(self) -> None:
		"""Terminate all running subprocesses and close log files."""
		for rm in self.running:
			try:
				rm.process.terminate()
			except OSError:
				pass
			try:
				rm.log_file.close()
			except OSError:
				pass
		if self.running:
			self._log(f"terminated {len(self.running)} running subprocess(es)")
		self.running.clear()

	def _print_status(self) -> None:
		"""Print a compact status block showing running models' progress."""
		if not self.running:
			return

		now: float = time.monotonic()
		n_done: int = len(self.completed) + len(self.failed)
		self._log(
			f"--- status: {len(self.running)} running, "
			f"{len(self.pending)} pending, "
			f"{n_done}/{self.total_models} done ---"
		)

		for rm in self.running:
			elapsed_str: str = _format_elapsed(now - rm.start_time)
			last_line: str = _tail_log(rm.log_path)
			progress: tuple[int, int, str] | None = _parse_tqdm(last_line)

			if progress is not None:
				current, total, rate = progress
				pct: float = 100.0 * current / total if total > 0 else 0.0
				status: str = f"{pct:3.0f}% ({current}/{total}) [{rate}]"
			elif last_line:
				status = last_line[:80]
			else:
				status = "(no output yet)"

			self._log(f"  {rm.model.name:<35s} {rm.device:<8s} {elapsed_str:>8s}  {status}")

	def run_all(self) -> None:
		"""Schedule and run all models. Blocks until all complete."""
		self._log(
			f"scheduling {self.total_models} models across devices {self.devices}"
		)
		self._log(f"CPU cores available: {self.core_pool.n_available}")

		try:
			while self.pending or self.running:
				self._poll_running()

				# try to schedule more models
				scheduled_any: bool = False
				while self.pending:
					fit: tuple[ScheduledModel, str] | None = self._find_best_fit()
					if fit is None:
						break
					model, device = fit
					self._spawn_model(model, device)
					scheduled_any = True
					# re-poll in case scheduling freed something
					self._poll_running()

				# guard: if nothing is running and nothing could be scheduled,
				# we're stuck — no point sleeping forever
				if self.pending and not self.running and not scheduled_any:
					self._log(
						f"ERROR: {len(self.pending)} model(s) cannot be scheduled "
						f"(insufficient VRAM or no reachable devices). Giving up."
					)
					for m in self.pending:
						self.failed.append((m.name, -1))
					self.pending.clear()
					break

				if self.running:
					self._print_status()
					time.sleep(_POLL_INTERVAL_SECONDS)
		except BaseException:
			self._log("interrupted — cleaning up running subprocesses")
			self._cleanup_running()
			raise

		# summary
		self._log(
			f"all done: {len(self.completed)} completed, {len(self.failed)} failed"
		)
		if self.failed:
			self._log("failed models:")
			for name, code in self.failed:
				self._log(f"  {name} (exit code {code})")
