"""VRAM-aware parallel model scheduler for s1 activation generation.

Spawns subprocesses running ``python -m pattern_lens.activations`` for each
model, packing multiple small models onto the same GPU when they fit and
giving large models exclusive access.
"""

from __future__ import annotations

import os
import platform
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
	return env


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

		self.running: list[RunningModel] = []
		self.completed: list[str] = []
		self.failed: list[tuple[str, int]] = []  # (model_name, exit_code)

		# track VRAM committed to running (but possibly not yet loaded) models
		self._device_committed: dict[str, int] = {d: 0 for d in devices}

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
				self.completed.append(rm.model.name)
				self._log(
					f"completed {rm.model.name} on {rm.device} "
					f"({len(self.completed)}/{self.total_models} done)"
				)
			else:
				self.failed.append((rm.model.name, retcode))
				self._log(
					f"FAILED {rm.model.name} on {rm.device} (exit code {retcode})"
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
		# allocate CPU cores: divide available cores among (running + 1) processes
		n_running: int = len(self.running) + 1
		cores_per_process: int = max(
			_MIN_CORES_PER_PROCESS,
			self.core_pool.n_available // max(1, n_running),
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
			)
		)

	def run_all(self) -> None:
		"""Schedule and run all models. Blocks until all complete."""
		self._log(
			f"scheduling {self.total_models} models across devices {self.devices}"
		)
		self._log(f"CPU cores available: {self.core_pool.n_available}")

		while self.pending or self.running:
			self._poll_running()

			# try to schedule more models
			while self.pending:
				fit: tuple[ScheduledModel, str] | None = self._find_best_fit()
				if fit is None:
					break
				model, device = fit
				self._spawn_model(model, device)
				# re-poll in case scheduling freed something
				self._poll_running()

			if self.running:
				time.sleep(_POLL_INTERVAL_SECONDS)

		# summary
		self._log(
			f"all done: {len(self.completed)} completed, {len(self.failed)} failed"
		)
		if self.failed:
			self._log("failed models:")
			for name, code in self.failed:
				self._log(f"  {name} (exit code {code})")
