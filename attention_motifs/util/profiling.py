"""Profile PyTorch training loops for both time and memory usage."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from time import time
import cProfile
import pstats
import warnings
import psutil
import gc
from contextlib import contextmanager, nullcontext
from enum import Enum, auto

import torch
from torch.profiler import profile, ProfilerActivity
import numpy as np
from typing import Sequence


class ProfilerMode(Enum):
	"""Different profiling modes available."""

	TIME = auto()  # Just time profiling
	MEMORY = auto()  # Just memory profiling
	FULL = auto()  # Both time and memory


@dataclass
class MemorySnapshot:
	"""Single snapshot of memory usage."""

	timestamp: float
	cpu_used: int
	cpu_percent: float
	gpu_allocated: Optional[int] = None
	gpu_reserved: Optional[int] = None

	@property
	def cpu_used_mb(self) -> float:
		"""CPU memory used in MB."""
		return self.cpu_used / (1024 * 1024)

	@property
	def gpu_allocated_mb(self) -> Optional[float]:
		"""GPU memory allocated in MB."""
		if self.gpu_allocated is None:
			return None
		return self.gpu_allocated / (1024 * 1024)

	@property
	def gpu_reserved_mb(self) -> Optional[float]:
		"""GPU memory reserved in MB."""
		if self.gpu_reserved is None:
			return None
		return self.gpu_reserved / (1024 * 1024)


@dataclass
class MemoryStats:
	"""Statistics about memory usage over time."""

	snapshots: list[MemorySnapshot] = field(default_factory=list)

	@property
	def cpu_peak_mb(self) -> float:
		"""Peak CPU memory usage in MB."""
		return max(s.cpu_used_mb for s in self.snapshots)

	@property
	def gpu_peak_mb(self) -> Optional[float]:
		"""Peak GPU memory usage in MB."""
		gpu_mems = [
			s.gpu_allocated_mb for s in self.snapshots if s.gpu_allocated_mb is not None
		]
		return max(gpu_mems) if gpu_mems else None

	def summarize(self) -> str:
		"""Get a string summary of memory usage."""
		lines = ["Memory Usage Summary:"]
		lines.append(f"Peak CPU: {self.cpu_peak_mb:.1f} MB")

		if self.gpu_peak_mb is not None:
			lines.append(f"Peak GPU: {self.gpu_peak_mb:.1f} MB")

		# Add percentile stats for both CPU and GPU
		cpu_mbs = [s.cpu_used_mb for s in self.snapshots]
		lines.append("\nCPU Memory (MB):")
		lines.extend(self._percentile_stats(cpu_mbs))

		gpu_mbs = [
			s.gpu_allocated_mb for s in self.snapshots if s.gpu_allocated_mb is not None
		]
		if gpu_mbs:
			lines.append("\nGPU Memory (MB):")
			lines.extend(self._percentile_stats(gpu_mbs))

		return "\n".join(lines)

	@staticmethod
	def _percentile_stats(
		values: Sequence[float], percentiles: Sequence[int] = (0, 25, 50, 75, 100)
	) -> list[str]:
		"""Get percentile statistics for a sequence of values."""
		if not values:
			return ["  No data available"]

		stats = np.percentile(values, percentiles)
		return [f"  {p}th percentile: {v:.1f}" for p, v in zip(percentiles, stats)]


@dataclass
class ProfilingResult:
	"""Results from a profiling run."""

	elapsed_time: float
	memory_stats: MemoryStats
	cpu_profile: Optional[pstats.Stats] = None
	gpu_trace_path: Optional[Path] = None

	def summarize(self) -> str:
		"""Get a string summary of the profiling results."""
		lines = [
			"=" * 50,
			"PROFILING RESULTS",
			"=" * 50,
			f"\nTotal time: {self.elapsed_time:.2f} seconds",
			"\n" + self.memory_stats.summarize(),
		]
		return "\n".join(lines)


class TrainingProfiler:
	"""Profile PyTorch training loops.

	# Usage:
	```python
	profiler = TrainingProfiler()

	# Profile the whole training loop
	with profiler.profile("training"):
		for epoch in range(n_epochs):
			for batch in dataloader:
				# Take a memory snapshot at specific points
				profiler.snapshot()

				# Your training code here
				...

	# Print results
	profiler.print_summary()
	```
	"""

	def __init__(
		self, mode: ProfilerMode = ProfilerMode.FULL, output_dir: Optional[Path] = None
	):
		"""Initialize the profiler.

		# Parameters:
		- `mode: ProfilerMode`
			What to profile (defaults to FULL)
		- `output_dir: Optional[Path]`
			Where to save detailed profiling data (defaults to None)
		"""
		self.mode = mode
		self.output_dir = Path(output_dir) if output_dir else None
		if self.output_dir:
			self.output_dir.mkdir(parents=True, exist_ok=True)

		# Results storage
		self.results: dict[str, ProfilingResult] = {}
		self._current_memory_stats: Optional[MemoryStats] = None
		self._start_time: Optional[float] = None

	def snapshot(self) -> None:
		"""Take a snapshot of current memory usage."""
		if self._current_memory_stats is None:
			return

		# Get CPU memory
		process = psutil.Process()
		snapshot = MemorySnapshot(
			timestamp=time() - (self._start_time or 0),
			cpu_used=process.memory_info().rss,
			cpu_percent=process.cpu_percent(),
		)

		# Get GPU memory if available
		if torch.cuda.is_available():
			snapshot.gpu_allocated = torch.cuda.memory_allocated()
			snapshot.gpu_reserved = torch.cuda.memory_reserved()

		self._current_memory_stats.snapshots.append(snapshot)

	@contextmanager
	def profile(self, name: str) -> Iterator[None]:
		"""Context manager for profiling a code block.

		# Parameters:
		- `name: str`
			Name for this profiling session
		"""
		# Setup
		self._start_time = time()
		self._current_memory_stats = MemoryStats()

		# Clear memory before starting
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.reset_peak_memory_stats()

		# Take initial snapshot
		self.snapshot()

		# Setup profilers based on mode
		cpu_profiler = (
			cProfile.Profile()
			if self.mode in (ProfilerMode.TIME, ProfilerMode.FULL)
			else None
		)
		if cpu_profiler:
			cpu_profiler.enable()

		gpu_profiler_ctx = (
			profile(
				activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
				record_shapes=True,
				profile_memory=True,
				with_stack=True,
				with_flops=True,
			)
			if torch.cuda.is_available()
			and self.mode in (ProfilerMode.TIME, ProfilerMode.FULL)
			else nullcontext()
		)

		try:
			with gpu_profiler_ctx as gpu_prof:
				yield
		except Exception as e:
			warnings.warn(f"Error during profiling: {e}")
		finally:
			# Cleanup
			if cpu_profiler:
				cpu_profiler.disable()

			# Save results
			elapsed = time() - self._start_time

			# Save GPU trace if available
			gpu_trace_path = None
			if self.output_dir and gpu_prof:  # pyright: ignore[reportPossiblyUnboundVariable]
				gpu_trace_path = self.output_dir / f"{name}_trace.json"

				try:
					gpu_prof.export_chrome_trace(gpu_trace_path.as_posix())  # pyright: ignore[reportPossiblyUnboundVariable]
				except Exception as e:
					warnings.warn(f"Error saving GPU trace: {e}")

			# Convert CPU profile to stats if available
			cpu_stats = None
			if cpu_profiler:
				cpu_stats = pstats.Stats(cpu_profiler)
				if self.output_dir:
					cpu_stats.dump_stats(self.output_dir / f"{name}_profile.stats")

			# Store results
			self.results[name] = ProfilingResult(
				elapsed_time=elapsed,
				memory_stats=self._current_memory_stats or MemoryStats(),
				cpu_profile=cpu_stats,
				gpu_trace_path=gpu_trace_path,
			)

			# Clear current stats
			self._current_memory_stats = None
			self._start_time = None

	def print_summary(self, name: Optional[str] = None) -> None:
		"""Print a summary of profiling results.

		# Parameters:
		- `name: Optional[str]`
			Name of specific profile to summarize (default: all profiles)
		"""
		if name:
			if name not in self.results:
				print(f"No results found for '{name}'")
				return
			print(self.results[name].summarize())
		else:
			for name, result in self.results.items():
				print(f"\nProfile: {name}")
				print(result.summarize())
