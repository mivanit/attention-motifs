"""Tests for model_table, model_scheduler, and parallel scheduling config/integration."""

from __future__ import annotations

import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from attention_motifs.pipeline.model_table import (
	ModelInfo,
	_download_csv,
	_parse_csv,
	fetch_model_table,
	get_model_params,
)
from attention_motifs.pipeline.model_scheduler import (
	CorePool,
	ModelScheduler,
	RunningModel,
	ScheduledModel,
	_build_subprocess_cmd,
	_build_subprocess_env,
	_parse_device_index,
	estimate_vram_bytes,
	get_free_vram,
	get_total_vram,
)
from attention_motifs.pipeline.cfg import PipelineConfig


# ---------------------------------------------------------------------------
# Sample CSV for model_table tests
# ---------------------------------------------------------------------------

SAMPLE_CSV: str = """\
name.default_alias,n_params.as_int,something_else
gpt2-small,85000000,ignored
pythia-14m,14000000,ignored
pythia-2.8b,2500000000,ignored
"""

SAMPLE_CSV_WITH_EMPTY: str = """\
name.default_alias,n_params.as_int,something_else
gpt2-small,85000000,ignored
,,ignored
pythia-14m,14000000,ignored
,12345,ignored
missing-params,,ignored
"""


# ===========================================================================
# model_table.py tests
# ===========================================================================


class TestParseCSV:
	def test_parse_csv(self) -> None:
		"""Parse sample CSV content, verify ModelInfo values."""
		table: dict[str, ModelInfo] = _parse_csv(SAMPLE_CSV)
		assert len(table) == 3
		assert table["gpt2-small"] == ModelInfo(name="gpt2-small", n_params=85_000_000)
		assert table["pythia-14m"] == ModelInfo(name="pythia-14m", n_params=14_000_000)
		assert table["pythia-2.8b"] == ModelInfo(
			name="pythia-2.8b", n_params=2_500_000_000
		)

	def test_parse_csv_skips_empty_rows(self) -> None:
		"""Rows with missing name or n_params are skipped."""
		table: dict[str, ModelInfo] = _parse_csv(SAMPLE_CSV_WITH_EMPTY)
		assert len(table) == 2
		assert "gpt2-small" in table
		assert "pythia-14m" in table


class TestGetModelParams:
	def test_get_model_params_found(self) -> None:
		"""Lookup known model returns correct count."""
		table: dict[str, ModelInfo] = _parse_csv(SAMPLE_CSV)
		assert get_model_params("gpt2-small", table) == 85_000_000

	def test_get_model_params_not_found(self) -> None:
		"""KeyError for unknown model."""
		table: dict[str, ModelInfo] = _parse_csv(SAMPLE_CSV)
		with pytest.raises(KeyError, match="nonexistent-model"):
			get_model_params("nonexistent-model", table)


class TestFetchModelTable:
	def test_fetch_model_table_from_cache(self, tmp_path: Path) -> None:
		"""Reads cache file, no network call."""
		cache_path: Path = tmp_path / "model_table.csv"
		cache_path.write_text(SAMPLE_CSV)

		with (
			patch(
				"attention_motifs.pipeline.model_table.MODEL_TABLE_CACHE", cache_path
			),
			patch(
				"attention_motifs.pipeline.model_table._download_csv"
			) as mock_download,
		):
			table: dict[str, ModelInfo] = fetch_model_table()
			mock_download.assert_not_called()
			assert "gpt2-small" in table

	def test_fetch_model_table_downloads(self, tmp_path: Path) -> None:
		"""Mock urlopen, verify download + cache write."""
		cache_path: Path = tmp_path / "model_table.csv"
		# cache doesn't exist yet, so it should download

		with (
			patch(
				"attention_motifs.pipeline.model_table.MODEL_TABLE_CACHE", cache_path
			),
			patch(
				"attention_motifs.pipeline.model_table._download_csv",
				return_value=SAMPLE_CSV,
			) as mock_download,
		):
			table: dict[str, ModelInfo] = fetch_model_table()
			mock_download.assert_called_once()
			assert "gpt2-small" in table

	def test_fetch_model_table_force_refresh(self, tmp_path: Path) -> None:
		"""Cache exists but force_refresh=True → _download_csv still called."""
		cache_path: Path = tmp_path / "model_table.csv"
		cache_path.write_text(SAMPLE_CSV)

		with (
			patch(
				"attention_motifs.pipeline.model_table.MODEL_TABLE_CACHE", cache_path
			),
			patch(
				"attention_motifs.pipeline.model_table._download_csv",
				return_value=SAMPLE_CSV,
			) as mock_download,
		):
			table: dict[str, ModelInfo] = fetch_model_table(force_refresh=True)
			mock_download.assert_called_once()
			assert "gpt2-small" in table


class TestDownloadCSV:
	@patch("attention_motifs.pipeline.model_table.urllib.request.urlopen")
	def test_download_csv_network_failure(
		self, mock_urlopen: MagicMock, tmp_path: Path
	) -> None:
		"""urllib.request.urlopen raises URLError → propagates."""
		mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
		with pytest.raises(urllib.error.URLError, match="Connection refused"):
			_download_csv("https://example.com/table.csv", tmp_path / "cache.csv")


# ===========================================================================
# model_scheduler.py tests
# ===========================================================================


class TestEstimateVramBytes:
	def test_estimate_vram_default(self) -> None:
		"""1M params → 12MB at safety_factor=3.0."""
		result: int = estimate_vram_bytes(1_000_000)
		assert result == 1_000_000 * 4 * 3  # 12_000_000

	def test_estimate_vram_custom_factor(self) -> None:
		"""Custom safety factor."""
		result: int = estimate_vram_bytes(1_000_000, safety_factor=2.0)
		assert result == 1_000_000 * 4 * 2  # 8_000_000


class TestCorePool:
	def test_core_pool_allocate(self) -> None:
		"""Allocate N cores, verify removed from available."""
		pool: CorePool = CorePool(available=list(range(8)))
		allocated: list[int] = pool.allocate(3)
		assert allocated == [0, 1, 2]
		assert pool.available == [3, 4, 5, 6, 7]

	def test_core_pool_allocate_more_than_available(self) -> None:
		"""Returns fewer cores when not enough available."""
		pool: CorePool = CorePool(available=[0, 1])
		allocated: list[int] = pool.allocate(5)
		assert allocated == [0, 1]
		assert pool.available == []

	def test_core_pool_release(self) -> None:
		"""Released cores go back sorted."""
		pool: CorePool = CorePool(available=[4, 5])
		pool.release([0, 2])
		assert pool.available == [0, 2, 4, 5]

	def test_core_pool_allocate_zero(self) -> None:
		"""Allocating 0 cores returns empty list, pool unchanged."""
		pool: CorePool = CorePool(available=[0, 1, 2, 3])
		allocated: list[int] = pool.allocate(0)
		assert allocated == []
		assert pool.available == [0, 1, 2, 3]
		assert pool.n_available == 4

	def test_core_pool_from_system(self) -> None:
		"""Creates with correct count."""
		pool: CorePool = CorePool.from_system(total_cores=4)
		assert pool.available == [0, 1, 2, 3]
		assert pool.n_available == 4


class TestBuildSubprocessCmd:
	def test_cmd_basic(self) -> None:
		"""All args present in correct order."""
		cmd: list[str] = _build_subprocess_cmd(
			model_name="gpt2-small",
			device="cuda:0",
			save_path="/data/patterns",
			prompts_path="/data/prompts.jsonl",
			n_samples=128,
			min_chars=64,
			max_chars=512,
			force=False,
			cpu_cores=[],
		)
		assert "--model" in cmd
		idx: int = cmd.index("--model")
		assert cmd[idx + 1] == "gpt2-small"
		assert "--device" in cmd
		assert "--save-path" in cmd
		assert "--prompts" in cmd
		assert "--raw-prompts" in cmd
		assert "--n-samples" in cmd
		assert "--batch-size" in cmd
		bs_idx: int = cmd.index("--batch-size")
		assert cmd[bs_idx + 1] == "32"  # default
		assert "--force" not in cmd

	def test_cmd_includes_batch_size(self) -> None:
		"""Custom batch_size appears in command."""
		cmd: list[str] = _build_subprocess_cmd(
			model_name="gpt2-small",
			device="cuda:0",
			save_path="/data",
			prompts_path="/data/p.jsonl",
			n_samples=10,
			min_chars=10,
			max_chars=100,
			force=False,
			cpu_cores=[],
			batch_size=64,
		)
		assert "--batch-size" in cmd
		idx: int = cmd.index("--batch-size")
		assert cmd[idx + 1] == "64"

	def test_cmd_with_force(self) -> None:
		"""Includes --force."""
		cmd: list[str] = _build_subprocess_cmd(
			model_name="gpt2-small",
			device="cuda:0",
			save_path="/data",
			prompts_path="/data/p.jsonl",
			n_samples=10,
			min_chars=10,
			max_chars=100,
			force=True,
			cpu_cores=[],
		)
		assert "--force" in cmd

	@patch(
		"attention_motifs.pipeline.model_scheduler.platform.system",
		return_value="Linux",
	)
	def test_cmd_taskset_linux(self, _mock_sys: MagicMock) -> None:
		"""On Linux with cpu_cores, verify taskset prefix."""
		cmd: list[str] = _build_subprocess_cmd(
			model_name="gpt2-small",
			device="cuda:0",
			save_path="/data",
			prompts_path="/data/p.jsonl",
			n_samples=10,
			min_chars=10,
			max_chars=100,
			force=False,
			cpu_cores=[0, 1, 2],
		)
		assert cmd[0] == "taskset"
		assert cmd[1] == "-c"
		assert cmd[2] == "0,1,2"

	@patch(
		"attention_motifs.pipeline.model_scheduler.platform.system",
		return_value="Darwin",
	)
	def test_cmd_no_taskset_non_linux(self, _mock_sys: MagicMock) -> None:
		"""No taskset on macOS."""
		cmd: list[str] = _build_subprocess_cmd(
			model_name="gpt2-small",
			device="cuda:0",
			save_path="/data",
			prompts_path="/data/p.jsonl",
			n_samples=10,
			min_chars=10,
			max_chars=100,
			force=False,
			cpu_cores=[0, 1],
		)
		assert cmd[0] != "taskset"


class TestBuildSubprocessEnv:
	def test_env_threads(self) -> None:
		"""OMP_NUM_THREADS and MKL_NUM_THREADS set correctly."""
		env: dict[str, str] = _build_subprocess_env(n_threads=4)
		assert env["OMP_NUM_THREADS"] == "4"
		assert env["MKL_NUM_THREADS"] == "4"

	def test_env_uv_nosync(self) -> None:
		"""UV_NOSYNC=1 set to avoid uv lock contention."""
		env: dict[str, str] = _build_subprocess_env(n_threads=1)
		assert env["UV_NOSYNC"] == "1"

	@patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"})
	def test_env_strips_cuda_visible_devices(self) -> None:
		"""CUDA_VISIBLE_DEVICES removed so subprocesses see all GPUs."""
		env: dict[str, str] = _build_subprocess_env(n_threads=4)
		assert "CUDA_VISIBLE_DEVICES" not in env


class TestParseDeviceIndex:
	def test_with_index(self) -> None:
		assert _parse_device_index("cuda:0") == 0

	def test_without_index(self) -> None:
		assert _parse_device_index("cuda") == 0

	def test_device_1(self) -> None:
		assert _parse_device_index("cuda:1") == 1

	def test_parse_device_index_no_colon(self) -> None:
		"""'cpu' has no colon — returns default 0 (documents behavior)."""
		assert _parse_device_index("cpu") == 0

	def test_parse_device_index_non_numeric(self) -> None:
		"""'cuda:abc' — int('abc') raises ValueError."""
		with pytest.raises(ValueError):
			_parse_device_index("cuda:abc")


class TestGetFreeVram:
	@patch("attention_motifs.pipeline.model_scheduler.torch.cuda.mem_get_info")
	def test_get_free_vram_with_index(self, mock_mem: MagicMock) -> None:
		"""'cuda:0' → device index 0."""
		mock_mem.return_value = (4_000_000_000, 24_000_000_000)
		result: int = get_free_vram("cuda:0")
		mock_mem.assert_called_once_with(0)
		assert result == 4_000_000_000

	@patch("attention_motifs.pipeline.model_scheduler.torch.cuda.mem_get_info")
	def test_get_free_vram_without_index(self, mock_mem: MagicMock) -> None:
		"""'cuda' → device index 0."""
		mock_mem.return_value = (4_000_000_000, 24_000_000_000)
		result: int = get_free_vram("cuda")
		mock_mem.assert_called_once_with(0)
		assert result == 4_000_000_000

	@patch("attention_motifs.pipeline.model_scheduler.torch.cuda.mem_get_info")
	def test_get_total_vram(self, mock_mem: MagicMock) -> None:
		"""get_total_vram returns second element."""
		mock_mem.return_value = (4_000_000_000, 24_000_000_000)
		result: int = get_total_vram("cuda:0")
		assert result == 24_000_000_000


# ===========================================================================
# ModelScheduler tests
# ===========================================================================


def _make_scheduled_model(
	name: str = "gpt2-small", n_params: int = 85_000_000, safety_factor: float = 3.0
) -> ScheduledModel:
	"""Helper to create a ScheduledModel with estimated VRAM."""
	return ScheduledModel(
		name=name,
		n_params=n_params,
		estimated_vram=estimate_vram_bytes(n_params, safety_factor),
	)


def _make_scheduler(
	models: list[ScheduledModel] | None = None,
	devices: list[str] | None = None,
	total_cpu_cores: int = 8,
) -> ModelScheduler:
	"""Helper to create a ModelScheduler with sensible defaults."""
	if models is None:
		models = [_make_scheduled_model()]
	if devices is None:
		devices = ["cuda:0"]
	return ModelScheduler(
		models=models,
		devices=devices,
		prompts_path="/data/prompts.jsonl",
		save_path="/tmp/test_patterns",
		n_samples=10,
		min_chars=10,
		max_chars=100,
		force=False,
		total_cpu_cores=total_cpu_cores,
	)


class TestFindBestFit:
	"""Tests for ModelScheduler._find_best_fit with mocked VRAM queries."""

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	def test_find_fit_model_fits(
		self, _mock_free: MagicMock, _mock_total: MagicMock
	) -> None:
		"""Enough VRAM → returns (model, device)."""
		small_model: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[small_model])
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is not None
		assert result[0] == small_model
		assert result[1] == "cuda:0"

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=4_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=100_000_000,
	)
	def test_find_fit_model_too_big(
		self, _mock_free: MagicMock, _mock_total: MagicMock
	) -> None:
		"""Not enough VRAM → returns None."""
		# pythia-2.8b at 2.5B params → ~30GB estimated VRAM
		big_model: ScheduledModel = _make_scheduled_model("pythia-2.8b", 2_500_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[big_model])
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is None

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=2_000_000_000,
	)
	def test_find_fit_largest_first(
		self, _mock_free: MagicMock, _mock_total: MagicMock
	) -> None:
		"""Returns largest model that fits, not smallest."""
		tiny: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		small: ScheduledModel = _make_scheduled_model("pythia-70m", 70_000_000)
		# tiny est ~168MB, small est ~840MB — both fit in 2GB
		scheduler: ModelScheduler = _make_scheduler(models=[tiny, small])
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is not None
		# scheduler sorts largest first, so small should be returned
		assert result[0] == small

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	def test_find_fit_no_cores(
		self, _mock_free: MagicMock, _mock_total: MagicMock
	) -> None:
		"""No available cores → returns None."""
		scheduler: ModelScheduler = _make_scheduler(total_cpu_cores=8)
		# exhaust all cores
		scheduler.core_pool.allocate(8)
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is None

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	def test_find_fit_insufficient_cores(
		self, _mock_free: MagicMock, _mock_total: MagicMock
	) -> None:
		"""1 core available but _MIN_CORES_PER_PROCESS is 2 → returns None."""
		scheduler: ModelScheduler = _make_scheduler(total_cpu_cores=8)
		# leave only 1 core — below the minimum of 2
		scheduler.core_pool.allocate(7)
		assert scheduler.core_pool.n_available == 1
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is None

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=24_000_000_000,
	)
	def test_find_fit_respects_committed_vram(
		self, _mock_free: MagicMock, _mock_total: MagicMock
	) -> None:
		"""Doesn't overschedule — committed VRAM reduces effective free."""
		# Model needs ~1GB estimated VRAM
		model: ScheduledModel = _make_scheduled_model("gpt2-small", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])

		# Pretend we've already committed 23.5GB on cuda:0
		scheduler._device_committed["cuda:0"] = 23_500_000_000

		# get_free_vram says 24GB free (stale — model not loaded yet)
		# but total - committed = 24GB - 23.5GB = 500MB
		# gpt2-small needs ~1020MB → should NOT fit
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is None

	@patch("attention_motifs.pipeline.model_scheduler.get_total_vram")
	@patch("attention_motifs.pipeline.model_scheduler.get_free_vram")
	def test_find_fit_vram_query_runtime_error(
		self, mock_free: MagicMock, mock_total: MagicMock
	) -> None:
		"""All devices raise RuntimeError → returns None."""
		mock_free.side_effect = RuntimeError("CUDA device not available")
		mock_total.side_effect = RuntimeError("CUDA device not available")

		model: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is None

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.get_free_vram")
	def test_find_fit_multiple_devices(
		self, mock_free: MagicMock, _mock_total: MagicMock
	) -> None:
		"""Picks device with enough free VRAM."""

		# cuda:0 has no space, cuda:1 has plenty
		def side_effect(device: str) -> int:
			if device == "cuda:0":
				return 100_000_000  # 100MB
			return 20_000_000_000  # 20GB

		mock_free.side_effect = side_effect

		model: ScheduledModel = _make_scheduled_model("gpt2-small", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(
			models=[model], devices=["cuda:0", "cuda:1"]
		)
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is not None
		assert result[1] == "cuda:1"

	@patch("attention_motifs.pipeline.model_scheduler.get_total_vram")
	@patch("attention_motifs.pipeline.model_scheduler.get_free_vram")
	def test_find_fit_partial_device_failure(
		self, mock_free: MagicMock, mock_total: MagicMock
	) -> None:
		"""cuda:0 raises RuntimeError, cuda:1 works → model on cuda:1."""

		def free_side_effect(device: str) -> int:
			if device == "cuda:0":
				raise RuntimeError("CUDA error on device 0")
			return 20_000_000_000

		def total_side_effect(device: str) -> int:
			if device == "cuda:0":
				raise RuntimeError("CUDA error on device 0")
			return 24_000_000_000

		mock_free.side_effect = free_side_effect
		mock_total.side_effect = total_side_effect

		model: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(
			models=[model], devices=["cuda:0", "cuda:1"]
		)
		result: tuple[ScheduledModel, str] | None = scheduler._find_best_fit()
		assert result is not None
		assert result[0] == model
		assert result[1] == "cuda:1"


class TestRunAll:
	"""Tests for ModelScheduler.run_all with mocked Popen and VRAM queries."""

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.time.sleep")
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_run_all_single_model(
		self,
		mock_popen: MagicMock,
		_mock_sleep: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Spawns one subprocess, polls completion."""
		mock_proc: MagicMock = MagicMock()
		# first poll: still running, second: done
		mock_proc.poll.side_effect = [None, 0]
		mock_popen.return_value = mock_proc

		model: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		scheduler.save_path = str(tmp_path)
		scheduler.run_all()

		assert scheduler.completed == ["pythia-14m"]
		assert scheduler.failed == []
		mock_popen.assert_called_once()

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.time.sleep")
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_run_all_handles_failure(
		self,
		mock_popen: MagicMock,
		_mock_sleep: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Subprocess exits non-zero, logged in failed."""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.side_effect = [None, 1]  # exit code 1
		mock_popen.return_value = mock_proc

		model: ScheduledModel = _make_scheduled_model("broken-model", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		scheduler.save_path = str(tmp_path)
		scheduler.run_all()

		assert scheduler.completed == []
		assert len(scheduler.failed) == 1
		assert scheduler.failed[0] == ("broken-model", 1)

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=24_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.time.sleep")
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_run_all_multiple_models_sequential(
		self,
		mock_popen: MagicMock,
		_mock_sleep: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Large models run one at a time — each nearly fills GPU."""
		# Each model est ~15GB (1.25B * 4 * 3), only one fits in 24GB at a time
		# because committed VRAM tracking prevents overscheduling
		model1: ScheduledModel = _make_scheduled_model("large-1", 1_250_000_000)
		model2: ScheduledModel = _make_scheduled_model("large-2", 1_250_000_000)

		proc1: MagicMock = MagicMock()
		proc1.poll.side_effect = [None, None, None, 0]
		proc2: MagicMock = MagicMock()
		proc2.poll.side_effect = [None, 0]
		mock_popen.side_effect = [proc1, proc2]

		scheduler: ModelScheduler = _make_scheduler(models=[model1, model2])
		scheduler.save_path = str(tmp_path)
		scheduler.run_all()

		assert len(scheduler.completed) == 2
		assert scheduler.failed == []
		assert mock_popen.call_count == 2

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=24_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.time.sleep")
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_run_all_multiple_models_concurrent(
		self,
		mock_popen: MagicMock,
		_mock_sleep: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Small models run concurrently — both fit on GPU together."""
		tiny1: ScheduledModel = _make_scheduled_model("tiny-1", 14_000_000)
		tiny2: ScheduledModel = _make_scheduled_model("tiny-2", 14_000_000)

		proc1: MagicMock = MagicMock()
		proc1.poll.side_effect = [None, None, 0]
		proc2: MagicMock = MagicMock()
		proc2.poll.side_effect = [None, 0]
		mock_popen.side_effect = [proc1, proc2]

		scheduler: ModelScheduler = _make_scheduler(models=[tiny1, tiny2])
		scheduler.save_path = str(tmp_path)
		scheduler.run_all()

		assert set(scheduler.completed) == {"tiny-1", "tiny-2"}
		assert scheduler.failed == []
		assert mock_popen.call_count == 2

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=24_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.time.sleep")
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_run_all_mixed_sizes(
		self,
		mock_popen: MagicMock,
		_mock_sleep: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Large model gets GPU alone, then small models pack together."""
		# Large: 1.9B params → est ~22.8GB. After spawn, effective = 1.2GB.
		# Small: 180M params → est ~2.16GB — doesn't fit alongside large.
		large: ScheduledModel = _make_scheduled_model("large", 1_900_000_000)
		small1: ScheduledModel = _make_scheduled_model("small-1", 180_000_000)
		small2: ScheduledModel = _make_scheduled_model("small-2", 180_000_000)

		large_proc: MagicMock = MagicMock()
		large_proc.poll.side_effect = [None, 0]
		small1_proc: MagicMock = MagicMock()
		small1_proc.poll.side_effect = [None, None, 0]
		small2_proc: MagicMock = MagicMock()
		small2_proc.poll.side_effect = [None, 0]
		mock_popen.side_effect = [large_proc, small1_proc, small2_proc]

		scheduler: ModelScheduler = _make_scheduler(models=[large, small1, small2])
		scheduler.save_path = str(tmp_path)
		scheduler.run_all()

		assert len(scheduler.completed) == 3
		assert scheduler.failed == []
		# large completes first, before smalls are scheduled
		assert scheduler.completed[0] == "large"
		assert mock_popen.call_count == 3

	def test_run_all_empty_models(self) -> None:
		"""No models → returns immediately."""
		scheduler: ModelScheduler = _make_scheduler(models=[])
		scheduler.run_all()
		assert scheduler.completed == []
		assert scheduler.failed == []

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=4_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=100_000_000,
	)
	def test_run_all_starvation_when_no_model_fits(
		self,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
	) -> None:
		"""All models too large → scheduler gives up and marks them as failed.

		When no model fits any device and nothing is running, the scheduler
		detects the starvation condition and marks remaining models as failed
		with exit code -1 rather than looping forever.
		"""
		# 100B params → ~1.2TB estimated VRAM, never fits in 4GB GPU
		huge: ScheduledModel = _make_scheduled_model("huge-model", 100_000_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[huge])
		scheduler.run_all()

		assert scheduler.completed == []
		assert len(scheduler.failed) == 1
		assert scheduler.failed[0] == ("huge-model", -1)

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.time.sleep")
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_run_all_keyboard_interrupt_cleans_up(
		self,
		mock_popen: MagicMock,
		mock_sleep: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""KeyboardInterrupt during sleep terminates running subprocesses.

		The try/finally in run_all ensures subprocesses are cleaned up
		when the scheduler is interrupted.
		"""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = None  # never finishes
		mock_popen.return_value = mock_proc
		mock_sleep.side_effect = KeyboardInterrupt()

		model: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		scheduler.save_path = str(tmp_path)

		with pytest.raises(KeyboardInterrupt):
			scheduler.run_all()

		# cleanup: process.terminate() is called on running subprocesses
		mock_proc.terminate.assert_called_once()

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=4_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=100_000_000,
	)
	def test_run_all_starvation_multiple_models(
		self,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
	) -> None:
		"""Multiple models too large → all marked failed with exit code -1."""
		huge1: ScheduledModel = _make_scheduled_model("huge-1", 100_000_000_000)
		huge2: ScheduledModel = _make_scheduled_model("huge-2", 50_000_000_000)
		huge3: ScheduledModel = _make_scheduled_model("huge-3", 25_000_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[huge1, huge2, huge3])
		scheduler.run_all()

		assert scheduler.completed == []
		assert len(scheduler.failed) == 3
		failed_names: set[str] = {name for name, _code in scheduler.failed}
		assert failed_names == {"huge-1", "huge-2", "huge-3"}
		assert all(code == -1 for _name, code in scheduler.failed)


class TestSpawnModel:
	"""Tests for _spawn_model internals."""

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_spawn_sets_env_threads(
		self,
		mock_popen: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Verifies OMP_NUM_THREADS in subprocess env."""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = None
		mock_popen.return_value = mock_proc

		model: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model], total_cpu_cores=8)
		scheduler.save_path = str(tmp_path)
		scheduler._spawn_model(model, "cuda:0")

		# check subprocess.Popen was called with env containing thread vars
		call_kwargs: dict[str, Any] = dict(mock_popen.call_args.kwargs)
		env: dict[str, str] = call_kwargs["env"]
		assert "OMP_NUM_THREADS" in env
		assert int(env["OMP_NUM_THREADS"]) >= 1

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_spawn_allocates_cores(
		self,
		mock_popen: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Verifies CPU cores allocated from pool."""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = None
		mock_popen.return_value = mock_proc

		model: ScheduledModel = _make_scheduled_model("pythia-14m", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model], total_cpu_cores=8)
		scheduler.save_path = str(tmp_path)
		initial_cores: int = scheduler.core_pool.n_available
		scheduler._spawn_model(model, "cuda:0")
		# cores should have been allocated
		assert scheduler.core_pool.n_available < initial_cores

	@patch(
		"attention_motifs.pipeline.model_scheduler.get_total_vram",
		return_value=24_000_000_000,
	)
	@patch(
		"attention_motifs.pipeline.model_scheduler.get_free_vram",
		return_value=20_000_000_000,
	)
	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_spawn_updates_committed_vram(
		self,
		mock_popen: MagicMock,
		_mock_free: MagicMock,
		_mock_total: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Verifies _device_committed is updated on spawn."""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = None
		mock_popen.return_value = mock_proc

		model: ScheduledModel = _make_scheduled_model("gpt2-small", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		scheduler.save_path = str(tmp_path)

		assert scheduler._device_committed["cuda:0"] == 0
		scheduler._spawn_model(model, "cuda:0")
		assert scheduler._device_committed["cuda:0"] == model.estimated_vram

	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_spawn_creates_log_file(
		self,
		mock_popen: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Log file created at expected path."""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = None
		mock_popen.return_value = mock_proc

		model: ScheduledModel = _make_scheduled_model("gpt2-small", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		scheduler.save_path = str(tmp_path)
		scheduler._spawn_model(model, "cuda:0")

		log_path: Path = tmp_path / "gpt2-small_parallel.log"
		assert log_path.exists()

	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_spawn_model_slash_in_name(
		self,
		mock_popen: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Model name with '/' → log file uses '_' instead (no nested dirs)."""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = None
		mock_popen.return_value = mock_proc

		model: ScheduledModel = _make_scheduled_model(
			"meta-llama/Llama-3.2-1B", 1_000_000_000
		)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		scheduler.save_path = str(tmp_path)
		scheduler._spawn_model(model, "cuda:0")

		expected_log: Path = tmp_path / "meta-llama_Llama-3.2-1B_parallel.log"
		assert expected_log.exists()
		# no nested directory created
		assert not (tmp_path / "meta-llama").exists()

	def test_spawn_log_path_missing_parent(self, tmp_path: Path) -> None:
		"""save_path parent dir doesn't exist → FileNotFoundError."""
		model: ScheduledModel = _make_scheduled_model("gpt2-small", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model])
		scheduler.save_path = str(tmp_path / "nonexistent" / "nested")

		with pytest.raises(FileNotFoundError):
			scheduler._spawn_model(model, "cuda:0")

	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_spawn_fair_share_cores(
		self,
		mock_popen: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Core allocation uses fair share: total_cores // n_models_remaining."""
		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = None
		mock_popen.return_value = mock_proc

		m1: ScheduledModel = _make_scheduled_model("m1", 14_000_000)
		m2: ScheduledModel = _make_scheduled_model("m2", 14_000_000)
		m3: ScheduledModel = _make_scheduled_model("m3", 14_000_000)
		scheduler: ModelScheduler = _make_scheduler(
			models=[m1, m2, m3], total_cpu_cores=12
		)
		scheduler.save_path = str(tmp_path)

		# 3 models pending, 0 running → 12 // 3 = 4 cores for first spawn
		scheduler._spawn_model(m1, "cuda:0")
		assert scheduler.core_pool.n_available == 8  # 12 - 4

	@patch("attention_motifs.pipeline.model_scheduler.subprocess.Popen")
	def test_spawn_popen_oserror_releases_cores(
		self,
		mock_popen: MagicMock,
		tmp_path: Path,
	) -> None:
		"""Popen raises OSError → cores returned to pool, error re-raised."""
		mock_popen.side_effect = OSError("No such file or directory")

		model: ScheduledModel = _make_scheduled_model("gpt2-small", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[model], total_cpu_cores=8)
		scheduler.save_path = str(tmp_path)
		initial_cores: int = scheduler.core_pool.n_available

		with pytest.raises(OSError, match="No such file"):
			scheduler._spawn_model(model, "cuda:0")

		# cores should be returned after the OSError
		assert scheduler.core_pool.n_available == initial_cores


# ===========================================================================
# _poll_running tests
# ===========================================================================


class TestPollRunning:
	"""Tests for ModelScheduler._poll_running resource cleanup."""

	def test_poll_running_releases_cores_on_failure(self) -> None:
		"""Non-zero exit → cores released, committed VRAM decreased, model in failed."""
		model: ScheduledModel = _make_scheduled_model("broken-model", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[])

		# simulate prior allocation: remove cores 0-2 from pool
		scheduler.core_pool.available = [3, 4, 5, 6, 7]
		scheduler._device_committed["cuda:0"] = model.estimated_vram

		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = 1  # non-zero exit
		mock_log: MagicMock = MagicMock()

		rm: RunningModel = RunningModel(
			model=model,
			process=mock_proc,
			device="cuda:0",
			cpu_cores=[0, 1, 2],
			log_file=mock_log,
		)
		scheduler.running.append(rm)

		scheduler._poll_running()

		# cores returned to pool
		assert scheduler.core_pool.n_available == 8
		assert 0 in scheduler.core_pool.available
		# committed VRAM released
		assert scheduler._device_committed["cuda:0"] == 0
		# model recorded as failed
		assert scheduler.failed == [("broken-model", 1)]
		assert scheduler.running == []
		# log file closed
		mock_log.close.assert_called_once()

	def test_poll_running_releases_cores_on_success(self) -> None:
		"""Exit code 0 → cores released, VRAM freed, model in completed."""
		model: ScheduledModel = _make_scheduled_model("good-model", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[])

		scheduler.core_pool.available = [3, 4, 5, 6, 7]
		scheduler._device_committed["cuda:0"] = model.estimated_vram

		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = 0
		mock_log: MagicMock = MagicMock()

		rm: RunningModel = RunningModel(
			model=model,
			process=mock_proc,
			device="cuda:0",
			cpu_cores=[0, 1, 2],
			log_file=mock_log,
		)
		scheduler.running.append(rm)

		scheduler._poll_running()

		assert scheduler.core_pool.n_available == 8
		assert 0 in scheduler.core_pool.available
		assert scheduler._device_committed["cuda:0"] == 0
		assert scheduler.completed == ["good-model"]
		assert scheduler.failed == []
		assert scheduler.running == []
		mock_log.close.assert_called_once()

	def test_poll_running_failure_logs_path(self) -> None:
		"""Failure message includes the log file path."""
		model: ScheduledModel = _make_scheduled_model("broken-model", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[])
		scheduler._device_committed["cuda:0"] = model.estimated_vram

		mock_proc: MagicMock = MagicMock()
		mock_proc.poll.return_value = 1
		mock_log: MagicMock = MagicMock()
		mock_log.name = "/tmp/broken-model_parallel.log"

		rm: RunningModel = RunningModel(
			model=model,
			process=mock_proc,
			device="cuda:0",
			cpu_cores=[],
			log_file=mock_log,
		)
		scheduler.running.append(rm)

		log_messages: list[str] = []
		scheduler._log = lambda msg: log_messages.append(msg)  # type: ignore[assignment]

		scheduler._poll_running()

		assert any("/tmp/broken-model_parallel.log" in msg for msg in log_messages)

	def test_poll_running_keeps_still_running(self) -> None:
		"""Mixed: one process done, one still running → running list filtered correctly."""
		done_model: ScheduledModel = _make_scheduled_model("done-model", 85_000_000)
		busy_model: ScheduledModel = _make_scheduled_model("busy-model", 85_000_000)
		scheduler: ModelScheduler = _make_scheduler(models=[])

		scheduler.core_pool.available = [4, 5, 6, 7]
		scheduler._device_committed["cuda:0"] = (
			done_model.estimated_vram + busy_model.estimated_vram
		)

		busy_proc: MagicMock = MagicMock()
		busy_proc.poll.return_value = None  # still running
		busy_log: MagicMock = MagicMock()

		done_proc: MagicMock = MagicMock()
		done_proc.poll.return_value = 0
		done_log: MagicMock = MagicMock()

		scheduler.running = [
			RunningModel(
				model=busy_model,
				process=busy_proc,
				device="cuda:0",
				cpu_cores=[0, 1],
				log_file=busy_log,
			),
			RunningModel(
				model=done_model,
				process=done_proc,
				device="cuda:0",
				cpu_cores=[2, 3],
				log_file=done_log,
			),
		]

		scheduler._poll_running()

		assert len(scheduler.running) == 1
		assert scheduler.running[0].model.name == "busy-model"
		assert scheduler.completed == ["done-model"]
		# done log closed, busy log NOT closed
		done_log.close.assert_called_once()
		busy_log.close.assert_not_called()


# ===========================================================================
# _cleanup_running tests
# ===========================================================================


class TestCleanupRunning:
	"""Tests for ModelScheduler._cleanup_running."""

	def test_cleanup_running_terminates_and_closes(self) -> None:
		"""All running processes terminated and log files closed."""
		scheduler: ModelScheduler = _make_scheduler(models=[])

		proc1: MagicMock = MagicMock()
		log1: MagicMock = MagicMock()
		proc2: MagicMock = MagicMock()
		log2: MagicMock = MagicMock()

		m1: ScheduledModel = _make_scheduled_model("m1", 14_000_000)
		m2: ScheduledModel = _make_scheduled_model("m2", 14_000_000)
		scheduler.running = [
			RunningModel(model=m1, process=proc1, device="cuda:0", cpu_cores=[0], log_file=log1),
			RunningModel(model=m2, process=proc2, device="cuda:0", cpu_cores=[1], log_file=log2),
		]

		scheduler._cleanup_running()

		proc1.terminate.assert_called_once()
		proc2.terminate.assert_called_once()
		log1.close.assert_called_once()
		log2.close.assert_called_once()
		assert scheduler.running == []

	def test_cleanup_running_oserror_continues(self) -> None:
		"""OSError on first terminate doesn't stop cleanup of second process."""
		scheduler: ModelScheduler = _make_scheduler(models=[])

		proc1: MagicMock = MagicMock()
		proc1.terminate.side_effect = OSError("process already dead")
		log1: MagicMock = MagicMock()
		proc2: MagicMock = MagicMock()
		log2: MagicMock = MagicMock()

		m1: ScheduledModel = _make_scheduled_model("m1", 14_000_000)
		m2: ScheduledModel = _make_scheduled_model("m2", 14_000_000)
		scheduler.running = [
			RunningModel(model=m1, process=proc1, device="cuda:0", cpu_cores=[0], log_file=log1),
			RunningModel(model=m2, process=proc2, device="cuda:0", cpu_cores=[1], log_file=log2),
		]

		scheduler._cleanup_running()  # should not raise

		proc2.terminate.assert_called_once()
		log1.close.assert_called_once()
		log2.close.assert_called_once()
		assert scheduler.running == []


# ===========================================================================
# cfg.py tests (parallel scheduling fields)
# ===========================================================================


class TestConfigParallelFields:
	def test_config_loads_parallel_fields(self, tmp_path: Path) -> None:
		"""TOML with new fields parsed correctly."""
		toml_content: str = """\
prompts_file = "data/text/pile_demo.jsonl"
patterns_dir = "data/patterns"
features_dir = "data/features"
prompts_n_samples = 10
prompts_min_chars = 10
prompts_max_chars = 100
models = ["gpt2-small"]
n_proc = 4
device = "cpu"
parallel_models = true
devices = ["cuda:0", "cuda:1"]
vram_safety_factor = 2.5
"""
		cfg_path: Path = tmp_path / "test.toml"
		cfg_path.write_text(toml_content)
		cfg: PipelineConfig = PipelineConfig.read(cfg_path)

		assert cfg.parallel_models is True
		assert cfg.devices == ["cuda:0", "cuda:1"]
		assert cfg.vram_safety_factor == 2.5
		assert cfg.batch_size == 32  # default when not specified

	def test_config_defaults_without_parallel_fields(self, tmp_path: Path) -> None:
		"""Missing parallel fields → sensible defaults."""
		toml_content: str = """\
prompts_file = "data/text/pile_demo.jsonl"
patterns_dir = "data/patterns"
features_dir = "data/features"
prompts_n_samples = 10
prompts_min_chars = 10
prompts_max_chars = 100
models = ["gpt2-small"]
n_proc = 4
device = "cpu"
"""
		cfg_path: Path = tmp_path / "test.toml"
		cfg_path.write_text(toml_content)
		cfg: PipelineConfig = PipelineConfig.read(cfg_path)

		assert cfg.parallel_models is False
		assert cfg.vram_safety_factor == 3.0
		assert cfg.batch_size == 32  # default

	def test_config_devices_fallback_to_device(self, tmp_path: Path) -> None:
		"""No `devices` key → [device] used."""
		toml_content: str = """\
prompts_file = "data/text/pile_demo.jsonl"
patterns_dir = "data/patterns"
features_dir = "data/features"
prompts_n_samples = 10
prompts_min_chars = 10
prompts_max_chars = 100
models = ["gpt2-small"]
n_proc = 4
device = "cuda:1"
"""
		cfg_path: Path = tmp_path / "test.toml"
		cfg_path.write_text(toml_content)
		cfg: PipelineConfig = PipelineConfig.read(cfg_path)

		# should fall back to [device]
		assert cfg.devices == ["cuda:1"]

	def test_config_cli_parallel_models(self) -> None:
		"""--parallel-models flag sets field."""
		cfg: PipelineConfig = PipelineConfig.from_cli(
			["tests/pipeline_cfg_test.toml", "--parallel-models"]
		)
		assert cfg.parallel_models is True

	def test_config_cli_devices(self) -> None:
		"""--devices cuda:0,cuda:1 parsed to list."""
		cfg: PipelineConfig = PipelineConfig.from_cli(
			["tests/pipeline_cfg_test.toml", "--devices", "cuda:0,cuda:1"]
		)
		assert cfg.devices == ["cuda:0", "cuda:1"]

	def test_config_cli_vram_safety_factor(self) -> None:
		"""--vram-safety-factor 2.5 parsed correctly."""
		cfg: PipelineConfig = PipelineConfig.from_cli(
			["tests/pipeline_cfg_test.toml", "--vram-safety-factor", "2.5"]
		)
		assert cfg.vram_safety_factor == 2.5

	def test_config_loads_batch_size(self, tmp_path: Path) -> None:
		"""TOML with explicit batch_size → parsed correctly."""
		toml_content: str = """\
prompts_file = "data/text/pile_demo.jsonl"
patterns_dir = "data/patterns"
features_dir = "data/features"
prompts_n_samples = 10
prompts_min_chars = 10
prompts_max_chars = 100
models = ["gpt2-small"]
n_proc = 4
device = "cpu"
batch_size = 64
"""
		cfg_path: Path = tmp_path / "test.toml"
		cfg_path.write_text(toml_content)
		cfg: PipelineConfig = PipelineConfig.read(cfg_path)
		assert cfg.batch_size == 64

	def test_config_cli_batch_size(self) -> None:
		"""--batch-size 16 parsed correctly."""
		cfg: PipelineConfig = PipelineConfig.from_cli(
			["tests/pipeline_cfg_test.toml", "--batch-size", "16"]
		)
		assert cfg.batch_size == 16

	def test_config_validates_devices_nonempty(self, tmp_path: Path) -> None:
		"""Empty devices list fails validation."""
		toml_content: str = """\
prompts_file = "data/text/pile_demo.jsonl"
patterns_dir = "data/patterns"
features_dir = "data/features"
prompts_n_samples = 10
prompts_min_chars = 10
prompts_max_chars = 100
models = ["gpt2-small"]
n_proc = 4
device = "cpu"
devices = []
"""
		cfg_path: Path = tmp_path / "test.toml"
		cfg_path.write_text(toml_content)
		with pytest.raises(AssertionError, match="devices"):
			PipelineConfig.read(cfg_path)

	def test_config_validates_vram_safety_positive(self, tmp_path: Path) -> None:
		"""Negative safety factor fails validation."""
		toml_content: str = """\
prompts_file = "data/text/pile_demo.jsonl"
patterns_dir = "data/patterns"
features_dir = "data/features"
prompts_n_samples = 10
prompts_min_chars = 10
prompts_max_chars = 100
models = ["gpt2-small"]
n_proc = 4
device = "cpu"
vram_safety_factor = -1.0
"""
		cfg_path: Path = tmp_path / "test.toml"
		cfg_path.write_text(toml_content)
		with pytest.raises(AssertionError, match="vram_safety_factor"):
			PipelineConfig.read(cfg_path)


# ===========================================================================
# s1_activations.py tests (branching logic)
# ===========================================================================


class TestGenerateActivationsBranching:
	"""Tests that generate_activations dispatches to the correct path.

	We mock ``pattern_lens.activations`` in sys.modules before importing
	s1_activations, because its module-level import can fail under
	beartype + jaxtyping in some environments.
	"""

	@patch.dict(
		"sys.modules",
		{"pattern_lens.activations": MagicMock()},
	)
	def test_generate_activations_sequential_default(self) -> None:
		"""parallel_models=False calls sequential."""
		# force re-import with mocked pattern_lens
		import importlib
		import attention_motifs.pipeline.s1_activations as s1_mod

		importlib.reload(s1_mod)

		cfg: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		cfg.parallel_models = False

		with (
			patch.object(s1_mod, "_generate_activations_sequential") as mock_seq,
			patch.object(s1_mod, "_generate_activations_parallel") as mock_par,
		):
			s1_mod.generate_activations(cfg)
			mock_seq.assert_called_once_with(cfg)
			mock_par.assert_not_called()

	@patch.dict(
		"sys.modules",
		{"pattern_lens.activations": MagicMock()},
	)
	def test_generate_activations_parallel_when_enabled(self) -> None:
		"""parallel_models=True calls parallel."""
		import importlib
		import attention_motifs.pipeline.s1_activations as s1_mod

		importlib.reload(s1_mod)

		cfg: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		cfg.parallel_models = True

		with (
			patch.object(s1_mod, "_generate_activations_sequential") as mock_seq,
			patch.object(s1_mod, "_generate_activations_parallel") as mock_par,
		):
			s1_mod.generate_activations(cfg)
			mock_par.assert_called_once_with(cfg)
			mock_seq.assert_not_called()


class TestGenerateActivationsParallel:
	"""Integration test for the full _generate_activations_parallel flow."""

	@patch.dict(
		"sys.modules",
		{"pattern_lens.activations": MagicMock()},
	)
	def test_generate_activations_parallel_full_flow(self) -> None:
		"""Known model → scheduler, unknown model → sequential fallback."""
		import importlib
		import attention_motifs.pipeline.s1_activations as s1_mod

		importlib.reload(s1_mod)

		cfg: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		cfg.parallel_models = True
		cfg.models = ["gpt2-small", "unknown-model"]
		cfg.devices = ["cuda:0"]
		cfg.vram_safety_factor = 3.0
		cfg.batch_size = 16

		mock_table: dict[str, ModelInfo] = {
			"gpt2-small": ModelInfo(name="gpt2-small", n_params=85_000_000),
		}
		mock_scheduler_instance: MagicMock = MagicMock()

		with (
			patch(
				"attention_motifs.pipeline.model_table.fetch_model_table",
				return_value=mock_table,
			) as mock_fetch,
			patch(
				"attention_motifs.pipeline.model_scheduler.ModelScheduler",
				return_value=mock_scheduler_instance,
			) as mock_scheduler_cls,
			patch.object(s1_mod, "activations_main") as mock_act_main,
		):
			s1_mod._generate_activations_parallel(cfg)

			# model table fetched
			mock_fetch.assert_called_once()

			# scheduler created with only the known model
			mock_scheduler_cls.assert_called_once()
			call_kwargs: dict[str, Any] = dict(mock_scheduler_cls.call_args.kwargs)
			scheduled_models: list[ScheduledModel] = call_kwargs["models"]
			assert len(scheduled_models) == 1
			assert scheduled_models[0].name == "gpt2-small"
			assert call_kwargs["batch_size"] == 16

			# run_all called on the scheduler
			mock_scheduler_instance.run_all.assert_called_once()

			# unknown model falls back to sequential with batch_size
			mock_act_main.assert_called_once()
			act_call_kwargs: dict[str, Any] = dict(mock_act_main.call_args.kwargs)
			assert act_call_kwargs["model_name"] == "unknown-model"
			assert act_call_kwargs["batch_size"] == 16

	@patch.dict(
		"sys.modules",
		{"pattern_lens.activations": MagicMock()},
	)
	def test_parallel_all_models_unknown(self) -> None:
		"""All models unknown → scheduler never created, all run sequentially."""
		import importlib
		import attention_motifs.pipeline.s1_activations as s1_mod

		importlib.reload(s1_mod)

		cfg: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		cfg.parallel_models = True
		cfg.models = ["unknown-1", "unknown-2"]
		cfg.devices = ["cuda:0"]
		cfg.vram_safety_factor = 3.0

		# empty table → every model raises KeyError
		mock_table: dict[str, ModelInfo] = {}

		with (
			patch(
				"attention_motifs.pipeline.model_table.fetch_model_table",
				return_value=mock_table,
			),
			patch(
				"attention_motifs.pipeline.model_scheduler.ModelScheduler",
			) as mock_scheduler_cls,
			patch.object(s1_mod, "activations_main") as mock_act_main,
		):
			s1_mod._generate_activations_parallel(cfg)

			# scheduler never created
			mock_scheduler_cls.assert_not_called()

			# both models run via sequential fallback
			assert mock_act_main.call_count == 2
			call_models: list[str] = [
				call.kwargs["model_name"] for call in mock_act_main.call_args_list
			]
			assert call_models == ["unknown-1", "unknown-2"]
