"""Tests for pipeline smart mode state tracking."""

import json
import tempfile
from pathlib import Path

from attention_motifs.pipeline.state import PipelineState, StepRecord, StepName
from attention_motifs.pipeline.cfg import PipelineConfig


def test_step_record_serialize_load() -> None:
	"""Test StepRecord serialization round-trip."""
	record: StepRecord = StepRecord(
		name="s1_activations",
		completed_at="2024-01-15T10:30:00",
		duration_seconds=125.45,
	)

	serialized: dict = record.serialize()
	assert serialized["name"] == "s1_activations"
	assert serialized["completed_at"] == "2024-01-15T10:30:00"
	assert serialized["duration_seconds"] == 125.45

	loaded: StepRecord = StepRecord.load(serialized)
	assert loaded.name == record.name
	assert loaded.completed_at == record.completed_at
	assert loaded.duration_seconds == record.duration_seconds


def test_pipeline_state_serialize_load() -> None:
	"""Test PipelineState serialization round-trip."""
	state: PipelineState = PipelineState(config_hash="abc123")
	state.mark_step_complete("s1_activations", 10.5)
	state.mark_step_complete("s1b_render_patterns", 5.2)

	serialized: dict = state.serialize()
	assert serialized["config_hash"] == "abc123"
	assert "s1_activations" in serialized["completed_steps"]
	assert "s1b_render_patterns" in serialized["completed_steps"]
	assert serialized["last_run"] is not None

	loaded: PipelineState = PipelineState.load(serialized)
	assert loaded.config_hash == state.config_hash
	assert loaded.is_step_complete("s1_activations")
	assert loaded.is_step_complete("s1b_render_patterns")
	assert not loaded.is_step_complete("s2_features")


def test_pipeline_state_save_read() -> None:
	"""Test PipelineState file save/read cycle."""
	state: PipelineState = PipelineState(config_hash="test_hash_123")
	state.mark_step_complete("s1_activations", 100.0)
	state.mark_step_complete("s2_features", 200.0)

	with tempfile.TemporaryDirectory() as tmpdir:
		state_path: Path = Path(tmpdir) / "pipeline_state.json"

		state.save(state_path)
		assert state_path.exists()

		# Verify JSON content
		with open(state_path) as f:
			data: dict = json.load(f)
		assert data["config_hash"] == "test_hash_123"
		assert len(data["completed_steps"]) == 2

		# Load and verify
		loaded: PipelineState = PipelineState.read(state_path)
		assert loaded.config_hash == "test_hash_123"
		assert loaded.is_step_complete("s1_activations")
		assert loaded.is_step_complete("s2_features")
		assert loaded.completed_steps["s1_activations"].duration_seconds == 100.0


def test_pipeline_state_is_step_complete() -> None:
	"""Test step completion checking."""
	state: PipelineState = PipelineState(config_hash="hash")

	assert not state.is_step_complete("s1_activations")
	state.mark_step_complete("s1_activations", 1.0)
	assert state.is_step_complete("s1_activations")
	assert not state.is_step_complete("s2_features")


def test_pipeline_state_invalidate_all() -> None:
	"""Test invalidating all completed steps."""
	state: PipelineState = PipelineState(config_hash="hash")
	state.mark_step_complete("s1_activations", 1.0)
	state.mark_step_complete("s2_features", 2.0)

	assert len(state.completed_steps) == 2
	state.invalidate_all()
	assert len(state.completed_steps) == 0
	assert not state.is_step_complete("s1_activations")


def test_pipeline_state_read_corrupted_file() -> None:
	"""Test reading a corrupted state file returns fresh state."""
	with tempfile.TemporaryDirectory() as tmpdir:
		state_path: Path = Path(tmpdir) / "pipeline_state.json"

		# Write invalid JSON
		with open(state_path, "w") as f:
			f.write("not valid json {{{")

		# Should not raise, returns fresh state
		import warnings

		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")
			loaded: PipelineState = PipelineState.read(state_path)
			assert len(w) == 1
			assert "Failed to load pipeline state" in str(w[0].message)

		assert loaded.config_hash == ""
		assert len(loaded.completed_steps) == 0


def test_config_compute_hash_deterministic() -> None:
	"""Test that config hash is deterministic."""
	cfg: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))

	hash1: str = cfg.compute_hash()
	hash2: str = cfg.compute_hash()

	assert hash1 == hash2
	assert len(hash1) == 64  # SHA-256 hex digest


def test_config_compute_hash_excludes_runtime_settings() -> None:
	"""Test that runtime settings don't affect the hash."""
	cfg1: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
	cfg2: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))

	# Change runtime settings that shouldn't affect output
	cfg2.n_proc = 999
	cfg2.device = "cpu"
	cfg2.verbose = 0
	cfg2.force_overwrite = True
	cfg2.smart_mode = True

	assert cfg1.compute_hash() == cfg2.compute_hash()


def test_config_compute_hash_changes_with_content_settings() -> None:
	"""Test that content-affecting settings change the hash."""
	cfg1: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
	cfg2: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))

	# Change a setting that affects output content
	cfg2.prompts_n_samples = 9999

	assert cfg1.compute_hash() != cfg2.compute_hash()


def test_config_smart_mode_cli_flag() -> None:
	"""Test that --smart flag sets smart_mode."""
	cfg_without: PipelineConfig = PipelineConfig.from_cli(
		["tests/pipeline_cfg_test.toml"]
	)
	assert cfg_without.smart_mode is False

	cfg_with: PipelineConfig = PipelineConfig.from_cli(
		["tests/pipeline_cfg_test.toml", "--smart"]
	)
	assert cfg_with.smart_mode is True
