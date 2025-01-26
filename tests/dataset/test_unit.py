from pathlib import Path

import torch
import pytest
import json

from attention_motifs.dataset.dataset import (
	APGenerationConfig,
	AttentionPatternDataset,
	AttentionPatternMetadata,
	CollectedAttentionPatternDataloader,
)
from attention_motifs.dataset.prompts import PromptDatasetConfig

TEMP_DIR: Path = Path("tests/_temp")


@pytest.fixture(scope="session", autouse=True)
def setup_temp_dir():
	"""Fixture to ensure the tests/_temp directory exists before tests run."""
	TEMP_DIR.mkdir(exist_ok=True, parents=True)
	yield
	# We do not remove it; you might want to do so in a real test environment.


def test_attention_pattern_dataset_basic():
	"""Unit test for the basic indexing and length of AttentionPatternDataset."""
	dummy_patterns = torch.zeros((5, 3, 3))  # n_patterns=5, n_ctx=3
	dummy_metadata = [
		AttentionPatternMetadata(
			model_name="dummy-model",
			idx_layer=0,
			idx_head=i,
			prompt_hash=f"hash_{i}",
			n_ctx=3,
		)
		for i in range(5)
	]
	ds = AttentionPatternDataset(
		n_ctx=3, n_patterns=5, patterns=dummy_patterns, metadata=dummy_metadata
	)

	assert len(ds) == 5
	pattern, meta = ds[2]
	assert pattern.shape == (3, 3)
	assert meta.idx_head == 2
	assert meta.n_ctx == 3


def test_dataloader_properties():
	"""Unit test for dataloader properties."""
	ds_patterns = torch.randn((4, 3, 3))
	ds_meta = [
		AttentionPatternMetadata("modelA", 0, i, f"hash{i}", 3) for i in range(4)
	]
	ds = AttentionPatternDataset(
		n_ctx=3,
		n_patterns=4,
		patterns=ds_patterns,
		metadata=ds_meta,
	)
	dummy_config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(source_path=Path("fake.jsonl")),
		model_names=["modelA", "modelB"],
		prompt_token_len_tolerance=2,
	)
	loader = CollectedAttentionPatternDataloader(
		config=dummy_config,
		prompts=[{"text": "A", "hash": "hash0"}],
		datasets=[ds],
		batch_size=2,
	)

	assert loader.model_names == ["modelA", "modelB"]
	assert loader.n_datasets == 1
	assert loader.n_total_samples == 4
	assert loader.n_ctx_counts[3] == 4
	meta_info = loader.dataset_metadata[0]
	assert meta_info["n_ctx"] == 3
	assert meta_info["n_patterns"] == 4


# ----------------------------------------------
# APGenerationConfig tests
# ----------------------------------------------


def test_config_load_empty_file():
	"""Test APGenerationConfig on an empty file -> should return []."""
	empty_file = TEMP_DIR / "empty_prompts.jsonl"
	empty_file.write_text("")  # no lines
	cfg = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(source_path=empty_file),
		model_names=[],
	)
	data = cfg.load_text_data()
	assert data == []


def test_config_load_all_filtered():
	"""If min_length is large, all lines get filtered."""
	big_file = TEMP_DIR / "big_min_length.jsonl"
	# single line with short text
	big_file.write_text(json.dumps({"text": "Short prompt"}) + "\n")
	cfg = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(source_path=big_file),
		model_names=[],
	)
	data = cfg.load_text_data()
	assert data == []


def test_config_splitting_behavior():
	"""Check that max_length causes splitting and that min_length re-filters."""
	splitted_file = TEMP_DIR / "splitted.jsonl"
	# single line with text length=50
	text_50 = "x" * 50
	splitted_file.write_text(json.dumps({"text": text_50}) + "\n")

	cfg = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(source_path=splitted_file),
		model_names=[],
	)
	data = cfg.load_text_data()
	# original is length=50
	# splitted into segments of length=20,20,10
	# after split, each is >= min_length=10, so we keep all 3
	assert len(data) == 3
	lengths = [len(d["text"]) for d in data]
	assert lengths == [20, 20, 10]


def test_attention_pattern_dataset_zero_length():
	"""Corner case: zero-length dataset is possible, though unusual."""
	ds = AttentionPatternDataset(
		n_ctx=3, n_patterns=0, patterns=torch.empty((0, 3, 3)), metadata=[]
	)
	assert len(ds) == 0
	with pytest.raises(IndexError):
		_ = ds[0]  # should raise an error


# ----------------------------------------------
# CollectedAttentionPatternDataloader tests
# ----------------------------------------------


def test_dataloader_negative_batch_size():
	"""Dataloader should raise ValueError if batch_size < 1."""
	dummy_config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(source_path=Path("fake.jsonl")),
		model_names=[],
	)
	ds = AttentionPatternDataset(
		n_ctx=3, n_patterns=0, patterns=torch.empty((0, 3, 3)), metadata=[]
	)
	with pytest.raises(ValueError):
		_ = CollectedAttentionPatternDataloader(
			config=dummy_config, prompts=[], datasets=[ds],
		)


def test_dataloader_iteration():
	"""Basic iteration test. Combine multiple datasets of different sizes."""
	ds1_patterns = torch.rand((2, 4, 4))
	ds1_meta = [
		AttentionPatternMetadata("modelA", 0, i, f"hash{i}", 4) for i in range(2)
	]
	ds1 = AttentionPatternDataset(
		n_ctx=4, n_patterns=2, patterns=ds1_patterns, metadata=ds1_meta
	)

	ds2_patterns = torch.rand((3, 4, 4))
	ds2_meta = [
		AttentionPatternMetadata("modelA", 1, i, f"hash{i+2}", 4) for i in range(3)
	]
	ds2 = AttentionPatternDataset(
		n_ctx=4, n_patterns=3, patterns=ds2_patterns, metadata=ds2_meta
	)

	dummy_config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(source_path=Path("fake.jsonl")),
		model_names=["modelA"],
	)

	loader = CollectedAttentionPatternDataloader(
		config=dummy_config, prompts=[], datasets=[ds1, ds2],
	)

	# total of 5 items => batch_size=2 => iteration yields 3 times
	all_yields = list(loader.batches(batch_size=2))
	assert len(all_yields) == 3

	# first two yields => batch_size=2
	for batch_patterns, batch_meta in all_yields[:2]:
		assert batch_patterns.shape == (2, 4, 4)
		assert len(batch_meta) == 2

	# last yield => leftover 1
	last_patterns, last_meta = all_yields[-1]
	assert last_patterns.shape == (1, 4, 4)
	assert len(last_meta) == 1


def test_dataloader_empty_datasets():
	"""If we pass an empty dataset list, iteration yields nothing."""
	dummy_config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(source_path=Path("fake.jsonl")),
		model_names=[],
	)
	loader = CollectedAttentionPatternDataloader(
		config=dummy_config, prompts=[], datasets=[],
	)
	all_batches = list(loader.batches(batch_size=2))
	assert all_batches == []
