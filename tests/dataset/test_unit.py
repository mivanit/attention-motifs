from pathlib import Path

import torch
import pytest

from attention_motifs.dataset import (
	APGenerationConfig,
	AttentionPatternDataset,
	AttentionPatternMetadata,
	CollectedAttentionPatternDataloader,
)

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


def test_dataloader_iteration():
	"""Unit test for iteration in CollectedAttentionPatternDataloader."""
	# We'll create two small datasets to test the iteration
	# Each dataset has length 2 or 3, to see if iteration over them is combined.
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
		prompts_path=Path("fake.jsonl"),
		model_names=["modelA"],
		min_length=5,
		max_length=100,
		prompt_token_len_tolerance=2,
	)

	loader = CollectedAttentionPatternDataloader(
		config=dummy_config, prompts=[], datasets=[ds1, ds2], batch_size=2
	)

	# flatten => total 5 items => with batch_size=2 => 3 yields
	# yields: (2 items), (2 items), (1 item)
	all_yields = list(loader)
	assert len(all_yields) == 3

	# first two yields have batch_size 2
	for batch_patterns, batch_meta in all_yields[:2]:
		assert batch_patterns.shape == (2, 4, 4)
		assert len(batch_meta) == 2

	# last yield has 1 leftover
	last_patterns, last_meta = all_yields[-1]
	assert last_patterns.shape == (1, 4, 4)
	assert len(last_meta) == 1


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
		prompts_path=Path("fake.jsonl"),
		model_names=["modelA", "modelB"],
		min_length=5,
		max_length=100,
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
