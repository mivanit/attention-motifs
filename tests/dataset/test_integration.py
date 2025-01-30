# file: test_integration.py

from pathlib import Path
import json

import pytest

from attention_motifs.dataset.dataset import (
	APGenerationConfig,
	CollectedAttentionPatternDataloader,
)
from attention_motifs.dataset.prompts import PromptDatasetConfig

TEMP_DIR: Path = Path("tests/_temp")


def make_sample_prompts_file() -> Path:
	"""Create a small prompts file inside tests/_temp with multiple lines."""
	# We'll not use a tempfile here so we can see the file in tests/_temp
	pfile = TEMP_DIR / "sample_prompts.jsonl"
	pfile.parent.mkdir(exist_ok=True, parents=True)
	# create a few sample prompts
	prompts = [
		{"text": "Hello world!"},
		{"text": "This is a test exceeding min length."},
		{"text": "This is another test exceeding min length."},
		{"text": "Short"},
	]
	with open(pfile, "w") as f:
		for p in prompts:
			f.write(json.dumps(p) + "\n")
	return pfile


SAMPLE_PROMPTS_FILE: Path = make_sample_prompts_file()


@pytest.mark.parametrize(
	"model_name", ["tiny-stories-1M"]
)  # Expand or change as desired
def test_integration_generate_save_read(model_name: str):
	# We'll use an actual model name "gpt2" by default for the test.

	config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(
			source_path=SAMPLE_PROMPTS_FILE,
			char_len_min=1,
		),
		token_len_min=1,
		model_names=[model_name],
		prompt_token_len_tolerance=2,
	)

	dl = CollectedAttentionPatternDataloader.generate(config=config)

	# we expect at least one dataset
	assert dl.n_datasets >= 1
	# short prompt "Short" is filtered out
	# so we have at least 2 prompts used

	# step 1: test iteration
	all_batches = list(dl.dataloader(batch_size=1, shuffle=False))
	# we can't assert an exact # because it depends on the model's # of layers * heads
	# but we expect something > 0
	assert len(all_batches) > 0

	# step 2: test save -> read

	tmp_path = Path(TEMP_DIR / "test_integration_generate_save_read")
	dl.save(tmp_path)

	dl2 = CollectedAttentionPatternDataloader.read(tmp_path)
	assert dl2.n_datasets == dl.n_datasets
	# the set of prompts should match
	assert len(dl2.prompts) == len(dl.prompts)

	# quick iteration check
	all_batches2 = list(dl2.dataloader(batch_size=1, shuffle=False))
	assert len(all_batches2) == len(all_batches)

	# check the patterns shape is the same
	for (pats1, meta1), (pats2, meta2) in zip(all_batches, all_batches2):
		assert pats1.shape == pats2.shape
		# metadata might differ in object identity, but check
		for m1, m2 in zip(meta1, meta2):
			assert m1.prompt_hash == m2.prompt_hash
			assert m1.model_name == m2.model_name
			assert m1.n_ctx == m2.n_ctx


@pytest.mark.parametrize(
	"model_names",
	[
		[],  # No models
		["tiny-stories-1M", "pythia-14m"],
	],
)
def test_integration_multiple_models(model_names: list[str]):
	"""Check the behavior with zero or multiple model names."""
	config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(
			source_path=SAMPLE_PROMPTS_FILE,
			char_len_min=1,
		),
		token_len_min=1,
		model_names=model_names,
		prompt_token_len_tolerance=2,
	)
	dl = CollectedAttentionPatternDataloader.generate(config)
	# If no models => no datasets
	# If multiple models => multiple sets
	if not model_names:
		assert dl.n_datasets == 0
		assert dl.n_total_samples == 0
	else:
		assert dl.n_datasets > 0
		assert dl.n_total_samples > 0


def test_integration_missing_metadata():
	"""Check read() if metadata.zanj is missing => should raise FileNotFoundError."""
	config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(
			source_path=SAMPLE_PROMPTS_FILE,
			char_len_min=1,
		),
		token_len_min=1,
		model_names=["gpt2"],
		prompt_token_len_tolerance=2,
	)
	dl = CollectedAttentionPatternDataloader.generate(config)

	tmp_path: Path = TEMP_DIR / "test_integration_missing_metadata"
	dl.save(tmp_path)

	# remove the metadata file
	(tmp_path / "metadata.zanj").unlink()

	with pytest.raises(FileNotFoundError):
		_ = CollectedAttentionPatternDataloader.read(tmp_path)


def test_integration_missing_dataset_files():
	"""Check read() if one dataset file is missing => should raise FileNotFoundError."""
	config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(
			source_path=SAMPLE_PROMPTS_FILE,
			char_len_min=1,
		),
		token_len_min=1,
		model_names=["gpt2"],
		prompt_token_len_tolerance=2,
	)
	dl = CollectedAttentionPatternDataloader.generate(config)

	tmp_path = TEMP_DIR / "test_integration_missing_dataset_files"
	dl.save(tmp_path)

	# remove one dataset file
	dataset_files: list[Path] = list(tmp_path.glob("dataset_*.zanj"))
	assert len(dataset_files) > 0
	dataset_files[0].unlink()

	with pytest.raises(FileNotFoundError):
		_ = CollectedAttentionPatternDataloader.read(tmp_path)


def test_integration_dummy_training_loop():
	"""A full pipeline, ending with a trivial training loop on the attention patterns."""
	config = APGenerationConfig(
		prompts_config=PromptDatasetConfig.from_source_path(
			source_path=SAMPLE_PROMPTS_FILE
		),
		token_len_min=1,
		model_names=["gpt2"],
		prompt_token_len_tolerance=2,
	)
	dl: CollectedAttentionPatternDataloader = (
		CollectedAttentionPatternDataloader.generate(config)
	)
	# We'll do a trivial "training" step: each step we compute a "loss" from the batch.
	# This ensures iteration and shapes are correct. We won't do actual backprop on a real model.

	# Just do 1 "epoch"
	for patterns_batch, meta_batch in dl.batches(batch_size=2):
		# patterns_batch is shape [B, n_ctx, n_ctx]
		# trivial "loss": the mean of the patterns + 1, squared
		if patterns_batch.numel() > 0:
			# requires grad
			patterns_batch.requires_grad_(True)
			loss = (patterns_batch.mean() + 1.0) ** 2
			loss.backward()  # see if it errors
		else:
			# no data means skip
			pass

	# If we got here without exception, the loop is good.
