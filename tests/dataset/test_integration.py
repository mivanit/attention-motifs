# file: test_integration.py

from pathlib import Path
import json

import pytest

from attention_motifs.dataset import (
	APGenerationConfig,
	CollectedAttentionPatternDataloader,
)

TEMP_DIR: Path = Path("tests/_temp")


@pytest.fixture(scope="session", autouse=True)
def setup_temp_dir():
	"""Fixture to ensure the tests/_temp directory exists before tests run."""
	TEMP_DIR.mkdir(exist_ok=True, parents=True)
	yield
	# We do not remove it; you might want to do so in a real test environment.


@pytest.fixture
def sample_prompts_file() -> Path:
	"""Create a small prompts file inside tests/_temp with multiple lines."""
	# We'll not use a tempfile here so we can see the file in tests/_temp
	pfile = TEMP_DIR / "sample_prompts.jsonl"
	# create a few sample prompts
	prompts = [
		{"text": "Hello world!"},
		{"text": "This is a test exceeding min length."},
		{"text": "Short"},
	]
	with open(pfile, "w") as f:
		for p in prompts:
			f.write(json.dumps(p) + "\n")
	return pfile


@pytest.mark.parametrize("model_name", ["gpt2"])  # Expand or change as desired
def test_integration_generate_save_read(model_name: str, sample_prompts_file: Path):
	# We'll use an actual model name "gpt2" by default for the test.
	# If you want a smaller model, specify "sshleifer/tiny-gpt2" or similar.

	config = APGenerationConfig(
		prompts_path=sample_prompts_file,
		model_names=[model_name],
		min_length=5,  # filter out 'Short'
		max_length=50,  # won't actually chunk, just used as an example
		prompt_token_len_tolerance=2,
	)

	dl = CollectedAttentionPatternDataloader.generate(config=config, batch_size=2)

	# we expect at least one dataset
	assert dl.n_datasets >= 1
	# short prompt "Short" is filtered out
	# so we have at least 2 prompts used

	# step 1: test iteration
	all_batches = list(dl)
	# we can't assert an exact # because it depends on the model's # of layers * heads
	# but we expect something > 0
	assert len(all_batches) > 0

	# step 2: test save -> read

	tmp_path = Path(TEMP_DIR / "test_integration_generate_save_read")
	dl.save(tmp_path)

	dl2 = CollectedAttentionPatternDataloader.read(tmp_path, batch_size=2)
	assert dl2.n_datasets == dl.n_datasets
	# the set of prompts should match
	assert len(dl2.prompts) == len(dl.prompts)

	# quick iteration check
	all_batches2 = list(dl2)
	assert len(all_batches2) == len(all_batches)

	# check the patterns shape is the same
	for (pats1, meta1), (pats2, meta2) in zip(all_batches, all_batches2):
		assert pats1.shape == pats2.shape
		# metadata might differ in object identity, but check
		for m1, m2 in zip(meta1, meta2):
			assert m1.prompt_hash == m2.prompt_hash
			assert m1.model_name == m2.model_name
			assert m1.n_ctx == m2.n_ctx
