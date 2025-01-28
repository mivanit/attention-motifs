import json
import pytest
from pathlib import Path

from attention_motifs.consts import (
	compute_text_hashes,
)

from attention_motifs.dataset.prompts import (
	Prompt,
	PromptDataset,
	PromptDatasetConfig,
)

# The directory where test output will be written
TEMP_DIR: Path = Path("tests/_temp")


@pytest.fixture(scope="session", autouse=True)
def ensure_temp_dir():
	"""
	A fixture that ensures the TEMP_DIR directory exists before any test runs.
	It will not be cleaned up after tests.
	"""
	TEMP_DIR.mkdir(parents=True, exist_ok=True)
	return TEMP_DIR


# -------------------------
# Tests for Prompt classes
# -------------------------


def test_prompt_from_text():
	txt = "Example text"
	prompt = Prompt.from_text(txt)
	assert prompt.text == txt
	# Check that the hash matches compute_text_hashes
	hash_int, hash_str = compute_text_hashes(txt)
	assert prompt.hash_int == hash_int
	assert prompt.hash_str == hash_str
	assert prompt.meta == {}


def test_prompt_from_dict_no_hash():
	"""
	Test that from_dict() correctly computes missing hashes
	and places the rest of the keys in the meta dictionary.
	"""
	data = {"text": "Hello from dict", "some_meta_key": "some_meta_value"}
	p = Prompt.from_dict(data)
	# Check that the text is correct
	assert p.text == "Hello from dict"
	# Check that from_dict() computed the hash
	recomputed_int, recomputed_str = compute_text_hashes("Hello from dict")
	assert p.hash_int == recomputed_int
	assert p.hash_str == recomputed_str
	# Check that meta was populated
	assert p.meta["some_meta_key"] == "some_meta_value"


def test_prompt_from_dict_with_hash():
	"""
	Test that from_dict() accepts existing correct hash values.
	"""
	text = "Hello from dict with correct hash"
	correct_hash_int, correct_hash_str = compute_text_hashes(text)
	data = {
		"text": text,
		"hash_int": correct_hash_int,
		"hash_str": correct_hash_str,
		"extra_key": 123,
	}
	p = Prompt.from_dict(data)
	assert p.text == text
	assert p.hash_int == correct_hash_int
	assert p.hash_str == correct_hash_str
	# The extra key should appear in meta
	assert p.meta["extra_key"] == 123


def test_prompt_from_dict_with_incorrect_hash():
	"""
	Test that from_dict() raises an AssertionError if the provided hash
	doesn't match the computed hash.
	"""
	text = "Hello from dict but with mismatch"
	correct_hash_int, correct_hash_str = compute_text_hashes(text)

	# Provide an incorrect hash_int on purpose
	data = {
		"text": text,
		"hash_int": correct_hash_int + 1,  # mismatch here
		"hash_str": correct_hash_str,
	}
	with pytest.raises(AssertionError):
		_ = Prompt.from_dict(data)

	# Provide an incorrect hash_str on purpose
	data2 = {
		"text": text,
		"hash_int": correct_hash_int,
		"hash_str": correct_hash_str + "ABCD",  # mismatch here
	}
	with pytest.raises(AssertionError):
		_ = Prompt.from_dict(data2)


def test_prompt_getitem():
	p = Prompt.from_text("Testing Prompt __getitem__")
	p.meta["foo"] = "bar"
	# Using "text"
	assert p["text"] == "Testing Prompt __getitem__"
	# Using meta
	assert p["foo"] == "bar"
	# Using "hash"
	# The "hash" case in __getitem__ returns p.hash, which is ambiguous in code snippet
	# If you meant to do return (hash_int, hash_str) or just hash_str, adapt accordingly.
	# The snippet does `case "hash": return self.hash`, but the code doesn't define `self.hash`.
	# For now, let's assume you want to test returning the integer hash or some combined thing:
	# We'll skip actually checking it if your code doesn't define it.
	# If you had `property hash(self) -> int`, you'd check that.
	# We'll do a minimal check that it doesn't error:
	_ = p["hash_int"]
	_ = p["hash_str"]
	# Using an invalid key
	with pytest.raises(KeyError):
		_ = p["invalid_key"]


# -------------------------
# Tests for PromptDataset*
# -------------------------


@pytest.fixture
def example_prompts_data():
	"""
	Returns a list of dictionaries that can be written to a JSONL file.
	Includes some with explicit hash, some without, to test the from_config loader.
	"""
	# We'll have 3 distinct lines
	data_list = []

	# 1) no hash fields
	data_list.append(
		{"text": "First prompt no hash fields", "some_meta_key": "meta_val_1"}
	)

	# 2) correct hash fields
	text2 = "Second prompt with correct hash"
	h2_int, h2_str = compute_text_hashes(text2)
	data_list.append(
		{"text": text2, "hash_int": h2_int, "hash_str": h2_str, "extra_info": True}
	)

	# 3) no hash fields again
	data_list.append({"text": "Third prompt, also no hash fields", "numeric_meta": 999})

	return data_list


def test_prompt_dataset_config_from_source_path(ensure_temp_dir, example_prompts_data):
	jsonl_path = TEMP_DIR / "test_prompts_config.jsonl"

	# Write out the example prompts
	with open(jsonl_path, "w", encoding="utf-8") as f:
		for row in example_prompts_data:
			f.write(json.dumps(row) + "\n")

	# Now create the config
	config = PromptDatasetConfig.from_source_path(jsonl_path)
	assert config.name == jsonl_path.stem, (
		"Expect the config name to match the file stem"
	)
	assert config.source_path == jsonl_path
	assert config.source_info["source_path"] == jsonl_path.as_posix()


def test_prompt_dataset_config_missing_path():
	"""
	Test that from_source_path raises FileNotFoundError if path is missing.
	"""
	bogus_path = Path("tests/_temp/this_file_does_not_exist.jsonl")
	with pytest.raises(FileNotFoundError):
		_ = PromptDatasetConfig.from_source_path(bogus_path)


def test_prompt_dataset_from_prompts(ensure_temp_dir):
	# We'll manually build some Prompts, then from_prompts them
	p1 = Prompt.from_text("Hello DS1")
	p2 = Prompt.from_text("Hello DS2")
	config = PromptDatasetConfig(
		name="my_dataset",
		source_path=Path("dummy/path"),  # doesn't matter here
		source_info={"desc": "testing from_prompts"},
	)
	ds = PromptDataset.from_prompts(config, [p1, p2])

	assert len(ds) == 2
	# The hash_map uses p.hash_str -> index
	assert p1.hash_str in ds.hash_map
	assert ds.hash_map[p1.hash_str] == 0
	# test the indexing
	assert ds.index_get(0) is p1
	assert ds.index_get(1) is p2
	# test iteration
	all_prompts = list(iter(ds))
	assert len(all_prompts) == 2
	# test hash-based retrieval
	print(ds.hash_map)
	print(p1.hash_str)
	print(p2.hash_str)
	print(p1.hash_int)
	print(p2.hash_int)
	assert ds.hash_str_get(p1.hash_str) is p1
	assert ds.hash_int_get(p2.hash_int) is p2

	# test hash_get
	assert ds.hash_get(p2.hash_str) is p2
	assert ds.hash_get(p2.hash_int) is p2

	# invalid type for hash_get
	with pytest.raises(TypeError):
		ds.hash_get(3.14159)


def test_prompt_dataset_from_config(ensure_temp_dir, example_prompts_data):
	"""
	Comprehensive test of from_config, verifying that lines lacking hash are computed,
	lines with hash are respected, and that the final dataset is consistent.
	"""
	jsonl_path = TEMP_DIR / "test_prompt_dataset.jsonl"
	with open(jsonl_path, "w", encoding="utf-8") as f:
		for row in example_prompts_data:
			f.write(json.dumps(row) + "\n")

	# Create config and load
	config = PromptDatasetConfig.from_source_path(jsonl_path, char_len_min=1, char_len_max=99999)
	ds = PromptDataset.from_config(config)

	# Check length
	assert len(ds) == len(example_prompts_data), "All lines must be loaded."

	# Check that the hash_map is built
	for i, row in enumerate(example_prompts_data):
		loaded_prompt = ds.index_get(i)
		# Confirm text matches
		assert loaded_prompt.text == row["text"]
		# Confirm meta keys
		# (the difference between row keys and PROMPT_SPECIAL_KEYS ends up in meta)
		# your code's PROMPT_SPECIAL_KEYS = {"text", "hash_int", "hash_str"}
		# so everything else should be in meta
		for k, v in row.items():
			if k not in ("text", "hash_int", "hash_str"):
				assert loaded_prompt.meta[k] == v

		# Confirm the dataset's hash_map points back to the correct index
		assert ds.hash_map[loaded_prompt.hash_str] == i

		# If the row had no hash fields, they should have been computed
		# If they existed, they should match
		recomputed_int, recomputed_str = compute_text_hashes(row["text"])
		assert loaded_prompt.hash_int == recomputed_int
		assert loaded_prompt.hash_str == recomputed_str


def test_prompt_dataset_serialization_roundtrip(ensure_temp_dir):
	"""
	Test that a PromptDataset can be serialized, saved to disk, reloaded, and remain consistent.
	"""
	# 1) Create a small dataset in memory
	p1 = Prompt.from_text("Roundtrip 1")
	p2 = Prompt.from_text("Roundtrip 2")
	config = PromptDatasetConfig(
		name="roundtrip_ds",
		source_path=TEMP_DIR / "roundtrip.jsonl",
		source_info={"test_field": "roundtrip_demo"},
	)
	ds_orig = PromptDataset.from_prompts(config, [p1, p2])

	# 2) Serialize ds_orig to a dictionary
	ds_dict = ds_orig.serialize()

	# 3) Write that dictionary as JSON to a file
	out_path = TEMP_DIR / "prompt_dataset_roundtrip.json"
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(ds_dict, f, indent=2)

	# 4) Read it back
	with open(out_path, "r", encoding="utf-8") as f:
		loaded_dict = json.load(f)

	# 5) Use PromptDataset.load() to build a new object
	ds_new = PromptDataset.load(loaded_dict)

	# 6) Compare ds_new with ds_orig
	assert len(ds_new) == len(ds_orig)
	for i in range(len(ds_orig)):
		assert ds_new.index_get(i).text == ds_orig.index_get(i).text
		assert ds_new.index_get(i).hash_int == ds_orig.index_get(i).hash_int
		assert ds_new.index_get(i).hash_str == ds_orig.index_get(i).hash_str

	# The config also should match
	assert ds_new.config.name == ds_orig.config.name
	assert ds_new.config.source_path == ds_orig.config.source_path
	assert ds_new.config.source_info["test_field"] == "roundtrip_demo"


def test_prompt_dataset_hash_collisions(ensure_temp_dir):
	"""
	(Optional) In the extremely rare event of collisions for SHA-256, we can't do much.
	But you might want to test if your code gracefully handles duplicates in the input.
	For demonstration, we'll create two lines with identical text & see what happens.
	"""
	text = "Duplicate text for collision test"
	# We'll place two identical lines in the file
	data_list = [
		{"text": text},
		{"text": text},
	]
	collision_path = TEMP_DIR / "collision_test.jsonl"
	with open(collision_path, "w", encoding="utf-8") as f:
		for row in data_list:
			f.write(json.dumps(row) + "\n")

	config = PromptDatasetConfig.from_source_path(collision_path, char_len_min=1)
	ds = PromptDataset.from_config(config)

	print(ds)

	# We now have two distinct prompts in ds.prompts, but they share the same hash.
	# The current code uses `hash_map: dict[PromptHashStr, int] = { p.hash_str: i ... }`
	# That means the final index in the hash_map will be the last prompt's index.
	# i.e., it overwrote the earlier one. Let's verify that:
	last_idx = len(ds) - 1
	p_last = ds.index_get(last_idx)
	# They have the same text, so they have the same hash.
	assert ds.hash_map[p_last.hash_str] == last_idx
	# Attempt retrieving by hash_str
	# We'll only get the last one in that list
	retrieved = ds.hash_str_get(p_last.hash_str)
	assert retrieved is p_last

	# This doesn't truly solve collisions, but it demonstrates the behavior of the code.
