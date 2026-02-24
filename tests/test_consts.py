import torch

from attention_motifs.consts import (
	compute_text_hashes,
	b64encode,
	batches,
	tensor_batches,
	tensor_batches_indexed,
)
from pattern_lens.consts import sanitize_model_name


def test_b64encode():
	data = b"hello world"
	encoded = b64encode(data)
	assert isinstance(encoded, str)
	# Check that decoding the string (with the same altchars) matches the original
	import base64

	decoded = base64.b64decode(encoded, altchars=b"_-")
	assert decoded == data


def test_compute_text_hashes():
	text = "Hello, world!"
	hash_int, hash_str = compute_text_hashes(text)
	# Basic checks
	assert isinstance(hash_int, int)
	assert isinstance(hash_str, str)
	# Now re-compute and verify the same values come out
	hash_int2, hash_str2 = compute_text_hashes(text)
	assert hash_int == hash_int2
	assert hash_str == hash_str2


def test_batches():
	items = list(range(10))
	batch_size = 3

	collected = []
	for batch in batches(items, batch_size, allow_last_incomplete=True):
		collected.append(batch)

	# We expect:
	#   [0,1,2], [3,4,5], [6,7,8], [9]
	assert len(collected) == 4
	assert collected[0] == [0, 1, 2]
	assert collected[-1] == [9]

	# If we set allow_last_incomplete=False, we lose the final short batch
	collected_strict = []
	for batch in batches(items, batch_size, allow_last_incomplete=False):
		collected_strict.append(batch)

	# We expect: [0,1,2], [3,4,5], [6,7,8]  (the last partial batch [9] is discarded)
	assert len(collected_strict) == 3
	assert collected_strict[-1] == [6, 7, 8]


def test_tensor_batches():
	arr = torch.arange(10).float()  # shape [10], a simple 1D float Tensor
	batch_size = 3
	results = list(tensor_batches(arr, batch_size, allow_last_incomplete=True))
	# We expect [0,1,2], [3,4,5], [6,7,8], [9]
	assert len(results) == 4
	assert torch.all(results[0] == torch.tensor([0.0, 1.0, 2.0]))
	assert torch.all(results[-1] == torch.tensor([9.0]))


def test_tensor_batches_indexed():
	arr = torch.arange(10).float()  # shape [10]
	batch_size = 3
	results = list(tensor_batches_indexed(arr, batch_size, allow_last_incomplete=True))
	# Each item is (start_idx, end_idx, slice_of_data)
	# We expect:
	#   (0, 3, [0,1,2]),
	#   (3, 6, [3,4,5]),
	#   (6, 9, [6,7,8]),
	#   (9,10,[9])
	assert len(results) == 4
	(s0, e0, d0) = results[0]
	(s_last, e_last, d_last) = results[-1]
	assert s0 == 0 and e0 == 3
	assert s_last == 9 and e_last == 10
	assert torch.all(d0 == torch.tensor([0.0, 1.0, 2.0]))
	assert torch.all(d_last == torch.tensor([9.0]))


def test_sanitize_model_name_passthrough() -> None:
	assert sanitize_model_name("gpt2-small") == "gpt2-small"


def test_sanitize_model_name_slash() -> None:
	assert sanitize_model_name("meta-llama/Llama-3.2-1B") == "meta-llama-Llama-3.2-1B"


def test_sanitize_model_name_multiple_slashes() -> None:
	assert sanitize_model_name("org/sub/model") == "org-sub-model"


def test_sanitize_model_name_preserves_dots_hyphens() -> None:
	assert sanitize_model_name("model-v1.0_test") == "model-v1.0_test"


def test_sanitize_model_name_idempotent() -> None:
	name: str = "meta-llama/Llama-3.2-1B"
	assert sanitize_model_name(sanitize_model_name(name)) == sanitize_model_name(name)
