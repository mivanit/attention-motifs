import hashlib
import base64
from typing import Iterable, Iterator, TypeVar


import numpy as np
import torch
from jaxtyping import Float, Int
from itertools import islice

# custom utils

DIVIDER_S1: str = "=" * 70
"divider string for separating sections"

DIVIDER_S2: str = "-" * 50
"divider string for separating subsections"

AttentionPattern = Float[torch.Tensor, "n_ctx n_ctx"]
AttentionPatternBatch = Float[torch.Tensor, "batch n_ctx n_ctx"]
TokenSequence = Int[torch.Tensor, "n_ctx"]
TokenSequenceBatch = Int[torch.Tensor, "batch n_ctx"]

PromptHashStr = str

PROMPT_HASH_BITS: int = 64
"32 bits is enough for 4.3B unique prompts, but to avoid collision let's use 64 bits"

PROMPT_HASH_MAX: int = 2 ** PROMPT_HASH_BITS

PATTERN_DTYPE: torch.dtype = torch.float16

def b64encode(data: bytes) -> str:
	return base64.b64encode(data, altchars=b"_-").decode("utf-8")



def compute_text_hashes(text: str, max_size: int = PROMPT_HASH_MAX) -> tuple[int, str]:
	hash_digest: bytes = hashlib.sha256(text.encode("utf-8")).digest()
	# get an integer hash
	hash_int: int = int.from_bytes(hash_digest, byteorder="big")
	# truncate to the desired size
	hash_int = hash_int % max_size
	# base64 encode it
	hash_str: str = b64encode(hash_digest)

	return hash_int, hash_str


T_Sample = TypeVar("T_Sample")


def batches(
	it: Iterable,
	batch_size: int,
	allow_last_incomplete: bool = True,
) -> Iterator[list[T_Sample]]:
	"""Yield successive batches from an iterator."""
	# https://stackoverflow.com/a/61435714
	iterator: Iterator = iter(it)
	while chunk := list(islice(iterator, batch_size)):
		if not allow_last_incomplete and len(chunk) < batch_size:
			break
		yield chunk


T_Tensor = TypeVar("T_Tensor", torch.Tensor, np.ndarray)


def tensor_batches(
	arr: Float[T_Tensor, " n_samples *data_dims"],
	batch_size: int,
	allow_last_incomplete: bool = True,
) -> Iterator[Float[T_Tensor, " batch_size *data_dims"]]:
	"""Yield successive batches from a tensor."""
	idx: int = 0
	while idx < len(arr):
		arr_slice: Float[T_Tensor, " batch_size *data_dims"] = arr[
			idx : idx + batch_size
		]
		if not allow_last_incomplete and len(arr_slice) < batch_size:
			assert idx + batch_size >= len(arr), "this state should be inaccesible"
			break
		idx += batch_size
		yield arr_slice


def tensor_batches_indexed(
	arr: Float[T_Tensor, " n_samples *data_dims"],
	batch_size: int|None = None,
	allow_last_incomplete: bool = True,
) -> Iterator[tuple[int, int, Float[T_Tensor, " batch_size *data_dims"]]]:
	"""Yield successive batches from a tensor."""
	if batch_size is None:
		batch_size = len(arr)
	idx_start: int = 0
	while idx_start < len(arr):
		# compute end index
		idx_end: int = idx_start + batch_size
		idx_end = min(idx_end, len(arr))
		# get slice
		arr_slice: Float[T_Tensor, " batch_size *data_dims"] = arr[idx_start:idx_end]
		# throw away last incomplete batch if not allowed
		if not allow_last_incomplete and len(arr_slice) < batch_size:
			assert idx_start + batch_size >= len(
				arr
			), "this state should be inaccesible"
			break
		# yield (start, end, slice)
		yield idx_start, idx_end, arr_slice
		# increment index
		idx_start += batch_size
