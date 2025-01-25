import hashlib
import base64
from typing import Iterable, Iterator, Sequence, TypeVar, Iter


import numpy as np
import torch
from jaxtyping import Float, Int, Array
from itertools import islice

# custom utils

AttentionPattern = Float[torch.Tensor, "n_ctx n_ctx"]
AttentionPatternBatch = Float[torch.Tensor, "batch n_ctx n_ctx"]
TokenSequence = Int[torch.Tensor, "n_ctx"]
TokenSequenceBatch = Int[torch.Tensor, "batch n_ctx"]


def b64encode(data: bytes) -> str:
	return base64.b64encode(data, altchars=b"_-").decode("utf-8")


def compute_text_hashes(text: str) -> tuple[int, str]:
	hash_digest: bytes = hashlib.sha256(text.encode("utf-8")).digest()
	# get an integer hash
	hash_int: int = int.from_bytes(hash_digest, byteorder="big")
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

def tensor_batches(
	arr: Float[Array, " n_samples *data_dims"],
	batch_size: int,
	allow_last_incomplete: bool = True,
) -> Iterator[Float[Array, " batch_size *data_dims"]]:
	"""Yield successive batches from a tensor."""
	idx: int = 0
	while idx < len(arr):
		arr_slice: Float[Array, " batch_size *data_dims"] = arr[idx:idx + batch_size]
		if not allow_last_incomplete and len(arr_slice) < batch_size:
			assert idx + batch_size >= len(arr), "this state should be inaccesible"
			break
		idx += batch_size
		yield arr_slice