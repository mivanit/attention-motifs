from __future__ import annotations

import hashlib
import base64
from typing import TYPE_CHECKING, Iterable, Iterator, TypeVar, overload
import os
import warnings

import numpy as np
from itertools import islice

if TYPE_CHECKING:
	import torch
	from jaxtyping import Float, Int

	AttentionPattern = Float[torch.Tensor, "n_ctx n_ctx"]
	AttentionPatternBatch = Float[torch.Tensor, "batch n_ctx n_ctx"]
	TokenSequence = Int[torch.Tensor, "n_ctx"]
	TokenSequenceBatch = Int[torch.Tensor, "batch n_ctx"]
	PromptHashIntSequence = Int[torch.Tensor, "n_samples"]

# custom utils

DIVIDER_S1: str = "=" * 70
"divider string for separating sections"

DIVIDER_S2: str = "-" * 50
"divider string for separating subsections"

PromptHashStr = str
PromptHashInt = int

PROMPT_HASH_BYTES: int = 4
"32 bits is enough for 4.3B unique prompts, but to avoid collision let's use 64 bits"

PROMPT_HASH_BITS: int = PROMPT_HASH_BYTES * 8


PROMPT_HASH_MAX: int = 2**PROMPT_HASH_BITS

DEFAULT_COMPRESS_LEVEL: int = 1
"zlib compression level for .npz saves: 0=none, 1=fast, 6=numpy default, 9=max"


def _load_hf_token() -> str:
	"""Load HuggingFace token from file and set HF_TOKEN env var."""
	for token_path in [".hf-token", ".meta/local/.hf-token"]:
		try:
			with open(token_path, "r") as f:
				token = f.read().strip()
			if not token.startswith("hf_"):
				raise ValueError("Invalid Hugging Face token")
			os.environ["HF_TOKEN"] = token
			print(f"Loaded HF token from {token_path}")
			return token
		except FileNotFoundError:
			continue
		except Exception as e:
			warnings.warn(
				f"Failed to get Hugging Face token -- info about certain models will be limited\n{e}"
			)
			return ""
	warnings.warn(
		"Failed to get Hugging Face token -- info about certain models will be limited\n"
		"Token not found in .hf-token or .meta/local/.hf-token"
	)
	return ""


HF_TOKEN: str = _load_hf_token()


def b64encode(data: bytes) -> str:
	return base64.b64encode(data, altchars=b"_-").decode("utf-8")


def b64decode(data: str) -> bytes:
	return base64.b64decode(data, altchars=b"_-")


def compute_text_hashes(
	text: str, max_size: int = PROMPT_HASH_BYTES
) -> tuple[int, str]:
	hash_digest: bytes = hashlib.sha256(text.encode("utf-8")).digest()
	# truncate to the desired size
	hash_digest = hash_digest[:max_size]
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


def str_batches(
	text: str,
	batch_size: int,
	allow_last_incomplete: bool = True,
) -> Iterator[str]:
	"""Yield successive batches from a tensor."""
	idx: int = 0
	while idx < len(text):
		str_slice: str = text[idx : idx + batch_size]
		if not allow_last_incomplete and len(str_slice) < batch_size:
			assert idx + batch_size >= len(text), "this state should be inaccesible"
			break
		idx += batch_size
		yield str_slice


@overload
def tensor_batches(
	arr: Float[torch.Tensor, " n_samples *data_dims"],
	batch_size: int,
	allow_last_incomplete: bool = True,
) -> Iterator[Float[torch.Tensor, " batch_size *data_dims"]]: ...


@overload
def tensor_batches(
	arr: Float[np.ndarray, " n_samples *data_dims"],
	batch_size: int,
	allow_last_incomplete: bool = True,
) -> Iterator[Float[np.ndarray, " batch_size *data_dims"]]: ...


def tensor_batches(
	arr: Float[torch.Tensor, " n_samples *data_dims"]
	| Float[np.ndarray, " n_samples *data_dims"],
	batch_size: int,
	allow_last_incomplete: bool = True,
) -> (
	Iterator[Float[torch.Tensor, " batch_size *data_dims"]]
	| Iterator[Float[np.ndarray, " batch_size *data_dims"]]
):
	"""Yield successive batches from a tensor."""
	idx: int = 0
	while idx < len(arr):
		arr_slice: (
			Float[torch.Tensor, " batch_size *data_dims"]
			| Float[np.ndarray, " batch_size *data_dims"]
		) = arr[idx : idx + batch_size]
		if not allow_last_incomplete and len(arr_slice) < batch_size:
			assert idx + batch_size >= len(arr), "this state should be inaccesible"
			break
		idx += batch_size
		# @overload narrows return type for callers; generator yields union which checker can't verify
		yield arr_slice  # pyright: ignore[reportReturnType]


@overload
def tensor_batches_indexed(
	arr: Float[torch.Tensor, " n_samples *data_dims"],
	batch_size: int | None = None,
	allow_last_incomplete: bool = True,
) -> Iterator[tuple[int, int, Float[torch.Tensor, " batch_size *data_dims"]]]: ...


@overload
def tensor_batches_indexed(
	arr: Float[np.ndarray, " n_samples *data_dims"],
	batch_size: int | None = None,
	allow_last_incomplete: bool = True,
) -> Iterator[tuple[int, int, Float[np.ndarray, " batch_size *data_dims"]]]: ...


def tensor_batches_indexed(
	arr: Float[torch.Tensor, " n_samples *data_dims"]
	| Float[np.ndarray, " n_samples *data_dims"],
	batch_size: int | None = None,
	allow_last_incomplete: bool = True,
) -> (
	Iterator[tuple[int, int, Float[torch.Tensor, " batch_size *data_dims"]]]
	| Iterator[tuple[int, int, Float[np.ndarray, " batch_size *data_dims"]]]
):
	"""Yield successive batches from a tensor."""
	if batch_size is None:
		batch_size = len(arr)
	idx_start: int = 0
	while idx_start < len(arr):
		# compute end index
		idx_end: int = idx_start + batch_size
		idx_end = min(idx_end, len(arr))
		# get slice
		arr_slice: (
			Float[torch.Tensor, " batch_size *data_dims"]
			| Float[np.ndarray, " batch_size *data_dims"]
		) = arr[idx_start:idx_end]
		# throw away last incomplete batch if not allowed
		if not allow_last_incomplete and len(arr_slice) < batch_size:
			assert idx_start + batch_size >= len(arr), (
				"this state should be inaccesible"
			)
			break
		# @overload narrows return type for callers; generator yields union which checker can't verify
		yield idx_start, idx_end, arr_slice  # type: ignore[misc]  # pyright: ignore[reportReturnType]  # ty: ignore[invalid-yield]
		# increment index
		idx_start += batch_size
