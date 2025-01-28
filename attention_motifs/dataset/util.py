from collections import defaultdict
from typing import Callable, overload

import numpy as np
import torch
from jaxtyping import Float, UInt64, Int, UInt16
from transformer_lens import HookedTransformer


# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from muutils.errormode import ErrorMode

from attention_motifs.consts import (
	AttentionPattern,
	AttentionPatternBatch,
	TokenSequence,
	TokenSequenceBatch,
	PromptHashStr,
	PromptHashIntSequence,
	compute_text_hashes,
	tensor_batches_indexed,
	PromptHashInt,
)

from attention_motifs.dataset.prompts import PromptDataset

AttentionPatternMetadataTuple = tuple[str, int, int, int, PromptHashInt]


@serializable_dataclass
class AttentionPatternMetadata(SerializableDataclass):
	model_name: str
	idx_layer: int
	idx_head: int
	n_ctx: int
	prompt_hash: PromptHashInt

	def tuple(self) -> AttentionPatternMetadataTuple:
		return (
			self.model_name,
			self.idx_layer,
			self.idx_head,
			self.n_ctx,
			self.prompt_hash,
		)

	@classmethod
	def from_tuple(
		cls, tup: AttentionPatternMetadataTuple
	) -> "AttentionPatternMetadata":
		return cls(
			model_name=tup[0],
			idx_layer=tup[1],
			idx_head=tup[2],
			n_ctx=tup[3],
			prompt_hash=tup[4],
		)

	def tuple_contrastive(self) -> tuple[str, int, int]:
		return (self.model_name, self.idx_layer, self.idx_head)

	def hash_int(self) -> int:
		return compute_text_hashes(self.tuple())[0]

	def hash_str(self) -> str:
		return compute_text_hashes(self.tuple())[1]

	def __hash__(self) -> int:
		return self.hash_int()

	@classmethod
	def contrastive_classes(
		cls,
		metadata: "list[AttentionPatternMetadata]",
	) -> Int["batch"]:
		"class matches if everything but prompt hash and n_ctx matches"

		contrastive_tuples: list[tuple] = [m.tuple_contrastive() for m in metadata]

		classes: set[tuple] = set(contrastive_tuples)

		# create mapping
		class_map: dict[AttentionPatternMetadataTuple, int] = {
			tup: idx for idx, tup in enumerate(classes)
		}

		# create output
		output: Int["batch"] = torch.array(
			[class_map[t] for t in contrastive_tuples],
			dtype=torch.int,
		)

		return output


# TODO: why is it warning us here? look into that error, ignoring for now.
@serializable_dataclass(on_typecheck_mismatch=ErrorMode.IGNORE)
class AttentionPatternMetadataArray(SerializableDataclass):
	model_names_map: list[str]
	# TODO: wtf? why are these not being deserialized properly?
	data: UInt16[np.ndarray, " model_name/idx_layer/idx_head/n_ctx=4 n_patterns"] = (
		serializable_field(
			deserialize_fn=lambda x: x["data"],
		)
	)
	prompt_hash: UInt64[np.ndarray, " n_patterns"] = serializable_field(
		deserialize_fn=lambda x: x["data"],
	)
	n_samples: int

	def __len__(self) -> int:
		return self.n_samples

	@overload
	def __getitem__(self, idx: int) -> AttentionPatternMetadata: ...
	@overload
	def __getitem__(self, idx: slice) -> list[AttentionPatternMetadata]: ...
	def __getitem__(
		self, idx: int | slice
	) -> AttentionPatternMetadata | list[AttentionPatternMetadata]:
		if isinstance(idx, slice):
			return [
				AttentionPatternMetadata(
					model_name=self.model_names_map[model_name_idx],
					idx_layer=layer,
					idx_head=head,
					n_ctx=n_ctx,
					prompt_hash=self.prompt_hash[idx],
				)
				for model_name_idx, layer, head, n_ctx in self.data[idx]
			]
		elif isinstance(idx, int):
			model_name_idx, layer, head, n_ctx = self.data[idx]
			return AttentionPatternMetadata(
				model_name=self.model_names_map[model_name_idx],
				idx_layer=layer,
				idx_head=head,
				n_ctx=n_ctx,
				prompt_hash=self.prompt_hash[idx],
			)
		else:
			raise TypeError(f"Invalid index type: {type(idx) = }, {idx = }")

	@classmethod
	def from_list(
		cls,
		metadata: list[AttentionPatternMetadata],
	) -> "AttentionPatternMetadataArray":
		model_names: set[str] = {m.model_name for m in metadata}
		model_names_map: list[str] = sorted(list(model_names))
		model_names_map_inv: dict[str, int] = {
			m: i for i, m in enumerate(model_names_map)
		}

		# allocate output
		n_samples: int = len(metadata)
		data: UInt16[
			np.ndarray, " model_name/idx_layer/idx_head/n_ctx=4 n_patterns"
		] = np.full((n_samples, 4), fill_value=0, dtype=np.uint16)
		prompt_hash: UInt64[np.ndarray, " n_patterns"] = np.zeros(
			n_samples, dtype=np.uint64
		)

		# fill in data
		for idx, m in enumerate(metadata):
			data[idx] = np.array(
				[
					model_names_map_inv[m.model_name],
					m.idx_layer,
					m.idx_head,
					m.n_ctx,
				],
				dtype=np.uint16,
			)
			prompt_hash[idx] = m.prompt_hash

		return cls(
			model_names_map=model_names_map,
			data=data,
			prompt_hash=prompt_hash,
			n_samples=n_samples,
		)


@serializable_dataclass
class AttentionPatternDataset(SerializableDataclass):
	n_ctx: int
	n_patterns: int
	patterns: AttentionPatternBatch
	metadata: AttentionPatternMetadataArray
	raw_scores: bool = serializable_field(default=False)

	def __len__(self) -> int:
		return self.n_patterns

	def shuffle(self) -> None:
		"shuffle the dataset in-place"
		perm: Int[torch.Tensor, " n_patterns"] = torch.randperm(self.n_patterns)
		self.patterns = self.patterns[perm]
		self.metadata = [self.metadata[i] for i in perm]

	def __getitem__(
		self,
		idx: int | slice,
	) -> tuple[AttentionPattern, AttentionPatternMetadata]:
		return self.patterns[idx], self.metadata[idx]


def tokenize_and_bin_prompts(
	model: HookedTransformer,
	prompts: PromptDataset,
	token_len_min: int,
	tolerance: int,
	storage_device: torch.device,
) -> dict[int, tuple[list[PromptHashInt], TokenSequenceBatch]]:
	"""Tokenize prompts and bin them by sequence length.

	# Parameters:
	- `model : HookedTransformer`
		Model to use for tokenization
	- `prompts : PromptDataset`
		prompts to tokenize
	- `token_len_min : int`
		Minimum token length to consider
	- `tolerance : int`
		anything longer than but within `tolerance` of the bin size will be truncated to the bin size

	# Returns:
	- `dict[int, list[tuple[PromptHashInt, TokenSequence]]]`
		Mapping from bin size to list of (prompt, tokens) pairs
	"""
	# tokenize all prompts
	# keep only hash_str, we can recover the text from the dataset
	tokenized_prompts: list[tuple[PromptHashInt, TokenSequence]] = [
		(
			p.hash_int,
			model.to_tokens(p.text)[0].to(storage_device),
		)
		for p in prompts
	]

	# group by rounded length
	bins_by_len: defaultdict[
		int,
		tuple[
			list[PromptHashStr],  # prompt hash, can look it up in the dataset
			list[TokenSequence],  # tokenized sequence
		],
	] = defaultdict(lambda: ([], []))

	# iterare over all tokenized prompts
	for prompt_hash, tokens in tokenized_prompts:
		# skip if too short
		if len(tokens) >= token_len_min:
			# round down to nearest bin
			desired_len: int = len(tokens) - len(tokens) % tolerance
			tokens_truncated: TokenSequence = tokens[:desired_len]

			bins_by_len[desired_len][0].append(prompt_hash)
			bins_by_len[desired_len][1].append(tokens_truncated)
			# print(bins_by_len)
		else:
			pass
			# print(f"Skipping prompt with too few tokens: {len(tokens) = }, {token_len_min = }, {tokens = }")

	output: dict[int, tuple[list[PromptHashInt], TokenSequenceBatch]] = {
		n_ctx: (
			prompt_hashes,
			torch.stack(token_seqs_list, dim=0).to(storage_device),
		)
		for n_ctx, (prompt_hashes, token_seqs_list) in bins_by_len.items()
	}

	return output


MHABatched = Float[torch.Tensor, "batch head_idx n_ctx n_ctx"]


def process_length_bin(
	model: HookedTransformer,
	n_ctx: int,
	prompt_hashes: PromptHashIntSequence,
	tokens_tensor: TokenSequenceBatch,
	model_device: torch.device,
	storage_device: torch.device,
	raw_scores: bool = False,
	model_name: str | None = None,
	max_batch_size: int | None = None,
) -> tuple[AttentionPatternBatch, list[AttentionPatternMetadata]]:
	"""Process a single bin of same-length sequences.

	# Parameters:
	 - `model : HookedTransformer`
	   Model to extract patterns from
	 - `n_ctx : int`
	   expected context length
	 - `prompt_hashes : PromptHashIntSequence`
	   List of prompt hashes (in order)
	 - `tokens : TokenSequenceBatch`
	   tensor of tokenized sequences
	 - `model_name : str | None`
	   name of model for metadata (if `None`, will be set to `model.cfg.model_name`)
	   (defaults to `None`)
	 - `max_batch_size : int | None`
	   max batch size for feeding into the model
	   (defaults to `None`)
	 - `raw_scores : bool`
	   returns raw scores if `True` or processed lower-triangular row-stochastic patterns if `False`
	   (defaults to `False`)

	# Returns:

	`tuple[AttentionPatternBatch, list[AttentionPatternMetadata]]`

	- `AttentionPatternBatch`
		Batch of attention patterns
	- `list[AttentionPatternMetadata]`
		List of metadata for each pattern (in order)
	"""
	# set model name
	if model_name is None:
		model_name = model.cfg.model_name

	# set up filter and key format
	names_filter: Callable[[str], bool] = (  # noqa: E731
		lambda s: s.endswith("scores")
		if raw_scores
		else lambda s: s.endswith("pattern")
	)
	key_format: str = (
		"blocks.{layer}.attn.hook_attn_scores"
		if raw_scores
		else "blocks.{layer}.attn.hook_pattern"
	)

	# allocate output
	output_patterns: list[AttentionPatternBatch] = list()
	output_metadata: list[AttentionPatternMetadata] = list()

	# batch process through model
	for idx_start, idx_end, tokens_batch in tensor_batches_indexed(
		tokens_tensor, max_batch_size
	):
		# print(f"Processing batch {idx_start=}, {idx_end=}")
		# print(f"{tokens_batch.shape=}")
		# get attention patterns
		cache: dict[str, MHABatched]
		with torch.no_grad():
			_, cache = model.run_with_cache(
				tokens_batch.to(model_device),
				return_type=None,
				names_filter=names_filter,
				return_cache_object=False,
			)
		# print(f"\tforwards done")

		# extract patterns for each layer and head
		layer: int
		head: int
		for layer in range(model.cfg.n_layers):
			layer_key: str = key_format.format(layer=layer)
			layer_patterns: MHABatched = cache[layer_key]
			layer_patterns.to(storage_device)
			for head in range(model.cfg.n_heads):
				# get patterns for this head
				head_patterns: AttentionPatternBatch = layer_patterns[:, head]

				# TODO: create AttentionPatternMetadataArray here instead, then concatenate them all at the end
				# will require messing around with the model index, maybe make that a hash?

				# create metadata for each pattern
				meta_list: list[AttentionPatternMetadata] = [
					AttentionPatternMetadata(
						model_name=model_name,
						idx_layer=layer,
						idx_head=head,
						prompt_hash=p,
						n_ctx=n_ctx,
					)
					for p in prompt_hashes[idx_start:idx_end]
				]

				# append to output
				output_patterns.append(head_patterns.to(storage_device))
				output_metadata.extend(meta_list)

			del layer_patterns
			del cache[layer_key]

		# delete cache to free up memory
		del cache

	# concatenate patterns
	output_patterns_tensor: AttentionPatternBatch = torch.cat(
		output_patterns, dim=0
	).to(storage_device)

	# tensor shape sanity check
	assert tuple(output_patterns_tensor.shape) == (
		len(output_metadata),
		n_ctx,
		n_ctx,
	)
	return output_patterns_tensor, output_metadata
