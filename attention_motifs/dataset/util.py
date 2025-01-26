from collections import defaultdict
from typing import Callable

import torch
from jaxtyping import Float, Int64
from transformer_lens import HookedTransformer

# custom utils


from jaxtyping import Int

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)

from attention_motifs.consts import (
	AttentionPattern,
	AttentionPatternBatch,
	TokenSequence,
	TokenSequenceBatch,
	PromptHashStr,
	compute_text_hashes,
	tensor_batches_indexed,
)

from attention_motifs.dataset.prompts import PromptDataset

AttentionPatternMetadataTuple = tuple[str, int, int, PromptHashStr, int]


@serializable_dataclass
class AttentionPatternMetadata(SerializableDataclass):
	model_name: str
	idx_layer: int
	idx_head: int
	prompt_hash: PromptHashStr
	n_ctx: int

	def tuple(self) -> AttentionPatternMetadataTuple:
		return (
			self.model_name,
			self.idx_layer,
			self.idx_head,
			self.prompt_hash,
			self.n_ctx,
		)
	
	@classmethod
	def from_tuple(cls, tup: AttentionPatternMetadataTuple) -> "AttentionPatternMetadata":
		return cls(
			model_name=tup[0],
			idx_layer=tup[1],
			idx_head=tup[2],
			prompt_hash=tup[3],
			n_ctx=tup[4],
		)

	def hash_int(self) -> int:
		return compute_text_hashes(self.tuple())[0]

	def hash_str(self) -> str:
		return compute_text_hashes(self.tuple())[1]

	def __hash__(self) -> int:
		return self.hash_int()


@serializable_dataclass
class AttentionPatternMetadataArray(SerializableDataclass):
	model_names_map: list[str]
	data: Int64[]

@serializable_dataclass
class AttentionPatternDataset(SerializableDataclass):
	n_ctx: int
	n_patterns: int
	patterns: AttentionPatternBatch
	metadata: list[AttentionPatternMetadata] = serializable_field(
		serialization_fn=lambda x: [m.tuple() for m in x],
		deserialize_fn=lambda x: [AttentionPatternMetadata.from_tuple(t) for t in x],
	)
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
) -> dict[int, tuple[list[PromptHashStr], TokenSequenceBatch]]:
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
	- `dict[int, list[tuple[str, TokenSequence]]]`
		Mapping from bin centers to list of (prompt, tokens) pairs
	"""
	# tokenize all prompts
	# keep only hash_str, we can recover the text from the dataset
	tokenized_prompts: list[tuple[str, TokenSequence]] = [
		(p.hash_str, model.to_tokens(p.text)[0]) for p in prompts
	]

	print(f"Tokenized {len(tokenized_prompts)} prompts")

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

	print(f"Grouped into {len(bins_by_len)} bins")

	print({k: (len(v1), len(v2)) for k, (v1, v2) in bins_by_len.items()})

	output: dict[int, tuple[list[PromptHashStr], TokenSequenceBatch]] = {
		n_ctx: (prompt_hash, torch.stack(token_seqs_list, dim=0))
		for n_ctx, (prompt_hash, token_seqs_list) in bins_by_len.items()
	}

	print({k : (len(v1), v2.shape) for k, (v1, v2) in output.items()})

	return output


def process_length_bin(
	model: HookedTransformer,
	n_ctx: int,
	prompt_hashes: list[PromptHashStr],
	tokens_tensor: TokenSequenceBatch,
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
	 - `prompt_hashes : list[PromptHashStr]`
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
		# get attention patterns
		_, cache = model.run_with_cache(
			tokens_tensor,
			return_type=None,
			names_filter=names_filter,
		)

		# extract patterns for each layer and head
		layer: int
		head: int
		for layer in range(model.cfg.n_layers):
			layer_patterns: Float[torch.Tensor, "batch head_idx n_ctx n_ctx"] = cache[
				key_format.format(layer=layer)
			]
			for head in range(model.cfg.n_heads):
				# get patterns for this head
				head_patterns: AttentionPatternBatch = layer_patterns[:, head]

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
				output_patterns.append(head_patterns)
				output_metadata.extend(meta_list)

	# concatenate patterns
	output_patterns_tensor: AttentionPatternBatch = torch.cat(output_patterns, dim=0)

	# tensor shape sanity check
	assert tuple(output_patterns_tensor.shape) == (
		len(output_metadata),
		n_ctx,
		n_ctx,
	)
	return output_patterns_tensor, output_metadata
