from collections import defaultdict

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

# custom utils

from attention_motifs.consts import (
	AttentionPatternBatch,
	TokenSequence,
	AttentionPatternMetadata,
)


def tokenize_and_bin_prompts(
	model: HookedTransformer,
	prompts: list[dict],
	token_len_min: int,
	tolerance: int,
) -> dict[int, list[tuple[dict, list[int]]]]:
	"""Tokenize prompts and bin them by sequence length.

	# Parameters:
	- `model : HookedTransformer`
		Model to use for tokenization
	- `prompts : list[dict]`
		List of prompt dictionaries
	- `token_len_min : int`
		Minimum token length to consider
	- `tolerance : int`
		anything longer than but within `tolerance` of the bin size will be truncated to the bin size

	# Returns:
	- `dict[int, list[tuple[dict, list[int]]]]`
		Mapping from bin centers to list of (prompt, tokens) pairs
	"""
	# tokenize all prompts
	tokenized_prompts: list[tuple[dict, TokenSequence]] = [
		(p, model.to_tokens(p["text"])) for p in prompts
	]

	# group by rounded length
	bins_by_len: defaultdict[
		int,
		list[
			tuple[
				dict,  # prompt and metadata
				TokenSequence,  # tokenized sequence
			]
		],
	] = defaultdict(list)
	for prompt, tokens in tokenized_prompts:
		# skip if too short
		if len(tokens) >= token_len_min:
			desired_len: int = len(tokens) - len(tokens) % tolerance
			tokens_truncated: TokenSequence = tokens[:desired_len]
			bins_by_len[desired_len].append((prompt, tokens_truncated))

	return bins_by_len


def process_length_bin(
	model: HookedTransformer,
	bin_contents: list[tuple[dict, TokenSequence]],
	model_name: str,
	max_batch_size: int | None = None,
) -> tuple[AttentionPatternBatch, list[AttentionPatternMetadata]]:
	"""Process a single bin of same-length sequences.

	# Parameters:
	- `model : HookedTransformer`
		Model to extract patterns from
	- `bin_contents : list[tuple[dict, TokenSequence]]`
		List of (prompt, tokens) pairs in this bin
	- `model_name : str`
		Name of the model (for metadata)

	# Returns:
	- `AttentionPatternBatch`
		Batch of attention patterns
	- `list[AttentionPatternMetadata]`
		List of metadata for each pattern (in order)
	"""
	# batch process through model
	tokens_tensor: Float[torch.Tensor, "batch n_ctx"] = torch.tensor(
		truncated_tokens, device=model.cfg.device
	)

	# get attention patterns
	_, cache = model.run_with_cache(
		tokens_tensor,
		return_type=None,
		names_filter=lambda n: n.endswith("pattern"),
	)

	# extract patterns for each layer and head
	patterns_and_meta: list[
		tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]
	] = []
	layer: int
	head: int
	for layer in range(model.cfg.n_layers):
		for head in range(model.cfg.n_heads):
			# get patterns for this head
			patterns: Float[torch.Tensor, "batch n_ctx n_ctx"] = cache[
				f"blocks.{layer}.attn.hook_pattern"
			][:, head, :, :]

			# create metadata for each pattern
			meta_list: list[AttentionPatternMetadata] = [
				AttentionPatternMetadata(
					model_name=model_name,
					idx_layer=layer,
					idx_head=head,
					prompt_hash=p["hash"],
					n_ctx=min_len,
				)
				for p, _ in bin_contents
			]

			# add all patterns from this head
			i: int
			for i in range(len(patterns)):
				patterns_and_meta.append((patterns[i], meta_list[i]))

	return patterns_and_meta
