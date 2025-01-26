from collections import defaultdict
from typing import Callable

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

# custom utils

import json
from pathlib import Path


from jaxtyping import Int

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
	JSONitem,
)

from attention_motifs.consts import (
	AttentionPattern,
	AttentionPatternBatch,
	TokenSequence,
	TokenSequenceBatch,
	PromptHashStr,
	b64encode,
	compute_text_hashes,
	tensor_batches_indexed,
)


@serializable_dataclass
class AttentionPatternMetadata(SerializableDataclass):
	model_name: str
	idx_layer: int
	idx_head: int
	prompt_hash: PromptHashStr
	n_ctx: int

	def tuple(self) -> tuple[str, int, int, PromptHashStr, int]:
		return (
			self.model_name,
			self.idx_layer,
			self.idx_head,
			self.prompt_hash,
			self.n_ctx,
		)

	def hash_int(self) -> int:
		return compute_text_hashes(self.tuple())[0]

	def hash_str(self) -> str:
		return compute_text_hashes(self.tuple())[1]

	def __hash__(self) -> int:
		return self.hash_int()


PROMPT_SPECIAL_KEYS: set[str] = {"text", "hash_int", "hash_str"}


@serializable_dataclass
class Prompt(SerializableDataclass):
	"""A prompt is a dictionary with a text key and an optional hash key."""

	text: str
	hash_int: int
	hash_str: PromptHashStr
	meta: dict[str, JSONitem]

	@classmethod
	def from_dict(cls, data: dict[str, JSONitem]) -> "Prompt":
		assert "text" in data

		# compute hashes if not present
		if ("hash_int" not in data) or ("hash_str" not in data):
			# if either is missing, recompute the hashes
			hash_int: int
			hash_str: str
			hash_int, hash_str = compute_text_hashes(data["text"])

			# and assert they match, if present
			assert data.get("hash_int", hash_int) == hash_int
			assert data.get("hash_str", hash_str) == hash_str

			# write to the data dict
			data["hash_int"] = hash_int
			data["hash_str"] = hash_str

		# return class
		return cls(
			text=data["text"],
			hash_int=data["hash_int"],
			hash_str=data["hash_str"],
			# anything else is in the metadata dict
			meta={k: v for k, v in data.items() if k not in PROMPT_SPECIAL_KEYS},
		)

	@classmethod
	def from_text(cls, text: str) -> "Prompt":
		hash_int, hash_str = compute_text_hashes(text)
		return cls(
			text=text,
			hash_int=hash_int,
			hash_str=hash_str,
			meta={},
		)

	def __getitem__(self, key: str) -> JSONitem:
		match key:
			case "hash":
				return self.hash
			case "text":
				return self.text
			case _:
				return self.meta[key]

	def __hash__(self) -> int:
		return self.hash_int


@serializable_dataclass
class PromptDatasetConfig(SerializableDataclass):
	"""holds the config for a prompt dataset"""

	name: str
	source_path: Path = serializable_field(
		serialization_fn=lambda p: p.as_posix(),
		deserialize_fn=lambda data: Path(data),
	)
	source_info: dict[str, JSONitem]

	@classmethod
	def from_source_path(cls, source_path: Path) -> "PromptDatasetConfig":
		return cls(
			name=source_path.stem,
			source_path=source_path,
			source_info={
				"source_path": source_path.as_posix(),
			},
		)


@serializable_dataclass
class PromptDataset(SerializableDataclass):
	"""holds a dataset of prompts

	# Parameters:
	 - `prompts: list[Prompt]`
	        a list of `Prompt` objects
	 - `hash_map: dict[str, int]`
	        a mapping from prompt hash to index in the `prompts` list.
		we use a the hash as a b64 encoded string as the key instead of an int for two reasons:
		 - int -> b64 string is easier than the reverse
		 - json dict keys must be strings, not ints, and so we avoid an unnecessary conversion

	we avoid storing the prompts in a dict to preserve order, and also to eventually dump them into a jsonl file
	"""

	config: PromptDatasetConfig
	prompts: list[Prompt] = serializable_field(
		serialization_fn=lambda p_lst: [p.serialize() for p in p_lst],
		deserialize_fn=lambda data: [Prompt.load(p) for p in data],
	)
	hash_map: dict[PromptHashStr, int]

	@classmethod
	def from_config(cls, config: PromptDatasetConfig) -> "PromptDataset":
		"""create a dataset from a config by loading the prompts from the source path"""
		if not config.source_path.exists():
			raise FileNotFoundError(
				f"Prompt dataset source path does not exist: {config.source_path = }"
			)
		# load the prompts from the source path
		prompts: list[Prompt] = []
		with open(config.source_path, "r") as f:
			for line in f:
				prompts.append(Prompt.from_dict(json.loads(line)))

		return cls.from_prompts(config=config, prompts=prompts)

	@classmethod
	def from_prompts(
		cls, config: PromptDatasetConfig, prompts: list[Prompt]
	) -> "PromptDataset":
		"""create a dataset from a config and prompts by building the hash map"""
		hash_map: dict[PromptHashStr, int] = {
			p.hash_str: i for i, p in enumerate(prompts)
		}
		return cls(config=config, prompts=prompts, hash_map=hash_map)

	def __len__(self) -> int:
		return len(self.prompts)

	def index_get(self, idx: int) -> Prompt:
		"get a prompt by it's index in the prompts list"
		return self.prompts[idx]

	def hash_str_get(self, hash_str: PromptHashStr) -> Prompt:
		"get a prompt by what the text hashes to (base64 encoded string)"
		return self.prompts[self.hash_map[hash_str]]

	def hash_int_get(self, hash_int: int) -> Prompt:
		"get a prompt by what the text hashes to (raw integer)"
		# convert to string
		hash_str: str = b64encode(hash_int)
		return self.prompts[self.hash_map[hash_str]]

	def hash_get(self, hash: int | PromptHashStr) -> Prompt:
		"get a prompt by what the text hashes to"
		if isinstance(hash, int):
			return self.hash_int_get(hash)
		elif isinstance(hash, str):
			return self.hash_str_get(hash)
		else:
			raise TypeError(f"hash must be int or str, not {type(hash) = }, {hash = }")

	def __iter__(self):
		return iter(self.prompts)


@serializable_dataclass
class AttentionPatternDataset(SerializableDataclass):
	n_ctx: int
	n_patterns: int
	patterns: AttentionPatternBatch
	metadata: list[AttentionPatternMetadata]
	raw_scores: bool

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
		(p.hash_str, model.to_tokens(p.text)) for p in prompts
	]

	# group by rounded length
	bins_by_len: defaultdict[
		int,
		tuple[
			list[PromptHashStr],  # prompt hash, can look it up in the dataset
			list[TokenSequence],  # tokenized sequence
		],
	] = defaultdict(default_factory=lambda: ([], []))

	# iterare over all tokenized prompts
	for prompt_hash, tokens in tokenized_prompts:
		# skip if too short
		if len(tokens) >= token_len_min:
			# round down to nearest bin
			desired_len: int = len(tokens) - len(tokens) % tolerance
			tokens_truncated: TokenSequence = tokens[:desired_len]

			bins_by_len[desired_len][0].append(prompt_hash)
			bins_by_len[desired_len][1].append(tokens_truncated)

	output: dict[int, tuple[list[PromptHashStr], TokenSequenceBatch]] = {
		n_ctx: (prompt_hash, torch.tensor(token_seqs_list))
		for n_ctx, (prompt_hash, token_seqs_list) in bins_by_len.items()
	}

	return output


def process_length_bin(
	model: HookedTransformer,
	n_ctx: int,
	prompt_hashes: list[PromptHashStr],
	tokens_tensor: TokenSequenceBatch,
	raw_scores: bool,
	model_name: str|None = None,
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
	names_filter: Callable[[str], bool] = ( # noqa: E731
		lambda s: s.endswith("scores")
		if raw_scores else
		lambda s: s.endswith("pattern")
	)
	key_format: str = "blocks.{layer}.attn.hook_attn_scores" if raw_scores else "blocks.{layer}.attn.hook_pattern"

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
						prompt_hash=p["hash"],
						n_ctx=n_ctx,
					)
					for p, _ in prompt_hashes[idx_start:idx_end]
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
