from pathlib import Path
import json
import hashlib
from typing import Any, Iterator, Optional

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

# custom utils
from muutils.json_serialize import SerializableDataclass, serializable_dataclass
from zanj import ZANJ


@serializable_dataclass
class AttentionPatternMetadata(SerializableDataclass):
	model_name: str
	idx_layer: int
	idx_head: int
	prompt_hash: str
	n_ctx: int


@serializable_dataclass
class AttentionPatternDataset(SerializableDataclass):
	n_ctx: int
	n_patterns: int
	patterns: Float[torch.Tensor, "n_patterns n_ctx n_ctx"]
	metadata: list[AttentionPatternMetadata]

	def __len__(self) -> int:
		return self.n_patterns

	def __getitem__(
		self, idx: int
	) -> tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]:
		return self.patterns[idx], self.metadata[idx]


@serializable_dataclass
class APGenerationConfig(SerializableDataclass):
	prompts_path: Path
	model_names: list[str]
	min_length: Optional[int]
	max_length: Optional[int]
	prompt_token_len_tolerance: int = 10

	def load_text_data(self) -> list[dict]:
		"""Split prompts from `prompts_path` into more reasonable sizes (by string length, not token count).

		# Returns:
		 - `list[dict]`
		    New, processed list of prompts. Each prompt has a `"text"` key with a string value,
		    and some metadata. This is not guaranteed to be the same length as the input list!

		# Usage:
		```python
		>>> cfg = APGenerationConfig(Path("some_prompts.jsonl"), ["gpt2"], 10, 100)
		>>> processed = cfg.load_text_data()
		>>> len(processed)  # might differ from the raw file lines
		```
		"""
		data_raw: list[dict] = []
		# open the prompts file
		with open(self.prompts_path, "r") as f_in:  # type: TextIO
			line: str
			for line in f_in:
				line_str: str = line.strip()
				if line_str:
					parsed: dict = json.loads(line_str)
					data_raw.append(parsed)

		# add fname metadata
		d: dict
		for d in data_raw:
			d["source_fname"] = self.prompts_path.as_posix()

		# trim too-short samples
		if self.min_length is not None:
			data_raw = [d for d in data_raw if len(d["text"]) >= self.min_length]

		# split up too-long samples
		if self.max_length is not None:
			data_new: list[dict] = []
			for d in data_raw:
				d_text: str = d["text"]
				while len(d_text) > self.max_length:
					truncated_dict: dict = {**d, "text": d_text[: self.max_length]}
					data_new.append(truncated_dict)
					d_text = d_text[self.max_length :]
				data_new.append({**d, "text": d_text})
			data_raw = data_new

		# trim too-short samples again
		if self.min_length is not None:
			data_raw = [d for d in data_raw if len(d["text"]) >= self.min_length]

		return data_raw


class CollectedAttentionPatternDataloader:
	"""Collected dataset of `AttentionPatternDataset` objects, returning a batch of patterns and metadata.

	# Usage in a training loop:
	```python
	dl = CollectedAttentionPatternDataloader.generate(cfg)
	for patterns_batch, meta_batch in dl.batches(batch_size=32):
	    # patterns_batch: [batch_size, n_ctx, n_ctx]
	    # meta_batch: list of metadata objects
	    ...
	```
	"""

	def __init__(
		self,
		config: APGenerationConfig,
		prompts: list[dict],
		datasets: list[AttentionPatternDataset],
	):
		"""
		# Parameters:
		 - `config : APGenerationConfig`
		    The config used to generate or load this dataloader
		 - `prompts : list[dict]`
		    A list of prompt data (each has "text" and "hash", etc.)
		 - `datasets : list[AttentionPatternDataset]`
		    The list of attention-pattern datasets
		"""
		self.config: APGenerationConfig = config
		self.prompts: list[dict] = prompts
		self.datasets: list[AttentionPatternDataset] = datasets

	@property
	def model_names(self) -> list[str]:
		return self.config.model_names

	@property
	def dataset_metadata(self) -> list[dict[str, Any]]:
		"""Return metadata about each sub-dataset."""
		return [dict(n_ctx=d.n_ctx, n_patterns=len(d)) for d in self.datasets]

	@property
	def n_datasets(self) -> int:
		return len(self.datasets)

	@property
	def n_total_samples(self) -> int:
		return sum(len(d) for d in self.datasets)

	@property
	def n_ctx_counts(self) -> dict[int, int]:
		# how many patterns exist per sequence length
		# (each dataset has a single n_ctx)
		out: dict[int, int] = {}
		ds: AttentionPatternDataset
		for ds in self.datasets:
			out[ds.n_ctx] = out.get(ds.n_ctx, 0) + len(ds)
		return out

	def __iter__(
		self,
	) -> Iterator[
		tuple[Float[torch.Tensor, "batch n_ctx n_ctx"], list[AttentionPatternMetadata]]
	]:
		"""
		Yield mini-batches of (patterns, metadata).
		- patterns: [batch_size, n_ctx, n_ctx]
		- metadata: list[AttentionPatternMetadata] of length batch_size
		"""
		# flatten all items from all datasets
		all_items: list[
			tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]
		] = []
		ds: AttentionPatternDataset
		for ds in self.datasets:
			i: int
			for i in range(len(ds)):
				all_items.append(ds[i])

		# chunk them by batch_size
		i: int
		for i in range(0, len(all_items), self.batch_size):
			chunk: list[
				tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]
			] = all_items[i : i + self.batch_size]

			pat_list: list[Float[torch.Tensor, "n_ctx n_ctx"]] = [x[0] for x in chunk]
			meta_list: list[AttentionPatternMetadata] = [x[1] for x in chunk]

			# stack patterns into a single tensor
			patterns_batch: Float[torch.Tensor, "batch n_ctx n_ctx"] = torch.stack(
				pat_list, dim=0
			)
			yield patterns_batch, meta_list

	def save(self, path: Path, z: Optional[ZANJ] = None) -> None:
		"""Save the dataset to ZANJ-based files.

		# Parameters:
		 - `path : Path`
		    Path to a directory where data will be stored
		 - `z : Optional[ZANJ]`
		    Instance of ZANJ to handle the saving
		"""
		z = z or ZANJ()

		obj_metadata: dict[str, Any] = dict(
			config=self.config.serialize(),
			dataset_metadata=self.dataset_metadata,
		)

		z.save(obj_metadata, path / "metadata.zanj")

		# save prompts
		with z.open(path / "prompts.jsonl", "w") as f_out:  # type: TextIO
			prompt: dict
			for prompt in self.prompts:
				json.dump(prompt, f_out)
				f_out.write("\n")

		i: int
		dataset: AttentionPatternDataset
		for i, dataset in enumerate(self.datasets):
			z.save(dataset, path / f"dataset_{i}.zanj")

	@classmethod
	def read(
		cls, path: Path, z: Optional[ZANJ] = None,
	) -> "CollectedAttentionPatternDataloader":
		"""Read the dataset from a directory.

		# Parameters:
		 - `path : Path`
		    The path to the directory containing:
		    - metadata.zanj
		    - prompts.jsonl
		    - dataset_0.zanj, dataset_1.zanj, ...
		 - `z : Optional[ZANJ]`
		    Instance of ZANJ to handle loading

		# Returns:
		 - `CollectedAttentionPatternDataloader`
		"""
		z = z or ZANJ()

		obj_metadata: dict[str, Any] = z.read(path / "metadata.zanj")
		config: APGenerationConfig = APGenerationConfig.load(obj_metadata["config"])

		# read prompts
		prompts: list[dict] = []
		with open(path / "prompts.jsonl", "r") as f_in:
			line: str
			for line in f_in:
				line_str: str = line.strip()
				if line_str:
					parsed_line: dict = json.loads(line_str)
					prompts.append(parsed_line)

		dataset_meta: list[dict[str, Any]] = obj_metadata["dataset_metadata"]
		datasets: list[AttentionPatternDataset] = []
		i: int
		for i in range(len(dataset_meta)):
			ds_path: Path = path / f"dataset_{i}.zanj"
			ds: AttentionPatternDataset = z.read(ds_path)
			datasets.append(ds)

		loader: CollectedAttentionPatternDataloader = cls(
			config=config,
			prompts=prompts,
			datasets=datasets,
		)
		return loader

	@classmethod
	def generate(
		cls, config: APGenerationConfig, z: Optional[ZANJ] = None,
	) -> "CollectedAttentionPatternDataloader":
		"""Generate attention patterns for each prompt, for each model in config,
		without adding any padding tokens. Instead, within each bin:

		 - We gather all prompts whose token-length L satisfies abs(L - bin_center) <= tolerance
		 - We compute bin_len = the min token-length among those prompts.
		 - We skip any that are shorter than bin_len (if that even occurs).
		 - We truncate any that are longer than bin_len.

		# Parameters:
		 - `config : APGenerationConfig`
		    The config specifying model names, path to prompts, etc.
		 - `z : Optional[ZANJ]`
		    Not used for generation here (unless you want to do something custom).

		# Returns:
		 - `CollectedAttentionPatternDataloader`
		"""
		_z: ZANJ = z or ZANJ()
		prompts_raw: list[dict] = config.load_text_data()

		pr: dict
		for pr in prompts_raw:
			h: str = hashlib.md5(pr["text"].encode("utf-8")).hexdigest()
			pr["hash"] = h

		datasets: list[AttentionPatternDataset] = []

		model_name: str
		for model_name in config.model_names:
			model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
			bins_by_len: dict[int, list[int]] = {}

			idx: int
			for idx, pr_data in enumerate(prompts_raw):
				tokens_no_pad: Float[torch.Tensor, "1 length"] = model.to_tokens(
					pr_data["text"], prepend_bos=False, pad_to_longest=False
				)
				token_len: int = tokens_no_pad.shape[1]

				found_bin: Optional[int] = None
				b: int
				for b in bins_by_len:
					if abs(b - token_len) <= config.prompt_token_len_tolerance:
						found_bin = b
						break

				if found_bin is None:
					# create a new bin with "representative" length token_len
					bins_by_len[token_len] = [idx]
				else:
					bins_by_len[found_bin].append(idx)

			bin_center_len: int
			idx_list: list[int]
			for bin_center_len, idx_list in bins_by_len.items():
				lengths_in_bin: list[int] = []
				i_prompt: int
				for i_prompt in idx_list:
					t_check: Float[torch.Tensor, "1 length"] = model.to_tokens(
						prompts_raw[i_prompt]["text"],
						prepend_bos=False,
						pad_to_longest=False,
					)
					lengths_in_bin.append(t_check.shape[1])

				bin_len: int = min(lengths_in_bin)

				all_tokens_list: list[Float[torch.Tensor, "1 length"]] = []
				valid_indices: list[int] = []

				for i_prompt in idx_list:
					t: Float[torch.Tensor, "1 length"] = model.to_tokens(
						prompts_raw[i_prompt]["text"],
						prepend_bos=False,
						pad_to_longest=False,
					)
					if t.shape[1] < bin_len:
						# skip (can't pad, user wants no padding at all)
						continue
					truncated: Float[torch.Tensor, "1 bin_len"] = t[:, :bin_len]
					all_tokens_list.append(truncated)
					valid_indices.append(i_prompt)

				if len(all_tokens_list) == 0:
					# no data left => skip
					continue

				tokens: Float[torch.Tensor, "batch_size_texts bin_len"] = torch.cat(
					all_tokens_list, dim=0
				)
				with torch.no_grad():
					_, cache = model.run_with_cache(tokens)

				batch_size_texts: int
				seq_len: int
				batch_size_texts, seq_len = tokens.shape
				n_layers: int = model.cfg.n_layers
				n_heads: int = model.cfg.n_heads

				list_patterns: list[Float[torch.Tensor, "n_ctx n_ctx"]] = []
				list_metadata: list[AttentionPatternMetadata] = []

				layer_idx: int
				for layer_idx in range(n_layers):
					layer_pattern: Float[
						torch.Tensor, "batch_size_texts n_heads seq_len seq_len"
					] = cache["pattern", layer_idx]
					head_idx: int
					for head_idx in range(n_heads):
						head_pattern: Float[
							torch.Tensor, "batch_size_texts seq_len seq_len"
						] = layer_pattern[:, head_idx, :, :]
						i_in_batch: int
						for i_in_batch in range(batch_size_texts):
							single_pattern: Float[torch.Tensor, "seq_len seq_len"] = (
								head_pattern[i_in_batch, :, :]
							)
							pm: AttentionPatternMetadata = AttentionPatternMetadata(
								model_name=model_name,
								idx_layer=layer_idx,
								idx_head=head_idx,
								prompt_hash=prompts_raw[valid_indices[i_in_batch]][
									"hash"
								],
								n_ctx=seq_len,
							)
							list_patterns.append(single_pattern)
							list_metadata.append(pm)

				patterns_tensor: Float[torch.Tensor, "n_patterns seq_len seq_len"] = (
					torch.stack(list_patterns, dim=0)
				)
				ds: AttentionPatternDataset = AttentionPatternDataset(
					n_ctx=seq_len,
					n_patterns=patterns_tensor.shape[0],
					patterns=patterns_tensor,
					metadata=list_metadata,
				)
				datasets.append(ds)

		loader: CollectedAttentionPatternDataloader = cls(
			config=config,
			prompts=prompts_raw,
			datasets=datasets,
		)
		return loader
