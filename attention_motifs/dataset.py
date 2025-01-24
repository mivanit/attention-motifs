from collections import defaultdict
from pathlib import Path
import json
import hashlib
from typing import Any, Iterator, Optional

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from zanj import ZANJ

from attention_motifs.consts import (
	AttentionPatternBatch,
	PromptDatasetConfig,
	PromptDataset,
	AttentionPatternDataset,
)
from attention_motifs.dataset_util import (
	tokenize_and_bin_prompts,
	AttentionPatternMetadata,
)


@serializable_dataclass
class APGenerationConfig(SerializableDataclass):
	prompts_config: PromptDatasetConfig
	model_names: list[str]
	token_len_min: int = serializable_field(default=5)
	prompt_token_len_tolerance: int = serializable_field(default=5)

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
		if self.chars_len_min is not None:
			data_raw = [d for d in data_raw if len(d["text"]) >= self.chars_len_min]

		# split up too-long samples
		if self.char_len_max is not None:
			data_new: list[dict] = []
			for d in data_raw:
				d_text: str = d["text"]
				while len(d_text) > self.char_len_max:
					truncated_dict: dict = {**d, "text": d_text[: self.char_len_max]}
					data_new.append(truncated_dict)
					d_text = d_text[self.char_len_max :]
				data_new.append({**d, "text": d_text})
			data_raw = data_new

		# trim too-short samples again
		if self.chars_len_min is not None:
			data_raw = [d for d in data_raw if len(d["text"]) >= self.chars_len_min]

		# add hash metadata
		pr: dict
		for pr in data_raw:
			h: str = hashlib.md5(pr["text"].encode("utf-8")).hexdigest()
			pr["hash"] = h

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

	def batches(
		self, batch_size: int
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
		cls,
		path: Path,
		z: Optional[ZANJ] = None,
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

		# save the metadata
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

		# read datasets of patterns
		dataset_meta: list[dict[str, Any]] = obj_metadata["dataset_metadata"]
		datasets: list[AttentionPatternDataset] = []
		i: int
		for i in range(len(dataset_meta)):
			ds_path: Path = path / f"dataset_{i}.zanj"
			ds: AttentionPatternDataset = z.read(ds_path)
			datasets.append(ds)

		# create the object and return
		loader: CollectedAttentionPatternDataloader = cls(
			config=config,
			prompts=prompts,
			datasets=datasets,
		)
		return loader

	@classmethod
	def _create_dataset(
		cls,
		n_ctx: int,
		patterns_and_meta: list[
			tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]
		],
	) -> AttentionPatternDataset:
		"""Create a dataset from patterns of the same sequence length.

		# Parameters:
		- `n_ctx : int`
			Sequence length for this dataset
		- `patterns_and_meta : list[tuple[tensor, metadata]]`
			List of (pattern, metadata) pairs to include

		# Returns:
		- `AttentionPatternDataset`
			Dataset containing all patterns and metadata
		"""
		# separate patterns and metadata
		patterns_list: list[Float[torch.Tensor, "n_ctx n_ctx"]] = [
			p for p, _ in patterns_and_meta
		]
		meta_list: list[AttentionPatternMetadata] = [m for _, m in patterns_and_meta]

		# stack patterns
		patterns_tensor: AttentionPatternBatch = torch.stack(patterns_list, dim=0)

		# create and return dataset
		return AttentionPatternDataset(
			n_ctx=n_ctx,
			n_patterns=len(patterns_list),
			patterns=patterns_tensor,
			metadata=meta_list,
		)

	@classmethod
	def generate(
		cls,
		config: APGenerationConfig,
		z: Optional[ZANJ] = None,
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
		# set up zanj
		_z: ZANJ = z or ZANJ()

		# load the text data
		prompts_raw: list[dict] = config.load_text_data()

		# collect patterns by sequence length
		data_raw_binned: defaultdict[
			int,
			list[tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]],
		] = defaultdict(list)

		# process each model
		model_name: str
		for model_name in config.model_names:
			# load model
			model: HookedTransformer = HookedTransformer.from_pretrained(model_name)

			# bin prompts by length
			bins_by_len: dict[int, list[tuple[dict, list[int]]]] = (
				tokenize_and_bin_prompts(
					model,
					prompts_raw,
					config.token_len_min,
					config.prompt_token_len_tolerance,
				)
			)

			# process each bin
			bin_center: int
			bin_contents: list[tuple[dict, list[int]]]
			for bin_center, bin_contents in bins_by_len.items():
				patterns_and_meta: list[
					tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]
				] = cls._process_length_bin(
					model, bin_contents, model_name, config.token_len_min
				)

				# get actual sequence length for this bin
				n_ctx: int = patterns_and_meta[0][0].shape[0]

				# add to binned data
				data_raw_binned[n_ctx].extend(patterns_and_meta)

		# create datasets from binned data
		datasets: list[AttentionPatternDataset] = [
			cls._create_dataset(n_ctx, patterns_and_meta)
			for n_ctx, patterns_and_meta in data_raw_binned.items()
		]

		# create and return the loader
		return cls(
			config=config,
			prompts=prompts_raw,
			datasets=datasets,
		)
