# file: my_attention_patterns.py
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
		"""split prompts from `prompts_path` up into more reasonable sizes (by string length, not token count)

		# Returns:
		 - `list[dict]`
		    new, processed list of prompts. Each prompt has a `"text"` key with a string value,
		    and some metadata. This is not guaranteed to be the same length as the input list!

		# Usage:
		```python
		>>> cfg = APGenerationConfig(Path("some_prompts.jsonl"), ["gpt2"], 10, 100)
		>>> processed = cfg.load_text_data()
		>>> len(processed)  # might differ from the raw file lines
		```
		"""
		# read raw data
		with open(self.prompts_path, "r") as f:
			data_raw: list[dict] = []
			for line in f:
				line_str = line.strip()
				if line_str:
					data_raw.append(json.loads(line_str))

		# add fname metadata
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
					data_new.append(
						{
							**d,
							"text": d_text[: self.max_length],
						}
					)
					d_text = d_text[self.max_length :]
				data_new.append(
					{
						**d,
						"text": d_text,
					}
				)
			data_raw = data_new

		# trim too-short samples again
		if self.min_length is not None:
			data_raw = [d for d in data_raw if len(d["text"]) >= self.min_length]

		return data_raw


class CollectedAttentionPatternDataloader:
	"""collected dataset of `AttentionPatternDataset` objects. returns a batch of patterns and metadata.

	# Usage in a training loop:
	```python
	dl = CollectedAttentionPatternDataloader.generate(cfg)
	for patterns_batch, meta_batch in dl:
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
		batch_size: int = 1,
	):
		"""
		# Parameters:
		 - `config : APGenerationConfig`
		    The config used to generate or load this dataloader
		 - `prompts : list[dict]`
		    A list of prompt data (each has "text" and "hash", etc.)
		 - `datasets : list[AttentionPatternDataset]`
		    The list of attention-pattern datasets
		 - `batch_size : int`
		    The number of items to yield per iteration
		"""
		self.config = config
		self.prompts = prompts
		self.datasets = datasets
		self.batch_size = batch_size

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
		for d in self.datasets:
			out[d.n_ctx] = out.get(d.n_ctx, 0) + len(d)
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
		for ds in self.datasets:
			for i in range(len(ds)):
				all_items.append(ds[i])

		# chunk them by batch_size
		for i in range(0, len(all_items), self.batch_size):
			chunk = all_items[i : i + self.batch_size]
			pat_list = [x[0] for x in chunk]
			meta_list = [x[1] for x in chunk]
			# stack patterns into a single tensor
			patterns_batch = torch.stack(pat_list, dim=0)
			yield patterns_batch, meta_list

	def save(self, path: Path, z: Optional[ZANJ] = None) -> None:
		"""save the dataset to a zanj file

		# Parameters:
		 - `path : Path`
		    path to a directory where data will be stored
		 - `z : Optional[ZANJ]`
		    instance of ZANJ to handle the saving
		"""
		z = z or ZANJ()

		# save metadata
		obj_metadata: dict[str, Any] = dict(
			config=self.config.serialize(),
			dataset_metadata=self.dataset_metadata,
		)

		z.save(
			obj_metadata,
			path / "metadata.zanj",
		)

		# save prompts
		with z.open(path / "prompts.jsonl", "w") as f:
			for prompt in self.prompts:
				json.dump(prompt, f)
				f.write("\n")

		# save datasets
		for i, dataset in enumerate(self.datasets):
			z.save(
				dataset,
				path / f"dataset_{i}.zanj",
			)

	@classmethod
	def read(
		cls, path: Path, z: Optional[ZANJ] = None, batch_size: int = 1
	) -> "CollectedAttentionPatternDataloader":
		"""read the dataset from a directory

		# Parameters:
		 - `path : Path`
		    the path to the directory containing:
		    - metadata.zanj
		    - prompts.jsonl
		    - dataset_0.zanj, dataset_1.zanj, ...
		 - `z : Optional[ZANJ]`
		    instance of ZANJ to handle loading
		 - `batch_size : int`
		    how large a mini-batch you want for iteration

		# Returns:
		 - `CollectedAttentionPatternDataloader`
		"""
		z = z or ZANJ()

		# read metadata
		obj_metadata: dict = z.read(path / "metadata.zanj")
		config = APGenerationConfig.deserialize(obj_metadata["config"])

		# read prompts
		prompts: list[dict] = []
		with z.open(path / "prompts.jsonl", "r") as f:
			for line in f:
				line_str = line.strip()
				if line_str:
					prompts.append(json.loads(line_str))

		# read datasets
		dataset_meta = obj_metadata["dataset_metadata"]
		datasets: list[AttentionPatternDataset] = []
		for i in range(len(dataset_meta)):
			ds = z.read(path / f"dataset_{i}.zanj")
			datasets.append(ds)

		return cls(
			config=config,
			prompts=prompts,
			datasets=datasets,
			batch_size=batch_size,
		)

	@classmethod
	def generate(
		cls, config: APGenerationConfig, z: Optional[ZANJ] = None, batch_size: int = 1
	) -> "CollectedAttentionPatternDataloader":
		"""generate attention patterns for each prompt, for each model in config

		# Parameters:
		 - `config : APGenerationConfig`
		    The config specifying model names, path to prompts, etc.
		 - `z : Optional[ZANJ]`
		    Not used for generation here (unless you want to do something custom).
		 - `batch_size : int`
		    how large a mini-batch you want for iteration

		# Returns:
		 - `CollectedAttentionPatternDataloader`
		"""
		_z: ZANJ = z or ZANJ()
		prompts_raw: list[dict] = config.load_text_data()

		# ensure each prompt has a stable hash
		for pr in prompts_raw:
			h = hashlib.md5(pr["text"].encode("utf-8")).hexdigest()
			pr["hash"] = h

		datasets: list[AttentionPatternDataset] = []

		# for each model
		for model_name in config.model_names:
			model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
			bins_by_len: dict[int, list[int]] = {}

			# group prompts by approximate length (within tolerance)
			for idx, pr in enumerate(prompts_raw):
				tokens = model.to_tokens(
					pr["text"], prepend_bos=False, pad_to_longest=False
				)
				token_len: int = tokens.shape[1]

				found_bin: Optional[int] = None
				for b in bins_by_len:
					if abs(b - token_len) <= config.prompt_token_len_tolerance:
						found_bin = b
						break

				if found_bin is None:
					bins_by_len[token_len] = [idx]
				else:
					bins_by_len[found_bin].append(idx)

			# for each bin of similar lengths, generate patterns
			for bin_len, idx_list in bins_by_len.items():
				texts: list[str] = [prompts_raw[i]["text"] for i in idx_list]
				# pad to longest automatically
				tokens = model.to_tokens(texts, prepend_bos=False, pad_to_longest=True)
				with torch.no_grad():
					_, cache = model.run_with_cache(tokens)

				batch_size_texts, seq_len = tokens.shape
				n_layers = model.cfg.n_layers
				n_heads = model.cfg.n_heads

				list_patterns: list[Float[torch.Tensor, "n_ctx n_ctx"]] = []
				list_metadata: list[AttentionPatternMetadata] = []

				# gather all layer-head patterns
				for layer_idx in range(n_layers):
					# shape = [batch_size, n_heads, seq_len, seq_len]
					layer_pattern: Float[
						torch.Tensor, "batch n_heads seq_len seq_len"
					] = cache["pattern", layer_idx]
					for head_idx in range(n_heads):
						# [batch_size, seq_len, seq_len]
						head_pattern: Float[torch.Tensor, "batch seq_len seq_len"] = (
							layer_pattern[:, head_idx, :, :]
						)
						for i_in_batch in range(batch_size_texts):
							single_pattern: Float[torch.Tensor, "seq_len seq_len"] = (
								head_pattern[i_in_batch, :, :]
							)
							pm = AttentionPatternMetadata(
								model_name=model_name,
								idx_layer=layer_idx,
								idx_head=head_idx,
								prompt_hash=prompts_raw[idx_list[i_in_batch]]["hash"],
								n_ctx=seq_len,
							)
							list_patterns.append(single_pattern)
							list_metadata.append(pm)

				patterns_tensor: Float[torch.Tensor, "n_patterns seq_len seq_len"] = (
					torch.stack(list_patterns, dim=0)
				)
				ds = AttentionPatternDataset(
					n_ctx=seq_len,
					n_patterns=patterns_tensor.shape[0],
					patterns=patterns_tensor,
					metadata=list_metadata,
				)
				datasets.append(ds)

		# build the loader
		loader = cls(
			config=config,
			prompts=prompts_raw,
			datasets=datasets,
			batch_size=batch_size,
		)
		return loader
