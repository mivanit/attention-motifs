from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Callable, Iterable, Literal, Optional, TypeVar, cast

import numpy as np
from jaxtyping import Float, Int64
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, ActivationCache
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# custom utils
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from zanj import ZANJ

# pattern_lens

# attention_motifs




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
	
	def __getitem__(self, idx: int) -> tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]:
		return self.patterns[idx], self.metadata[idx]


@serializable_dataclass
class APGenerationConfig(SerializableDataclass):
	prompts_path: Path
	model_names: list[str]
	# min and max length in chars
	min_length: int|None
	max_length: int|None
	# tolerance in token length
	prompt_token_len_tolerance: int = 10

	def load_text_data(self) -> list[dict]:
		"""split prompts from `prompts_path` up into more reasonable sizes (by string length, not token count)

		# Returns:
		- `list[dict]`
		new, processed list of prompts. Each prompt has a "text" key with a string value, and some metadata. this is not guaranteed to be the same length as the input list!

		modified from https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/prompts.py
		"""
		# read raw data
		with open(self.prompts_path, "r") as f:
			data_raw: list[dict] = [json.loads(d) for d in f.readlines()]

		# add fname metadata
		for d in data_raw:
			d["source_fname"] = self.prompts_path.as_posix()

		# trim too-short samples
		if self.min_chars is not None:
			data_raw = list(
				filter(
					lambda x: len(x["text"]) >= self.min_chars,
					data_raw,
				)
			)

		# split up too-long samples
		if self.max_chars is not None:
			data_new: list[dict] = []
			for d in data_raw:
				d_text: str = d["text"]
				while len(d_text) > self.max_chars:
					data_new.append(
						{
							**d,
							"text": d_text[:self.max_chars],
						}
					)
					d_text = d_text[self.max_chars:]
				data_new.append(
					{
						**d,
						"text": d_text,
					}
				)
			data_raw = data_new

		# trim too-short samples again
		if self.min_chars is not None:
			data_raw = list(
				filter(
					lambda x: len(x["text"]) >= self.min_chars,
					data_raw,
				)
			)
		
		return data_raw

class CollectedAttentionPatternDataloader:
	"""collected dataset of `AttentionPatternDataset` objects. returns a batch of patterns, each with the same size"""
	
	config: APGenerationConfig
	prompts: list[dict] # has keys "text" and "hash"
	datasets: list[AttentionPatternDataset]

	@property
	def model_names(self) -> list[str]:
		return self.config.model_names

	@property
	def dataset_metadata(self) -> list[dict[int, str]]:
		return [
			dict(n_ctx = d.n_ctx, n_patterns = len(d))
			for d in self.datasets
		]
	
	@property
	def n_datasets(self) -> int:
		return len(self.datasets)
	
	@property
	def n_total_samples(self) -> int:
		return sum(len(d) for d in self.datasets)
	
	@property
	def n_ctx_counts(self) -> dict[int, int]:
		return {d.n_ctx: len(d) for d in self.datasets}

	def save(self, path: Path, z: ZANJ|None = None) -> None:
		"""save the dataset to a zanj file"""
		z = z or ZANJ()
		
		# save metadata
		obj_metadata: dict = dict(
			config = self.config.serialize(),
			dataset_metadata = self.dataset_metadata,			
		)

		z.save(
			obj_metadata,
			path / "metadata.zanj"
		)

		# save prompts
		with z.open(path / "prompts.jsonl", "w") as f:
			for prompt in self.prompts:
				json.dump(prompt, f)
				f.write("\n")
		
		# save prompts
		for i, dataset in enumerate(self.datasets):
			z.save(
				dataset,
				path / f"dataset_{i}.zanj"
			)

	@classmethod
	def read(cls, path: Path, z: ZANJ|None = None) -> "CollectedAttentionPatternDataloader":
		"read the dataset from a directory"
		z = z or ZANJ()
		raise NotImplementedError("TODO")

		# assert we have the right files

		# read metadata with z.read(path / "metadata.zanj")

		# read prompts

		# read datasets, each with `z.read(path / f"dataset_{i}.zanj")`


	@classmethod
	def generate(cls, config: APGenerationConfig, z: ZANJ|None = None) -> "CollectedAttentionPatternDataloader":
		# load data

		# for each model:

		# 	tokenize prompts
		# 	cut down into bins by prompt_token_len_tolerance

		# put each bin into a dataset

		# return the collected dataset

