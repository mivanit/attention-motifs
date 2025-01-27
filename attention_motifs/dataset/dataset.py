from collections import defaultdict
from pathlib import Path
import json
from typing import Any, Iterator, Optional

import torch
from jaxtyping import Float
import tqdm
from transformer_lens import HookedTransformer

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
	JSONitem,
)
from muutils.statcounter import StatCounter
from muutils.spinner import SpinnerContext, NoOpContextManager
from muutils.dictmagic import condense_tensor_dict
from zanj import ZANJ

from attention_motifs.consts import (
	PATTERN_DTYPE,
	AttentionPatternBatch,
	PromptHashIntSequence,
	TokenSequenceBatch,
	DIVIDER_S1,
	DIVIDER_S2,
)
from attention_motifs.dataset.util import (
	AttentionPatternDataset,
	AttentionPatternMetadataArray,
	process_length_bin,
	tokenize_and_bin_prompts,
	AttentionPatternMetadata,
)

from attention_motifs.dataset.prompts import PromptDataset, PromptDatasetConfig


@serializable_dataclass
class APGenerationConfig(SerializableDataclass):
	prompts_config: PromptDatasetConfig
	model_names: list[str]
	token_len_min: int = serializable_field(default=64)
	prompt_token_len_tolerance: int = serializable_field(default=64)
	raw_scores: bool = serializable_field(default=False)


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
		prompts: PromptDataset,
		datasets: dict[int, AttentionPatternDataset],
	):
		"""
		# Parameters:
		 - `config : APGenerationConfig`
		    The config used to generate or load this dataloader
		 - `prompts : PromptDataset`
		    The dataset of prompts
		 - `datasets : list[AttentionPatternDataset]`
		    The list of attention-pattern datasets
		"""
		self.config: APGenerationConfig = config
		self.prompts: PromptDataset = prompts
		self.datasets: dict[int, AttentionPatternDataset] = datasets

	def summary(self):
		return dict(
			model_names=self.model_names,
			# dataset_metadata=self.dataset_metadata,
			dataset_shapes_summary=self.dataset_shapes_summary,
			n_datasets=self.n_datasets,
			n_total_samples=self.n_total_samples,
			# n_ctx_counts=self.n_ctx_counts,
			n_ctx_stats=self.n_ctx_stats.summary(),
			config=self.config.serialize(),
			prompts=self.prompts.summary(),
		)

	def __str__(self) -> str:
		return json.dumps(self.summary(), indent=2)

	@property
	def model_names(self) -> list[str]:
		return self.config.model_names

	@property
	def dataset_metadata(self) -> list[dict[str, JSONitem]]:
		"""Return metadata about each sub-dataset."""
		return [dict(n_ctx=d.n_ctx, n_patterns=len(d)) for d in self.datasets.values()]

	@property
	def dataset_shapes_summary(self) -> dict[int, str]:
		"""Return a summary of the shapes of the patterns in each sub-dataset."""
		return [str(tuple(d.patterns.shape)) for d in self.datasets.values()]

	@property
	def n_datasets(self) -> int:
		return len(self.datasets)

	@property
	def n_total_samples(self) -> int:
		return sum(len(d) for d in self.datasets.values())

	@property
	def n_ctx_counts(self) -> dict[int, int]:
		# how many patterns exist per sequence length
		# (each dataset has a single n_ctx)
		out: dict[int, int] = {}
		ds: AttentionPatternDataset
		for ds in self.datasets.values():
			out[ds.n_ctx] = out.get(ds.n_ctx, 0) + len(ds)
		return out

	@property
	def n_ctx_stats(self) -> StatCounter:
		return StatCounter(self.n_ctx_counts)

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
		for ds in self.datasets.values():
			i: int
			for i in range(len(ds)):
				all_items.append(ds[i])

		# chunk them by batch_size
		i: int
		for i in range(0, len(all_items), batch_size):
			chunk: list[
				tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]
			] = all_items[i : i + batch_size]

			pat_list: list[Float[torch.Tensor, "n_ctx n_ctx"]] = [x[0] for x in chunk]
			meta_list: list[AttentionPatternMetadata] = [x[1] for x in chunk]

			# stack patterns into a single tensor
			patterns_batch: Float[torch.Tensor, "batch n_ctx n_ctx"] = torch.stack(
				pat_list, dim=0
			)
			yield patterns_batch, meta_list

	def save(
		self,
		path: Path,
		z: Optional[ZANJ] = None,
		verbose: bool = False,
	) -> None:
		"""Save the dataset to ZANJ-based files.

		# Parameters:
		 - `path : Path`
		    Path to a directory where data will be stored
		 - `z : Optional[ZANJ]`
		    Instance of ZANJ to handle the saving
		"""
		# setup
		path = Path(path)
		z = z or ZANJ()

		spinner = SpinnerContext if verbose else NoOpContextManager

		# save metadata
		with spinner(message="Saving metadata"):
			obj_metadata: dict[str, Any] = dict(
				config=self.config.serialize(),
				dataset_metadata=self.dataset_metadata,
			)

			z.save(obj_metadata, path / "metadata.zanj")

		# save prompts
		with spinner(message="Saving prompts"):
			z.save(self.prompts, path / "prompts.zanj")

		# save datasets
		i: int
		dataset: AttentionPatternDataset
		for i, (n_ctx, dataset) in tqdm.tqdm(
			enumerate(self.datasets.items()),
			total=len(self.datasets),
			desc="Saving datasets",
			unit="dataset",
			disable=not verbose,
		):
			z.save(dataset, path / f"dataset_{n_ctx}.zanj")

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
		prompts: PromptDataset = z.read(path / "prompts.zanj")

		# read datasets of patterns
		dataset_meta: list[dict[str, Any]] = obj_metadata["dataset_metadata"]
		datasets: dict[int, AttentionPatternDataset] = []
		for d_m in dataset_meta:
			n_ctx: int = d_m["n_ctx"]
			ds_path: Path = path / f"dataset_n{n_ctx}.zanj"
			ds: AttentionPatternDataset = z.read(ds_path)
			datasets[n_ctx] = ds

		# create the object and return
		loader: CollectedAttentionPatternDataloader = cls(
			config=config,
			prompts=prompts,
			datasets=datasets,
		)
		return loader

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
		prompts: PromptDataset = PromptDataset.from_config(config.prompts_config)

		# collect patterns by sequence length
		data_raw_binned: defaultdict[
			int,
			tuple[list[AttentionPatternMetadata], list[AttentionPatternBatch]],
		] = defaultdict(lambda: ([], []))

		# process each model
		model_name: str
		for model_name in config.model_names:
			print(DIVIDER_S1)
			print(f"Processing model: {model_name}")
			print(DIVIDER_S2)
			# load model
			with SpinnerContext(message=f"Loading model {model_name}"):
				model: HookedTransformer = HookedTransformer.from_pretrained(model_name)
			print(f"\tloaded {model_name} with {model.cfg.n_params} parameters")

			# bin prompts by length
			# with SpinnerContext(message="Tokenizing and binning prompts"):
			bins_by_len: dict[int, tuple[PromptHashIntSequence, TokenSequenceBatch]] = (
				tokenize_and_bin_prompts(
					model=model,
					prompts=prompts,
					token_len_min=config.token_len_min,
					tolerance=config.prompt_token_len_tolerance,
				)
			)
			total_tokens: int = sum(
				len(bin_contents[1]) 
				for bin_contents in bins_by_len.values()
			)
			print(f"{total_tokens = } tokens in {len(bins_by_len) = } bins")
			with tqdm.tqdm(
				total=total_tokens,
				desc="",
				unit="Seq",
				unit_scale=True,
			) as pbar:
				for n_ctx, bin_contents in bins_by_len.items():
					pbar.set_description(
						f"{n_ctx = }",
					)
					patterns: AttentionPatternBatch
					metadata: list[AttentionPatternMetadata]
					patterns, metadata = process_length_bin(
						# model, bin_contents, model_name, config.token_len_min
						model=model,
						n_ctx=n_ctx,
						prompt_hashes=bin_contents[0],
						tokens_tensor=bin_contents[1],
						raw_scores=False,
						model_name=model_name,
						max_batch_size=None,
					)
					n_samples: int = len(metadata)
					assert len(patterns) == n_samples

					# add to binned data
					data_raw_binned[n_ctx][0].extend(metadata)
					data_raw_binned[n_ctx][1].append(patterns)

					pbar.update(len(metadata))

		# create datasets from binned data
		datasets: dict[int, AttentionPatternDataset] = {
			n_ctx: AttentionPatternDataset(
				n_ctx=n_ctx,
				n_patterns=len(metadata),
				patterns=torch.cat(patterns_list, dim=0).type(PATTERN_DTYPE),
				metadata=AttentionPatternMetadataArray.from_list(metadata),
				raw_scores=config.raw_scores,
			)
			for n_ctx, (metadata, patterns_list) in data_raw_binned.items()
		}

		# create and return the loader
		return cls(
			config=config,
			prompts=prompts,
			datasets=datasets,
		)
