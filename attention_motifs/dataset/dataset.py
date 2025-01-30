from collections import defaultdict
from pathlib import Path
import json
from typing import Any, Iterator, Optional
import random

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
from muutils.misc import shorten_numerical_to_str
from zanj import ZANJ

from attention_motifs.consts import (
	PATTERN_DTYPE,
	AttentionPatternBatch,
	PromptHashIntSequence,
	TokenSequenceBatch,
	DIVIDER_S1,
	DIVIDER_S2,
	tensor_batches_indexed,
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

	def summary(self) -> dict[str, JSONitem]:
		return dict(
			prompts_config=self.prompts_config.summary(),
			model_names=self.model_names,
			token_len_min=self.token_len_min,
			prompt_token_len_tolerance=self.prompt_token_len_tolerance,
			raw_scores=self.raw_scores,
		)


class DatasetMock:
	def __init__(self, n_samples: int) -> None:
		self.n_samples: int = n_samples

	def __len__(self) -> int:
		return self.n_samples


class DataloaderMock:
	def __init__(
		self,
		iter_func,
		batch_size: int,
		shuffle: bool,
		n_batches: int,
		n_samples: int,
	) -> None:
		self.iter_func = iter_func
		self.batch_size: int = batch_size
		self.shuffle: bool = shuffle
		self.n_samples: int = n_samples
		self.n_batches: int = n_batches
		self.dataset: DatasetMock = DatasetMock(n_samples)

	def __len__(self) -> int:
		return self.n_batches

	def __iter__(self):
		for x in self.iter_func():
			yield x


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
		assert isinstance(config, APGenerationConfig)
		self.config: APGenerationConfig = config
		assert isinstance(prompts, PromptDataset)
		self.prompts: PromptDataset = prompts
		assert isinstance(datasets, dict)
		self.datasets: dict[int, AttentionPatternDataset] = datasets

	def summary(self):
		n_ctx_stats: dict
		try:
			n_ctx_stats = self.n_ctx_stats.summary()
		except Exception as e:
			n_ctx_stats = dict(stat_summary_failed=str(e))

		return dict(
			model_names=self.model_names,
			# dataset_metadata=self.dataset_metadata,
			dataset_shapes_summary=self.dataset_shapes_summary,
			n_datasets=self.n_datasets,
			n_total_samples=self.n_total_samples,
			# n_ctx_counts=self.n_ctx_counts,
			n_ctx_stats=n_ctx_stats,
			config=self.config.serialize(),
			prompts=self.prompts.summary(),
		)

	def summary_short(self):
		return dict(
			n_total_samples=self.n_total_samples,
			n_prompts=len(self.prompts),
			n_datasets=self.n_datasets,
			config=self.config.summary(),
			dataset_shapes_summary=self.dataset_shapes_summary,
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
		self,
		batch_size: int,
		shuffle: bool = False,
		max_batches: int|None = None,
	) -> Iterator[
		tuple[Float[torch.Tensor, "batch n_ctx n_ctx"], list[AttentionPatternMetadata]]
	]:
		"""Yield mini-batches of (patterns, metadata).

		This function serves up batches from multiple datasets stored in `self.datasets`.
		Each dataset has its own `patterns` and `metadata`. If `shuffle` is True, we:
		1. Shuffle each dataset independently.
		2. Randomly pick from any dataset that is not yet exhausted to yield the next batch.

		# Parameters:
		- `batch_size : int`
			Size of each mini-batch (must be >= 1)
		- `shuffle : bool`
			If True, shuffle within and across datasets

		# Returns:
		- `Iterator[ tuple[Float[torch.Tensor, "batch n_ctx n_ctx"], list[AttentionPatternMetadata]] ]`
		Yields `(batch, metadata)` pairs where:
		- `batch` has shape `[batch_size, n_ctx, n_ctx]`
		- `metadata` is a list of `AttentionPatternMetadata` objects of length `batch_size`

		# Modifies:
		- `ds : self.datasets[...]`
		Shuffles each dataset in-place if `shuffle` is True

		# Usage:
		```python
		>>> for batch, meta in self.batches(batch_size=32, shuffle=True):
		...     pass
		```

		# Raises:
		- `AssertionError`
		If `batch_size <= 0`
		"""
		assert batch_size >= 1, "batch_size must be positive"


		batches_count: int = 0
		if shuffle:
			# Shuffle each dataset
			for ds in self.datasets.values():
				ds.shuffle()

			# Build an iterator for each dataset
			iters: dict[
				int, Iterator[tuple[int, int, Float[torch.Tensor, "batch n_ctx n_ctx"]]]
			] = {}
			for n_ctx, ds in self.datasets.items():
				iters[n_ctx] = iter(
					tensor_batches_indexed(ds.patterns, batch_size=batch_size)
				)

			# Randomly pick from any dataset that isn't exhausted
			while iters and (max_batches is None or batches_count < max_batches):
				n_ctx: int = random.choice(list(iters.keys()))
				dataset_iter: Iterator[
					tuple[int, int, Float[torch.Tensor, "batch n_ctx n_ctx"]]
				] = iters[n_ctx]
				try:
					idx_start: int
					idx_end: int
					batch: Float[torch.Tensor, "batch n_ctx n_ctx"]
					idx_start, idx_end, batch = next(dataset_iter)
				except StopIteration:
					del iters[n_ctx]
					continue

				ds = self.datasets[n_ctx]
				metadata: list[AttentionPatternMetadata] = ds.metadata[
					idx_start:idx_end
				]
				yield batch, metadata
				batches_count += 1

		else:
			# Non-shuffled: yield batches from each dataset in sequence
			for n_ctx, ds in self.datasets.items():
				for idx_start, idx_end, batch in tensor_batches_indexed(
					ds.patterns, batch_size=batch_size
				):
					metadata: list[AttentionPatternMetadata] = ds.metadata[
						idx_start:idx_end
					]
					yield batch, metadata
					batches_count += 1
					if max_batches is not None and batches_count >= max_batches:
						raise StopIteration()

	def dataloader(
		self,
		batch_size: int,
		shuffle: bool,
		max_batches: int|None = None,
	) -> DataloaderMock:
		"""Return a dataloader that yields batches of patterns and metadata."""
		n_batches: int = max_batches or self.n_total_samples // batch_size
		return DataloaderMock(
			iter_func=lambda: self.batches(batch_size=batch_size, shuffle=shuffle, max_batches=max_batches),
			batch_size=batch_size,
			shuffle=shuffle,
			n_batches=n_batches,
			n_samples=n_batches * batch_size,
		)

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
				summary=self.summary(),
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
			# TODO: switch to `_n{n_ctx}` for the dataset name
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
		path = Path(path)
		z = z or ZANJ()

		# save the metadata
		obj_metadata: dict[str, Any] = z.read(path / "metadata.zanj")
		config: APGenerationConfig = APGenerationConfig.load(obj_metadata["config"])

		# read prompts
		prompts: PromptDataset = z.read(path / "prompts.zanj")

		# read datasets of patterns
		dataset_meta: list[dict[str, Any]] = obj_metadata["dataset_metadata"]
		datasets: dict[int, AttentionPatternDataset] = dict()
		for d_m in dataset_meta:
			n_ctx: int = d_m["n_ctx"]
			# TODO: switch to `_n{n_ctx}` for the dataset name
			ds_path: Path = path / f"dataset_{n_ctx}.zanj"
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
		model_device: torch.device = torch.device("cuda")
		if torch.cuda.is_available()
		else torch.device("cpu"),
		storage_device: torch.device = torch.device("cpu"),
		max_batch_size: Optional[int] = None,
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
			print(f"# Processing model: {model_name}")
			print(DIVIDER_S2)
			# load model
			with SpinnerContext(message=f"Loading model {model_name}"):
				model: HookedTransformer = HookedTransformer.from_pretrained(
					model_name,
					device=model_device,
				)
				model.eval()
			print(
				f"#\tloaded {model_name} with {model.cfg.n_params} ({shorten_numerical_to_str(model.cfg.n_params)}) parameters"
			)
			model_devices: set[torch.device] = {p.device for p in model.parameters()}
			print(f"#\tmodel devices: {model_devices}")

			with SpinnerContext(message="tokenizing and binning prompts"):
				# bin prompts by length
				# with SpinnerContext(message="Tokenizing and binning prompts"):
				bins_by_len: dict[
					int, tuple[PromptHashIntSequence, TokenSequenceBatch]
				] = tokenize_and_bin_prompts(
					model=model,
					prompts=prompts,
					token_len_min=config.token_len_min,
					tolerance=config.prompt_token_len_tolerance,
					storage_device=storage_device,
				)
				total_sequences: int = sum(
					len(bin_contents[1]) for bin_contents in bins_by_len.values()
				)

			bin_shapes: str = ", ".join(
				[str(tuple(x[1].shape)) for x in bins_by_len.values()]
			)
			print(f"#\tshapes of each bin contents (n_seqs, n_ctx): [ {bin_shapes} ]")

			print("# getting attention patterns:")
			with tqdm.tqdm(
				total=total_sequences,
				desc="",
				unit="Seq",
				unit_scale=True,
			) as pbar:
				for n_ctx, bin_contents in bins_by_len.items():
					pbar.set_description(
						f"{n_ctx = }",
					)
					n_sequences: int = len(bin_contents[1])
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
						max_batch_size=max_batch_size,
						model_device=model_device,
						storage_device=storage_device,
					)
					n_patterns: int = len(metadata)
					assert len(patterns) == n_patterns

					# add to binned data
					data_raw_binned[n_ctx][0].extend(metadata)
					data_raw_binned[n_ctx][1].append(patterns.to(storage_device))

					pbar.update(n_sequences)

			del model

		print(DIVIDER_S1)

		# create datasets from binned data
		# TODO: save them incrementally. not enough dedidated wam
		with SpinnerContext(message="assembling datasets"):
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
		output: CollectedAttentionPatternDataloader = cls(
			config=config,
			prompts=prompts,
			datasets=datasets,
		)

		print("# done generating datasets! summary:")
		print(DIVIDER_S2)
		print(json.dumps(output.summary_short(), indent=2))
		print(DIVIDER_S2)

		return output
