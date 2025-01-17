from dataclasses import dataclass
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from jaxtyping import Float, Bool

###############################################################################
# Classes
###############################################################################


class LoadableTensor:
	"""Holds a 4D array [L, H, n, n].

	- `get()` loads the entire array (if not yet loaded) and returns it as a Torch tensor [L,H,n,n].
	- `get_idx(idx)` returns a single [n,n] slice from (i,j), and if *all* slices have been
	  accessed once, it unloads from memory.
	"""

	def __init__(self, npy_path: Path, shape: tuple[int, ...]) -> None:
		"""
		# Parameters:
		 - `npy_path : Path`
		    Path to the .npy file containing a 4D NumPy array
		 - `shape : tuple[int, ...]`
		    Expected shape, e.g. (L, H, n, n)
		"""
		self.npy_path: Path = npy_path
		self.shape: tuple[int, ...] = shape
		# We'll store the loaded data as a NumPy array (or Torch tensor).
		# The code below does a Torch conversion on every get() call, which is feasible.
		self.data: None | np.ndarray = None
		# Track which (i,j) have been accessed
		self.accessed_idxs: Bool[torch.Tensor, "L H"] = torch.zeros(
			shape[:2], dtype=torch.bool
		)

	def get(self) -> Float[torch.Tensor, "L H n n"]:
		"""
		Load data if not already loaded, and return as a Torch tensor [L,H,n,n].
		"""
		if self.data is None:
			loaded = np.load(self.npy_path)  # shape must be (L,H,n,n)
			assert loaded.shape == self.shape, (
				f"File {self.npy_path} has shape {loaded.shape}, "
				f"expected {self.shape}"
			)
			self.data = loaded  # store as a NumPy array
		return torch.tensor(self.data)  # each call returns a fresh Tensor

	def get_idx(self, idx: tuple[int, int]) -> Float[torch.Tensor, "n n"]:
		"""
		Return one slice [n,n] from the (i,j) position.
		Unload from memory if *all* slices have been accessed once.
		"""
		i, j = idx
		self.accessed_idxs[i, j] = True

		arr_4d: Float[torch.Tensor, "L H n n"] = self.get()
		sub_2d: Float[torch.Tensor, "n n"] = arr_4d[i, j]  # shape [n,n]

		# If we've accessed all (i,j), unload to free memory
		if self.accessed_idxs.all():
			sub_2d = sub_2d.clone()  # keep a copy so it remains valid
			self.data = None
			self.accessed_idxs = torch.zeros(self.shape[:2], dtype=torch.bool)
		return sub_2d


@dataclass
class SampleMetadata:
	"""Metadata for a single sub-sample, shape (n,n)."""

	n: int
	sample_id: str

	@property
	def shape(self) -> tuple[int, int]:
		return (self.n, self.n)


@dataclass
class FPSampleMetadata:
	"""
	A grid [L,H] of SampleMetadata objects, each describing a sub-sample.

	Ensures that all sub-samples have the same n.
	"""

	data: list[list[SampleMetadata]]

	def __getitem__(self, idx: tuple[int, int]) -> SampleMetadata:
		return self.data[idx[0]][idx[1]]

	@property
	def n(self) -> int:
		return self.data[0][0].n

	@property
	def shape(self) -> tuple[int, int]:
		return (self.n, self.n)

	def __post_init__(self):
		n = self.n
		shape = self.shape
		for i, row in enumerate(self.data):
			for j, meta in enumerate(row):
				assert meta.n == n, f"Sample at ({i},{j}) has n={meta.n} != {n}"
				assert (
					meta.shape == shape
				), f"Sample at ({i},{j}) has shape={meta.shape} != {shape}"


###############################################################################
# Type alias for a single sub-sample
###############################################################################
Sample = tuple[Float[torch.Tensor, "n n"], SampleMetadata]


@dataclass
class FPSample:
	"""
	Wraps a single 4D array [L,H,n,n] plus corresponding [L,H] metadata.

	`get(idx)` -> returns the sub-sample (tensor [n,n], metadata).
	"""

	data: LoadableTensor
	metadata: FPSampleMetadata

	def get(self, idx: tuple[int, int]) -> Sample:
		"""
		Return ( [n,n], SampleMetadata ) for sub-sample at (i,j).
		"""
		i, j = idx
		return (self.data.get_idx((i, j)), self.metadata[i, j])


###############################################################################
# Flattened dataset of sub-samples
###############################################################################
class FPSubsampleDataset(Dataset):
	"""
	Flatten a list of FPSamples into individual sub-samples.
	Each dataset index => one (i,j) sub-sample from one FPSample.
	"""

	def __init__(self, samples: list[FPSample]) -> None:
		"""
		# Parameters:
		 - `samples : list[FPSample]`
		    A list of FPSample objects, each shape [L,H,n,n].
		"""
		super().__init__()
		self.samples: list[FPSample] = samples

		# We'll build a map: dataset_index -> (fpsample_idx, i, j)
		self._index_map: list[tuple[int, int, int]] = []

		for fpsample_idx, fpsample in enumerate(samples):
			L, H, n, n_ = fpsample.data.shape
			assert (
				n == n_
			), f"FPSample shape mismatch: got n={n}, n_={n_} for sample_idx={fpsample_idx}"
			for i in range(L):
				for j in range(H):
					self._index_map.append((fpsample_idx, i, j))

	def __len__(self) -> int:
		"""Total count of sub-samples across all FPSamples."""
		return len(self._index_map)

	def __getitem__(self, idx: int) -> Sample:
		"""
		Return a single sub-sample: (tensor [n,n], metadata).
		"""
		fpsample_idx, i, j = self._index_map[idx]
		return self.samples[fpsample_idx].get((i, j))

	def get_subsample_shape(self, idx: int) -> tuple[int, int]:
		"""
		Return (n,n) shape for the sub-sample at dataset index=idx
		without fully loading the data if not needed.

		Because each sub-sample within the same FPSample has the same (n,n),
		we can rely on the metadata in the FPSample itself.
		"""
		fpsample_idx, i, j = self._index_map[idx]
		# All sub-samples in the same FPSample share the same shape (n,n).
		# We'll just return the shape from metadata[i,j].
		return self.samples[fpsample_idx].metadata[i, j].shape


###############################################################################
# Bucketing Sampler
###############################################################################
class ShapeBucketingSampler(Sampler[list[int]]):
	"""
	Groups dataset indices by (n,n) shape, yields mini-batches of uniform shape.
	"""

	def __init__(
		self,
		dataset: FPSubsampleDataset,
		batch_size: int,
		shuffle: bool = True,
	) -> None:
		"""
		# Parameters:
		 - `dataset : FPSubsampleDataset`
		 - `batch_size : int`
		 - `shuffle : bool`
		"""
		super().__init__(dataset)
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle

		# 1) Group indices by shape
		shape_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
		for idx in range(len(dataset)):
			shape_ij = dataset.get_subsample_shape(idx)  # e.g. (n,n)
			shape_buckets[shape_ij].append(idx)

		# 2) Break each shape's list into mini-batches
		self.batches: list[list[int]] = []
		for _, indices in shape_buckets.items():
			if self.shuffle:
				random.shuffle(indices)
			for start in range(0, len(indices), self.batch_size):
				self.batches.append(indices[start : start + self.batch_size])

		# 3) Shuffle the order of these shape-batched groups
		if self.shuffle:
			random.shuffle(self.batches)

	def __iter__(self):
		"""Yield each list of indices (mini-batch)."""
		for batch_indices in self.batches:
			yield batch_indices

	def __len__(self) -> int:
		"""Number of mini-batches total."""
		return len(self.batches)


###############################################################################
# Collate function
###############################################################################
def stack_collate_fn(
	batch: list[Sample],
) -> tuple[Float[torch.Tensor, "batch n n"], list[SampleMetadata]]:
	"""
	Collate a list of (tensor [n,n], metadata) into:
	  (stacked [B,n,n], [metadata_0, ..., metadata_{B-1}])
	"""
	# separate the data
	tensors = [sample[0] for sample in batch]
	metas = [sample[1] for sample in batch]

	# stack the [n,n] Tensors => [B,n,n]
	stacked: Float[torch.Tensor, "batch n n"] = torch.stack(tensors, dim=0)
	return (stacked, metas)
