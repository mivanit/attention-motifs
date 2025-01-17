from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from jaxtyping import Float, Bool


class LoadableTensor:
    def __init__(self, npy_path: Path, shape: tuple[int, ...]) -> None:
        self.npy_path: Path = npy_path
        self.shape: tuple[int, ...] = shape
        self.data: None | Float[torch.Tensor, "L H n n"] = None
        self.accessed_idxs: Bool[torch.Tensor, "L H"] = torch.zeros(shape[:2], dtype=torch.bool)

    def get(self) -> Float[torch.Tensor, "L H n n"]:
        # load data if not already loaded
        if self.data is None:
            self.data = np.load(self.npy_path)
            assert self.data.shape == self.shape

        # return data
        return torch.tensor(self.data)
    
    def get_idx(self, idx: tuple[int,int]) -> Float[torch.Tensor, "n n"]:
        # get the index, mark it as accessed
        self.accessed_idxs[idx] = True
        idx_data: Float[torch.Tensor, "n n"] = self.get()[idx]
        # if all the indices have been accessed, unload the data
        if self.accessed_idxs.all():
            idx_data = idx_data.clone()
            self.data = None
            self.accessed_idxs = torch.zeros(self.shape[:2], dtype=torch.bool) # reset
        # return
        return idx_data

    
@dataclass
class SampleMetadata:
    n: int
    sample_id: str

    shape = property(lambda self: (self.n, self.n))

@dataclass
class FPSampleMetadata:
    data: list[list[SampleMetadata]]

    def __getitem__(self, idx: tuple[int,int]) -> SampleMetadata:
        return self.data[idx[0]][idx[1]]
    
    n = property(lambda self: self.data[0][0].n)
    shape = property(lambda self: (self.n, self.n))

    def __post_init__(self):
        n = self.n
        shape = self.shape
        for i, row in enumerate(self.data):
            for j, meta in enumerate(row):
                assert meta.n == n, f"Sample at ({i},{j}) has n = {meta.n} != {n}"
                assert meta.shape == shape, f"Sample at ({i},{j}) has shape = {meta.shape} != {shape}"


Sample = tuple[Float[torch.Tensor, "n n"], SampleMetadata]

@dataclass
class FPSample:
    data: LoadableTensor
    metadata: FPSampleMetadata

    def get(self, idx: tuple[int,int]) -> Sample:
        return (
            self.data.get()[idx],
            self.metadata[idx]
        )

    

class FPSubsampleDataset(Dataset):
    """Flatten a list of FPSamples into individual sub-samples.

    Each dataset index corresponds to one (i, j) sub-sample from exactly one FPSample.
    """

    def __init__(self, samples: list[FPSample]) -> None:
        """
        # Parameters:
         - `samples : list[FPSample]`
            A list of FPSample objects, each with shape [L, H, n, n].
        """
        super().__init__()
        self.samples = samples  # list of FPSample

        # We'll build a map: dataset_index -> (fpsample_idx, i, j)
        self._index_map: list[tuple[int, int, int]] = []

        for fpsample_idx, fpsample in enumerate(samples):
            L, H, n, n_ = fpsample.data.shape
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


class FPSubsampleDataset(Dataset):
    ...
    def __init__(self, samples: list[FPSample]) -> None:
        ...
        self._shapes: list[tuple[int, int]] = []

        for fpsample_idx, fpsample in enumerate(samples):
            L, H, n, n_ = fpsample.data.shape
            # store (n, n) once
            shape_ij = (n, n_)
            for i in range(L):
                for j in range(H):
                    self._index_map.append((fpsample_idx, i, j))
                    self._shapes.append(shape_ij)
    ...

    def get_subsample_shape(self, idx: int) -> tuple[int, int]:
        """Return (n,n) shape for sub-sample at dataset index = idx."""
        return self._shapes[idx]


from torch.utils.data import Sampler
import random
from collections import defaultdict

class ShapeBucketingSampler(Sampler[list[int]]):
    """
    Groups dataset indices by shape, yields mini-batches of uniform shape.
    """

    def __init__(
        self,
        dataset: FPSubsampleDataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by shape
        shape_buckets: dict[tuple[int,int], list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            shape_ij = dataset.get_subsample_shape(idx)  # (n,n)
            shape_buckets[shape_ij].append(idx)
        
        # Now we break each shape's list into mini-batches
        self.batches: list[list[int]] = []
        for shape_k, indices in shape_buckets.items():
            if self.shuffle:
                random.shuffle(indices)
            # chunk into groups of size = batch_size
            for start in range(0, len(indices), self.batch_size):
                self.batches.append(indices[start : start + self.batch_size])
        
        # shuffle the order of these shape-batched groups
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        # yield each list of indices
        for batch_indices in self.batches:
            yield batch_indices

    def __len__(self) -> int:
        # number of mini-batches total
        return len(self.batches)


from torch.utils.data.dataloader import default_collate

def stack_collate_fn(
    batch: list[Sample],  
    # Sample = (Float[torch.Tensor, "n n"], SampleMetadata)
) -> tuple[Float[torch.Tensor, "batch n n"], list[SampleMetadata]]:
    """
    Collate a list of (tensor [n,n], metadata) into:
      (stacked [B,n,n], list_of_metadata)
    """
    # separate the data
    tensors = [sample[0] for sample in batch]
    metas   = [sample[1] for sample in batch]
    # stack the [n,n] Tensors => [B, n, n]
    stacked = torch.stack(tensors, dim=0)
    return (stacked, metas)
