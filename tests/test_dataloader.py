# tests/test_fp_subsample.py

import pytest
import torch
import numpy as np
import shutil
import random
from pathlib import Path

from attention_motifs.dataloader import (
    LoadableTensor,
    SampleMetadata,
    FPSampleMetadata,
    FPSample,
    FPSubsampleDataset,
    ShapeBucketingSampler,
    stack_collate_fn,
)

##########################################################################
# GLOBAL TEMP DIRECTORY
##########################################################################
TEMP_DIR: Path = Path("tests/_temp")

##########################################################################
# FIXTURES
##########################################################################

@pytest.fixture(scope="session", autouse=True)
def prepare_global_temp_dir():
    """Create a global TEMP_DIR for test files, then remove after the tests."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.fixture
def clear_random_seed():
    """Clear random seed for reproducibility (optional)."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    yield


##########################################################################
# HELPERS
##########################################################################

def write_npy_file(dir_: Path, name: str, arr: np.ndarray) -> Path:
    """
    Saves `arr` as `name.npy` in `dir_` and returns the full path.
    """
    p = dir_ / f"{name}.npy"
    np.save(p, arr)
    return p


def build_fp_sample(
    dir_: Path, 
    file_basename: str,
    data_4d: np.ndarray,
    meta_prefix: str = "sample"
) -> FPSample:
    """
    Write `data_4d` to (dir_ / f"{file_basename}.npy").
    Build a corresponding FPSample with consistent metadata.
    """
    path = write_npy_file(dir_, file_basename, data_4d)
    L, H, n, n_ = data_4d.shape
    # Build FPSampleMetadata
    meta_grid = []
    for i in range(L):
        row = []
        for j in range(H):
            row.append(SampleMetadata(n=n, sample_id=f"{meta_prefix}_{i}_{j}"))
        meta_grid.append(row)
    fps_meta = FPSampleMetadata(meta_grid)
    # Wrap in LoadableTensor
    lt = LoadableTensor(path, (L, H, n, n_))
    return FPSample(data=lt, metadata=fps_meta)


##########################################################################
# TESTS
##########################################################################

def test_basic_loadable_tensor(clear_random_seed):
    """
    Basic test for LoadableTensor: 
      - ensure it loads from file 
      - confirm shape 
      - confirm unload after all sub-samples accessed 
    """
    arr = np.random.randn(2, 3, 4, 4)
    path = write_npy_file(TEMP_DIR, "basic_test", arr)
    lt = LoadableTensor(path, arr.shape)

    # Not loaded until get()/get_idx() is called
    assert lt.data is None
    t1 = lt.get()
    assert t1.shape == arr.shape
    assert lt.data is not None

    # Access sub-sample => still loaded until all are used
    sub = lt.get_idx((0, 0))
    assert sub.shape == (4, 4)
    assert lt.data is not None

    # Access all => triggers unload
    L, H, n, _ = arr.shape
    for i in range(L):
        for j in range(H):
            _ = lt.get_idx((i, j))
    assert lt.data is None  # unloaded now


def test_fp_sample_metadata():
    """
    Validate FPSampleMetadata constraints and shape property.
    """
    row0 = [SampleMetadata(n=4, sample_id="r0c0"), SampleMetadata(n=4, sample_id="r0c1")]
    row1 = [SampleMetadata(n=4, sample_id="r1c0"), SampleMetadata(n=4, sample_id="r1c1")]
    fps_meta = FPSampleMetadata([row0, row1])
    assert fps_meta.n == 4
    assert fps_meta.shape == (4,4)
    assert fps_meta[(1,1)].sample_id == "r1c1"


def test_fpsample(clear_random_seed):
    """
    Test building an FPSample from a real .npy file, 
    and retrieving sub-samples (tensor, metadata).
    """
    arr = np.random.randn(2, 2, 5, 5)
    fpsample = build_fp_sample(TEMP_DIR, "fpsample_test", arr, "meta")
    # Check shape
    L, H, n, n_ = arr.shape
    assert fpsample.data.shape == (L, H, n, n_)
    # Fetch sub-sample
    sub0, meta0 = fpsample.get((0,0))
    assert sub0.shape == (5,5)
    assert meta0.sample_id == "meta_0_0"

    # Ensure partial usage doesn't unload
    assert fpsample.data.data is not None

    # Use them all => triggers unload
    for i in range(L):
        for j in range(H):
            fpsample.get((i,j))
    assert fpsample.data.data is None


def test_fpsubsample_dataset_single(clear_random_seed):
    """
    Test FPSubsampleDataset with a single FPSample (2,3,4,4).
    """
    arr = np.random.randn(2, 3, 4, 4)
    fpsample = build_fp_sample(TEMP_DIR, "single_ds", arr, "dsmeta")
    ds = FPSubsampleDataset([fpsample])

    assert len(ds) == 2*3  # L*H=6
    x0, m0 = ds[0]  # => (0,0)
    assert x0.shape == (4,4)
    assert m0.sample_id == "dsmeta_0_0"

    x_last, m_last = ds[5]
    assert x_last.shape == (4,4)
    assert m_last.sample_id == "dsmeta_1_2"

    # test get_subsample_shape
    shape0 = ds.get_subsample_shape(0)
    shape5 = ds.get_subsample_shape(5)
    assert shape0 == (4,4)
    assert shape5 == (4,4)


def test_fpsubsample_dataset_multiple(clear_random_seed):
    """
    Test FPSubsampleDataset with multiple FPSample, each different shape.
    """
    arr1 = np.random.randn(1, 2, 3, 3)
    fpsample1 = build_fp_sample(TEMP_DIR, "ds_multi_1", arr1, "m1")
    arr2 = np.random.randn(2, 2, 5, 5)
    fpsample2 = build_fp_sample(TEMP_DIR, "ds_multi_2", arr2, "m2")

    ds = FPSubsampleDataset([fpsample1, fpsample2])
    # total sub-samples = (1*2) + (2*2) = 2 + 4 = 6
    assert len(ds) == 6

    # Indices: 
    #   fpsample1 => (0,0)->(0,1) 
    #   fpsample2 => (1,0)->(1,1) ...
    x0, m0 = ds[0]
    assert x0.shape == (3,3)
    assert m0.sample_id == "m1_0_0"
    x2, m2 = ds[2]  
    # That should be the first sub-sample from fpsample2 => (0,0) in that sample
    assert x2.shape == (5,5)
    assert m2.sample_id == "m2_0_0"


def test_shape_bucketing_sampler_no_shuffle(clear_random_seed):
    """
    ShapeBucketingSampler with shuffle=False:
    Confirm that sub-samples of the same shape are batched together 
    in the order they appear.
    """
    # We'll mock a dataset with shapes:
    #   idx=0..2 => shape(3,3)
    #   idx=3..5 => shape(5,5)
    class MockDataset(FPSubsampleDataset):
        def __init__(self):
            pass
        def __len__(self):
            return 6
        def get_subsample_shape(self, idx: int):
            if idx < 3:
                return (3,3)
            else:
                return (5,5)
        def __getitem__(self, idx: int):
            return None

    ds = MockDataset()
    sampler = ShapeBucketingSampler(ds, batch_size=2, shuffle=False)
    all_batches = list(iter(sampler))

    # shape(3,3)-> [0,1,2], shape(5,5)-> [3,4,5]
    # chunk by size=2 => 
    #   [0,1], [2], [3,4], [5]
    assert len(all_batches) == 4
    assert all_batches[0] == [0,1]
    assert all_batches[1] == [2]
    assert all_batches[2] == [3,4]
    assert all_batches[3] == [5]


def test_shape_bucketing_sampler_shuffle(clear_random_seed):
    """
    With shuffle=True, we can't guarantee the *exact* order, 
    but we can check that shape groups are never mixed.
    """
    class MockDataset(FPSubsampleDataset):
        def __init__(self):
            pass
        def __len__(self):
            return 6
        def get_subsample_shape(self, idx: int):
            if idx < 3:
                return (3,3)
            else:
                return (5,5)
        def __getitem__(self, idx: int):
            return None

    ds = MockDataset()
    sampler = ShapeBucketingSampler(ds, batch_size=2, shuffle=True)
    all_batches = list(iter(sampler))

    # We expect two shape groups: {0,1,2} for (3,3) and {3,4,5} for (5,5).
    # chunk by 2 => e.g. [0,1], [2] in some order, and [3,4], [5] in some order.
    # the overall group order might shuffle, but each group shouldn't mix shapes.
    # We'll just verify that no batch has mixed shapes.
    for batch_idxs in all_batches:
        shapes = [ds.get_subsample_shape(i) for i in batch_idxs]
        assert len(set(shapes)) == 1, f"Batch {batch_idxs} has mixed shapes"


def test_stack_collate_fn(clear_random_seed):
    """
    Test collate function merges a list of (tensor, meta) 
    into (stacked_tensor, list_of_meta).
    """
    # Make fake samples
    t1 = torch.ones(3,3)
    t2 = torch.zeros(3,3)
    m1 = SampleMetadata(n=3, sample_id="s1")
    m2 = SampleMetadata(n=3, sample_id="s2")

    batch = [(t1, m1), (t2, m2)]
    stacked, metas = stack_collate_fn(batch)
    assert stacked.shape == (2,3,3)
    assert metas[0].sample_id == "s1"
    assert metas[1].sample_id == "s2"
    assert torch.allclose(stacked[0], t1)
    assert torch.allclose(stacked[1], t2)


def test_partial_usage_unload(clear_random_seed):
    """
    Verify that if not all sub-samples are accessed, data remains loaded, 
    and that a subsequent complete usage triggers unloading.
    """
    arr = np.random.randn(2, 2, 4, 4)
    fpsample = build_fp_sample(TEMP_DIR, "partial_unload", arr, "partial")

    # We have L=2, H=2 => 4 sub-samples total.
    # Access only 3 of them => data remains loaded
    _ = fpsample.get((0,0))
    _ = fpsample.get((0,1))
    _ = fpsample.get((1,0))
    assert fpsample.data.data is not None, "not all sub-samples used => still loaded"

    # Access the last one => triggers unload
    _ = fpsample.get((1,1))
    assert fpsample.data.data is None, "all sub-samples used => unloaded"


def test_reload_after_unload(clear_random_seed):
    """
    If the data is unloaded, calling get() again should reload from disk.
    """
    arr = np.random.randn(1, 2, 4, 4)
    fpsample = build_fp_sample(TEMP_DIR, "reload", arr, "reloadmeta")

    # Access everything => unload
    L, H, n, _ = arr.shape
    for i in range(L):
        for j in range(H):
            fpsample.get((i,j))
    assert fpsample.data.data is None

    # Now access again => reload from disk
    sub, meta = fpsample.get((0,1))
    assert fpsample.data.data is not None, "should have reloaded"
    assert sub.shape == (4,4)
    assert meta.sample_id == "reloadmeta_0_1"
