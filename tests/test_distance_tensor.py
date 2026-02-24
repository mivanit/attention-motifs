"""Tests for DistanceTensorResult.build_distance_tensor — serial/parallel, reduced/full."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from jaxtyping import Float

from attention_motifs.features.analysis import (
	DistanceTensorResult,
	_build_distance_tensor,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

N_HEADS: int = 5
N_PROMPTS: int = 10
N_FEATURES: int = 3


def _make_df(
	n_heads: int = N_HEADS,
	n_prompts: int = N_PROMPTS,
	n_features: int = N_FEATURES,
	seed: int = 42,
) -> pl.DataFrame:
	"""Build a synthetic DataFrame matching the schema expected by build_distance_tensor."""
	rng: np.random.Generator = np.random.default_rng(seed)
	rows: list[dict[str, object]] = []
	for p_idx in range(n_prompts):
		for h_idx in range(n_heads):
			row: dict[str, object] = {
				"activation.cls": f"model.L{h_idx // 2}.H{h_idx % 2}",
				"activation.prompt": f"prompt_{p_idx}",
			}
			for f_idx in range(n_features):
				row[f"feat.f{f_idx}"] = float(rng.standard_normal())
			rows.append(row)
	return pl.DataFrame(rows)


def _make_df_missing(
	n_heads: int = N_HEADS,
	n_prompts: int = N_PROMPTS,
	n_features: int = N_FEATURES,
	seed: int = 42,
	drop_fraction: float = 0.1,
) -> pl.DataFrame:
	"""Like _make_df but with some (cls, prompt) pairs removed."""
	df: pl.DataFrame = _make_df(n_heads, n_prompts, n_features, seed)
	rng: np.random.Generator = np.random.default_rng(seed + 1)
	n_drop: int = int(len(df) * drop_fraction)
	drop_indices: np.ndarray = rng.choice(len(df), size=n_drop, replace=False)
	keep_mask: list[bool] = [i not in drop_indices for i in range(len(df))]
	return df.filter(pl.Series(keep_mask))


# ---------------------------------------------------------------------------
# _build_distance_tensor (core helper)
# ---------------------------------------------------------------------------


class TestBuildDistanceTensorHelper:
	"""Tests for the free-standing _build_distance_tensor helper."""

	def test_reduce_true_shape(self) -> None:
		"""reduce=True → (h, h) output."""
		rng: np.random.Generator = np.random.default_rng(0)
		data: Float[np.ndarray, "p h d"] = rng.standard_normal((8, 4, 3))
		result: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, reduce=True
		)
		assert result.shape == (4, 4)

	def test_reduce_false_shape(self) -> None:
		"""reduce=False → (h, h, p) output."""
		rng: np.random.Generator = np.random.default_rng(0)
		data: Float[np.ndarray, "p h d"] = rng.standard_normal((8, 4, 3))
		result: Float[np.ndarray, "h h p"] = _build_distance_tensor(
			data, reduce=False
		)
		assert result.shape == (4, 4, 8)

	def test_reduce_matches_full_mean(self) -> None:
		"""reduce=True result equals reduce=False result averaged over prompts."""
		rng: np.random.Generator = np.random.default_rng(1)
		data: Float[np.ndarray, "p h d"] = rng.standard_normal((12, 6, 4))

		reduced: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, reduce=True
		)
		full: Float[np.ndarray, "h h p"] = _build_distance_tensor(
			data, reduce=False
		)
		np.testing.assert_allclose(reduced, full.mean(axis=-1), rtol=1e-12)

	def test_symmetric(self) -> None:
		"""Distance matrix is symmetric."""
		rng: np.random.Generator = np.random.default_rng(2)
		data: Float[np.ndarray, "p h d"] = rng.standard_normal((5, 7, 3))
		result: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, reduce=True
		)
		np.testing.assert_allclose(result, result.T, atol=1e-14)

	def test_diagonal_zero(self) -> None:
		"""Diagonal of distance matrix is zero (distance to self)."""
		rng: np.random.Generator = np.random.default_rng(3)
		data: Float[np.ndarray, "p h d"] = rng.standard_normal((5, 4, 3))
		result: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, reduce=True
		)
		np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-14)

	def test_identical_features_zero(self) -> None:
		"""All-identical feature vectors → all distances zero."""
		data: Float[np.ndarray, "p h d"] = np.ones((5, 3, 2), dtype=np.float64)
		result: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, reduce=True
		)
		np.testing.assert_allclose(result, 0.0, atol=1e-14)

	def test_single_prompt(self) -> None:
		"""Single prompt still produces valid (h, h) output."""
		rng: np.random.Generator = np.random.default_rng(4)
		data: Float[np.ndarray, "p h d"] = rng.standard_normal((1, 4, 3))
		result: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, reduce=True
		)
		assert result.shape == (4, 4)
		np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-14)

	def test_single_head(self) -> None:
		"""Single head → (1, 1) of zeros."""
		rng: np.random.Generator = np.random.default_rng(5)
		data: Float[np.ndarray, "p h d"] = rng.standard_normal((10, 1, 3))
		result: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, reduce=True
		)
		assert result.shape == (1, 1)
		assert result[0, 0] == pytest.approx(0.0)

	def test_l1_norm(self) -> None:
		"""order=1 uses Manhattan distance."""
		data: Float[np.ndarray, "p h d"] = np.array(
			[[[0.0, 0.0], [1.0, 1.0]]], dtype=np.float64
		)
		result: Float[np.ndarray, "h h"] = _build_distance_tensor(
			data, order=1, reduce=True
		)
		# Manhattan distance between [0,0] and [1,1] = 2.0
		assert result[0, 1] == pytest.approx(2.0)
		assert result[1, 0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# build_distance_tensor (classmethod, full pipeline)
# ---------------------------------------------------------------------------


class TestBuildDistanceTensor:
	"""Tests for DistanceTensorResult.build_distance_tensor classmethod."""

	def test_serial_reduced(self) -> None:
		"""Serial + reduce=True → is_reduced=True, (h, h) shape."""
		df: pl.DataFrame = _make_df()
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		assert result.is_reduced is True
		assert result.distances.shape == (N_HEADS, N_HEADS)

	def test_serial_full(self) -> None:
		"""Serial + reduce=False → is_reduced=False, (h, h, p) shape."""
		df: pl.DataFrame = _make_df()
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=False)
		)
		assert result.is_reduced is False
		assert result.distances.shape == (N_HEADS, N_HEADS, N_PROMPTS)

	def test_parallel_reduced(self) -> None:
		"""Parallel + reduce=True (default) works."""
		df: pl.DataFrame = _make_df()
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(
				df, parallel=True, n_proc=2
			)
		)
		assert result.is_reduced is True
		assert result.distances.shape == (N_HEADS, N_HEADS)

	def test_parallel_requires_reduce(self) -> None:
		"""parallel=True with reduce=False raises AssertionError."""
		df: pl.DataFrame = _make_df()
		with pytest.raises(AssertionError, match="parallel mode requires reduce"):
			DistanceTensorResult.build_distance_tensor(
				df, parallel=True, reduce=False
			)


class TestEquivalence:
	"""Equivalence between serial full, serial reduced, and parallel reduced."""

	@pytest.fixture()
	def df(self) -> pl.DataFrame:
		return _make_df()

	def test_serial_reduced_matches_full_mean(
		self, df: pl.DataFrame
	) -> None:
		"""Serial reduce=True matches reduce=False .mean_dists."""
		full: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=False)
		)
		reduced: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		np.testing.assert_allclose(
			reduced.distances, full.mean_dists, rtol=1e-12
		)

	def test_parallel_matches_serial(self, df: pl.DataFrame) -> None:
		"""Parallel n_proc=2 matches serial reduce=True."""
		serial: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		parallel: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(
				df, parallel=True, n_proc=2
			)
		)
		np.testing.assert_allclose(
			parallel.distances, serial.distances, rtol=1e-12
		)

	def test_parallel_nproc_1_matches_serial(
		self, df: pl.DataFrame
	) -> None:
		"""Parallel with n_proc=1 (single-process fast path) matches serial."""
		serial: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		parallel_1: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(
				df, parallel=True, n_proc=1
			)
		)
		np.testing.assert_allclose(
			parallel_1.distances, serial.distances, rtol=1e-12
		)

	def test_parallel_nproc_4_matches_serial(
		self, df: pl.DataFrame
	) -> None:
		"""Parallel with n_proc=4 matches serial."""
		serial: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		parallel_4: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(
				df, parallel=True, n_proc=4
			)
		)
		np.testing.assert_allclose(
			parallel_4.distances, serial.distances, rtol=1e-12
		)


class TestMissingPrompts:
	"""Tests for include_missing_prompts filtering."""

	def test_missing_prompts_filtered(self) -> None:
		"""Prompts with missing cls rows are dropped by default."""
		df: pl.DataFrame = _make_df_missing()
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		assert result.is_reduced is True
		# fewer prompts than original since some were incomplete
		assert len(result.prompt_values) < N_PROMPTS
		assert result.distances.shape == (N_HEADS, N_HEADS)

	def test_missing_prompts_parallel_matches_serial(self) -> None:
		"""Parallel and serial agree on data with missing prompts."""
		df: pl.DataFrame = _make_df_missing()
		serial: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		parallel: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(
				df, parallel=True, n_proc=2
			)
		)
		np.testing.assert_allclose(
			parallel.distances, serial.distances, rtol=1e-12
		)
		assert serial.prompt_values == parallel.prompt_values


class TestProperties:
	"""Test structural properties of the output."""

	@pytest.fixture()
	def df(self) -> pl.DataFrame:
		return _make_df()

	def test_symmetric(self, df: pl.DataFrame) -> None:
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		np.testing.assert_allclose(
			result.distances, result.distances.T, atol=1e-14
		)

	def test_diagonal_zero(self, df: pl.DataFrame) -> None:
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		np.testing.assert_allclose(
			np.diag(result.distances), 0.0, atol=1e-14
		)

	def test_nonnegative(self, df: pl.DataFrame) -> None:
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		assert np.all(result.distances >= -1e-14)

	def test_cls_values_match_data(self, df: pl.DataFrame) -> None:
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		expected_cls: set[str] = set(
			df.get_column("activation.cls").unique().to_list()
		)
		assert set(result.cls_values) == expected_cls

	def test_mean_dists_property_consistent(
		self, df: pl.DataFrame
	) -> None:
		"""mean_dists property returns distances when is_reduced=True."""
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=True)
		)
		np.testing.assert_array_equal(result.mean_dists, result.distances)

	def test_full_mean_dists_property(self, df: pl.DataFrame) -> None:
		"""mean_dists on full tensor equals distances.mean(axis=-1)."""
		result: DistanceTensorResult = (
			DistanceTensorResult.build_distance_tensor(df, reduce=False)
		)
		np.testing.assert_allclose(
			result.mean_dists, result.distances.mean(axis=-1), rtol=1e-12
		)
