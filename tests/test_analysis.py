"""Tests for features.analysis — null stats, normalization, aggregation, covariance groups."""

from __future__ import annotations


import numpy as np
import polars as pl
import pytest
from jaxtyping import Float

from attention_motifs.features.analysis import (
	aggregate_feature_stats,
	groups_by_covariance,
	normalize_data,
	null_stats,
)


# ===========================================================================
# null_stats
# ===========================================================================


class TestNullStats:
	def test_no_missing(self) -> None:
		"""Clean dataframe → all counts 0."""
		df: pl.DataFrame = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
		result: pl.DataFrame = null_stats(df)
		assert result["missing_count"].sum() == 0

	def test_float_nan(self) -> None:
		"""Float column with NaN → counted as missing."""
		df: pl.DataFrame = pl.DataFrame({"x": [1.0, float("nan"), 3.0]})
		result: pl.DataFrame = null_stats(df)
		row: dict = result.filter(pl.col("feature") == "x").to_dicts()[0]
		assert row["missing_count"] == 1
		assert row["missing_frac"] == pytest.approx(1.0 / 3.0)

	def test_string_null(self) -> None:
		"""String column with null → counted as missing (not NaN check)."""
		df: pl.DataFrame = pl.DataFrame({"s": ["a", None, "c"]})
		result: pl.DataFrame = null_stats(df)
		row: dict = result.filter(pl.col("feature") == "s").to_dicts()[0]
		assert row["missing_count"] == 1

	def test_mixed_columns(self) -> None:
		"""Float NaN and int null handled separately by dtype branch."""
		df: pl.DataFrame = pl.DataFrame(
			{
				"f": [1.0, float("nan"), float("nan")],
				"i": [1, 2, 3],
			}
		)
		result: pl.DataFrame = null_stats(df)
		f_row: dict = result.filter(pl.col("feature") == "f").to_dicts()[0]
		i_row: dict = result.filter(pl.col("feature") == "i").to_dicts()[0]
		assert f_row["missing_count"] == 2
		assert i_row["missing_count"] == 0


# ===========================================================================
# normalize_data
# ===========================================================================


class TestNormalizeData:
	def test_basic(self) -> None:
		"""Computed stats → mean≈0, std≈1 per column."""
		df: pl.DataFrame = pl.DataFrame(
			{
				"id": ["a", "b", "c", "d"],
				"feat.x": [1.0, 2.0, 3.0, 4.0],
				"feat.y": [10.0, 20.0, 30.0, 40.0],
			}
		)
		norm_df, stats_df = normalize_data(df)

		# non-feature columns preserved
		assert norm_df["id"].to_list() == ["a", "b", "c", "d"]
		# normalized columns have mean ≈ 0
		assert norm_df["feat.x"].mean() == pytest.approx(0.0, abs=1e-10)
		assert norm_df["feat.y"].mean() == pytest.approx(0.0, abs=1e-10)
		# stats_df has correct schema
		assert set(stats_df.columns) == {"feature", "mean", "std"}
		assert stats_df.shape[0] == 2

	def test_with_presupplied_stats(self) -> None:
		"""Pre-supplied stats used instead of computing from data."""
		df: pl.DataFrame = pl.DataFrame(
			{
				"feat.x": [10.0, 20.0, 30.0],
			}
		)
		stats: pl.DataFrame = pl.DataFrame(
			{
				"feature": ["feat.x"],
				"mean": [0.0],
				"std": [10.0],
			}
		)
		norm_df, stats_df = normalize_data(df, stats=stats)

		# (10 - 0) / 10 = 1.0, (20 - 0) / 10 = 2.0, etc.
		assert norm_df["feat.x"].to_list() == pytest.approx([1.0, 2.0, 3.0])
		# returned stats_df is the one we passed in
		assert stats_df["mean"].to_list() == [0.0]

	def test_bad_stats_schema_raises(self) -> None:
		"""Stats with wrong columns → ValueError."""
		df: pl.DataFrame = pl.DataFrame({"feat.x": [1.0, 2.0]})
		bad_stats: pl.DataFrame = pl.DataFrame({"feature": ["feat.x"], "avg": [1.0]})
		with pytest.raises(ValueError, match="must have columns"):
			normalize_data(df, stats=bad_stats)

	def test_stats_missing_feature_raises(self) -> None:
		"""Stats missing an entry for a feature column → ValueError."""
		df: pl.DataFrame = pl.DataFrame({"feat.x": [1.0, 2.0], "feat.y": [3.0, 4.0]})
		stats: pl.DataFrame = pl.DataFrame(
			{"feature": ["feat.x"], "mean": [1.0], "std": [1.0]}
		)
		with pytest.raises(ValueError, match="lacks entries"):
			normalize_data(df, stats=stats)

	def test_no_feature_columns_raises(self) -> None:
		"""No feat.* columns and no explicit feature_cols → ValueError."""
		df: pl.DataFrame = pl.DataFrame({"col_a": [1.0, 2.0]})
		with pytest.raises(ValueError, match="No feature columns"):
			normalize_data(df)


# ===========================================================================
# aggregate_feature_stats
# ===========================================================================


class TestAggregateFeatureStats:
	def _make_df(self) -> pl.DataFrame:
		return pl.DataFrame(
			{
				"feature": [
					"feat.hist.mean",
					"feat.hist.variance",
					"feat.linreg.slope",
					"feat.linreg.r2",
				],
				"abs_mean": [0.5, 0.3, 0.8, 0.9],
			}
		)

	def test_group_by_last_token(self) -> None:
		"""side=-1 groups by last dot-separated token."""
		df: pl.DataFrame = self._make_df()
		result: pl.DataFrame = aggregate_feature_stats(df, side=-1)
		tokens: set[str] = set(result["token"].to_list())
		assert tokens == {"mean", "variance", "slope", "r2"}

	def test_group_by_first_token(self) -> None:
		"""side=0 groups by first token — all features start with 'feat'."""
		df: pl.DataFrame = self._make_df()
		result: pl.DataFrame = aggregate_feature_stats(df, side=0)
		# all 4 features share "feat" prefix → single group
		assert result.shape[0] == 1
		assert result["token"][0] == "feat"
		assert result["mean"][0] == pytest.approx((0.5 + 0.3 + 0.8 + 0.9) / 4.0)

	def test_group_by_middle_token(self) -> None:
		"""side=1 groups by second token (hist vs linreg)."""
		df: pl.DataFrame = self._make_df()
		result: pl.DataFrame = aggregate_feature_stats(df, side=1)
		tokens: set[str] = set(result["token"].to_list())
		assert tokens == {"hist", "linreg"}

	def test_skips_nan(self) -> None:
		"""NaN values in abs_mean are skipped, not included in aggregation."""
		df: pl.DataFrame = pl.DataFrame(
			{
				"feature": ["a.x", "a.y", "b.z"],
				"abs_mean": [1.0, float("nan"), 3.0],
			}
		)
		result: pl.DataFrame = aggregate_feature_stats(df, side=0)
		a_row: dict = result.filter(pl.col("token") == "a").to_dicts()[0]
		# only 1.0 counted (nan skipped)
		assert a_row["mean"] == pytest.approx(1.0)

	def test_set_side(self) -> None:
		"""side as set of indices joins selected tokens."""
		df: pl.DataFrame = pl.DataFrame(
			{
				"feature": ["a.b.c.d", "a.b.e.f"],
				"abs_mean": [1.0, 2.0],
			}
		)
		result: pl.DataFrame = aggregate_feature_stats(df, side={0, 1})
		# both features have "a.b" as tokens 0+1
		assert result.shape[0] == 1
		assert result["token"][0] == "a.b"


# ===========================================================================
# groups_by_covariance
# ===========================================================================


class TestGroupsByCovariance:
	def test_identity_all_singletons(self) -> None:
		"""Identity covariance with threshold > 1 → all singletons (diagonal=1, off-diag=0)."""
		features: list[str] = ["a", "b", "c"]
		cov: Float[np.ndarray, "3 3"] = np.eye(3)
		groups: list[list[str]] = groups_by_covariance(features, cov, threshold=0.5)
		# diagonal is 1.0 > 0.5, but each feature is only connected to itself
		# so each is its own group (diagonal makes self-loops, not cross-links)
		# Actually: adjacency[i,i] = True (1.0 > 0.5), so self-loops exist
		# but no off-diagonal edges → singletons
		assert len(groups) == 3
		flat: list[str] = [f for g in groups for f in g]
		assert set(flat) == {"a", "b", "c"}

	def test_two_blocks(self) -> None:
		"""Block-diagonal covariance → 2 connected components."""
		features: list[str] = ["a", "b", "c", "d"]
		# Two 2x2 blocks with high covariance, zero between blocks
		cov: Float[np.ndarray, "4 4"] = np.array(
			[
				[1.0, 0.9, 0.0, 0.0],
				[0.9, 1.0, 0.0, 0.0],
				[0.0, 0.0, 1.0, 0.8],
				[0.0, 0.0, 0.8, 1.0],
			]
		)
		groups: list[list[str]] = groups_by_covariance(features, cov, threshold=0.5)
		assert len(groups) == 2
		group_sets: list[set[str]] = [set(g) for g in groups]
		assert {"a", "b"} in group_sets
		assert {"c", "d"} in group_sets

	def test_high_threshold_all_singletons(self) -> None:
		"""Threshold above max covariance → everything disconnected."""
		features: list[str] = ["a", "b", "c"]
		cov: Float[np.ndarray, "3 3"] = np.array(
			[
				[1.0, 0.5, 0.3],
				[0.5, 1.0, 0.4],
				[0.3, 0.4, 1.0],
			]
		)
		groups: list[list[str]] = groups_by_covariance(features, cov, threshold=1.5)
		assert len(groups) == 3

	def test_low_threshold_single_group(self) -> None:
		"""Threshold below all off-diagonal values → single connected component."""
		features: list[str] = ["a", "b", "c"]
		cov: Float[np.ndarray, "3 3"] = np.array(
			[
				[1.0, 0.5, 0.3],
				[0.5, 1.0, 0.4],
				[0.3, 0.4, 1.0],
			]
		)
		groups: list[list[str]] = groups_by_covariance(features, cov, threshold=0.1)
		assert len(groups) == 1
		assert set(groups[0]) == {"a", "b", "c"}

	def test_use_abs_negative_covariance(self) -> None:
		"""use_abs=True treats negative covariance as strong connection."""
		features: list[str] = ["a", "b"]
		cov: Float[np.ndarray, "2 2"] = np.array(
			[
				[1.0, -0.9],
				[-0.9, 1.0],
			]
		)
		# With use_abs=True, |-0.9| = 0.9 > 0.5 → connected
		groups_abs: list[list[str]] = groups_by_covariance(
			features, cov, threshold=0.5, use_abs=True
		)
		assert len(groups_abs) == 1

		# With use_abs=False, -0.9 < 0.5 → disconnected
		groups_raw: list[list[str]] = groups_by_covariance(
			features, cov, threshold=0.5, use_abs=False
		)
		assert len(groups_raw) == 2
