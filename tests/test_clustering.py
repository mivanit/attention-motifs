"""Tests for features.clustering — hierarchical clustering and serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attention_motifs.features.clustering import HierarchicalClusteringResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_distance_matrix() -> tuple[np.ndarray, list[str]]:
	"""Create a 4×4 symmetric distance matrix with two clear clusters."""
	# heads 0,1 close; heads 2,3 close; far apart between groups
	distances: np.ndarray = np.array(
		[
			[0.0, 0.1, 2.0, 2.1],
			[0.1, 0.0, 2.1, 2.0],
			[2.0, 2.1, 0.0, 0.2],
			[2.1, 2.0, 0.2, 0.0],
		]
	)
	cls_values: list[str] = ["m:L0:H0", "m:L0:H1", "m:L1:H0", "m:L1:H1"]
	return distances, cls_values


def _make_clustering() -> HierarchicalClusteringResult:
	distances, cls_values = _make_distance_matrix()
	return HierarchicalClusteringResult.from_distance_matrix(
		distances, cls_values, method="average"
	)


# ---------------------------------------------------------------------------
# from_distance_matrix
# ---------------------------------------------------------------------------


class TestFromDistanceMatrix:
	def test_basic(self) -> None:
		result: HierarchicalClusteringResult = _make_clustering()
		assert len(result.cls_values) == 4
		assert result.linkage_method == "average"
		# linkage matrix has n-1 rows for n items
		assert result.linkage_matrix.shape == (3, 4)

	def test_different_methods(self) -> None:
		distances, cls_values = _make_distance_matrix()
		for method in ("average", "complete", "single"):
			result: HierarchicalClusteringResult = (
				HierarchicalClusteringResult.from_distance_matrix(
					distances, cls_values, method=method
				)
			)
			assert result.linkage_method == method
			assert result.linkage_matrix.shape[0] == 3


# ---------------------------------------------------------------------------
# get_clusters
# ---------------------------------------------------------------------------


class TestGetClusters:
	def test_n_clusters(self) -> None:
		result: HierarchicalClusteringResult = _make_clustering()
		assignments: dict[str, int] = result.get_clusters(n_clusters=2)
		assert len(assignments) == 4
		# 0-indexed
		assert all(v >= 0 for v in assignments.values())
		# exactly 2 unique clusters
		assert len(set(assignments.values())) == 2
		# heads in same group should be in same cluster
		assert assignments["m:L0:H0"] == assignments["m:L0:H1"]
		assert assignments["m:L1:H0"] == assignments["m:L1:H1"]
		# heads in different groups should differ
		assert assignments["m:L0:H0"] != assignments["m:L1:H0"]

	def test_cut_height(self) -> None:
		result: HierarchicalClusteringResult = _make_clustering()
		# low cut → more clusters
		assignments: dict[str, int] = result.get_clusters(cut_height=0.5)
		assert len(set(assignments.values())) >= 2

	def test_neither_raises(self) -> None:
		result: HierarchicalClusteringResult = _make_clustering()
		with pytest.raises(ValueError, match="Must specify"):
			result.get_clusters()


# ---------------------------------------------------------------------------
# get_max_height
# ---------------------------------------------------------------------------


class TestGetMaxHeight:
	def test_positive(self) -> None:
		result: HierarchicalClusteringResult = _make_clustering()
		max_h: float = result.get_max_height()
		assert max_h > 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
	def test_serialize_load_roundtrip(self) -> None:
		original: HierarchicalClusteringResult = _make_clustering()
		data: dict = original.serialize()
		restored: HierarchicalClusteringResult = HierarchicalClusteringResult.load(data)
		assert restored.cls_values == original.cls_values
		assert restored.linkage_method == original.linkage_method
		np.testing.assert_array_almost_equal(
			restored.linkage_matrix, original.linkage_matrix
		)

	def test_save_read_roundtrip(self, tmp_path: Path) -> None:
		original: HierarchicalClusteringResult = _make_clustering()
		output_dir: Path = tmp_path / "clustering"
		original.save(output_dir)

		assert (output_dir / "clustering_meta.json").exists()
		assert (output_dir / "linkage.npy").exists()
		assert (output_dir / "linkage.json").exists()

		restored: HierarchicalClusteringResult = HierarchicalClusteringResult.read(
			output_dir
		)
		assert restored.cls_values == original.cls_values
		assert restored.linkage_method == original.linkage_method
		np.testing.assert_array_almost_equal(
			restored.linkage_matrix, original.linkage_matrix
		)
