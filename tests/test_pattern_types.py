"""Tests for pattern_types.pattern_types — serialization, queries, and clustering factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attention_motifs.features.clustering import HierarchicalClusteringResult
from attention_motifs.pattern_types.pattern_types import (
	PatternType,
	PatternTypeDict,
	PatternTypes,
	PatternTypesDict,
	PatternTypesMeta,
	PatternTypesStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern_types() -> PatternTypes:
	"""Create a minimal PatternTypes for testing."""
	return PatternTypes(
		meta=PatternTypesMeta(
			cut_height=1.5,
			n_clusters=2,
			linkage_method="average",
			clustering_path="/data/clustering",
			created_at="2025-01-01T00:00:00+00:00",
			min_cluster_size=None,
		),
		stats=PatternTypesStats(
			n_heads=4,
			cluster_sizes={"0": 2, "1": 2},
		),
		types=[
			PatternType(id=0, name="induction", description="Induction heads"),
			PatternType(id=1, name="previous", description="Previous token heads"),
		],
		assignments={
			"gpt2:L0:H0": 0,
			"gpt2:L0:H1": 0,
			"gpt2:L1:H0": 1,
			"gpt2:L1:H1": 1,
		},
	)


# ---------------------------------------------------------------------------
# PatternType
# ---------------------------------------------------------------------------


class TestPatternType:
	def test_serialize_load_roundtrip(self) -> None:
		pt: PatternType = PatternType(id=3, name="pos", description="Positional")
		data: PatternTypeDict = pt.serialize()
		restored: PatternType = PatternType.load(data)
		assert restored.id == 3
		assert restored.name == "pos"
		assert restored.description == "Positional"

	def test_load_defaults(self) -> None:
		"""Missing name/description default to 'none'."""
		data: PatternTypeDict = {"id": 0}  # type: ignore[typeddict-item]
		pt: PatternType = PatternType.load(data)
		assert pt.name == "none"
		assert pt.description == "none"


# ---------------------------------------------------------------------------
# PatternTypes — serialization
# ---------------------------------------------------------------------------


class TestPatternTypesSerialization:
	def test_serialize_load_roundtrip(self) -> None:
		original: PatternTypes = _make_pattern_types()
		data: PatternTypesDict = original.serialize()
		restored: PatternTypes = PatternTypes.load(data)

		assert restored.meta.cut_height == original.meta.cut_height
		assert restored.meta.n_clusters == original.meta.n_clusters
		assert restored.meta.linkage_method == original.meta.linkage_method
		assert restored.stats.n_heads == original.stats.n_heads
		assert restored.stats.cluster_sizes == original.stats.cluster_sizes
		assert len(restored.types) == len(original.types)
		assert restored.assignments == original.assignments

	def test_save_read_roundtrip(self, tmp_path: Path) -> None:
		original: PatternTypes = _make_pattern_types()
		json_path: Path = tmp_path / "pattern_types.json"
		original.save(json_path)

		assert json_path.exists()
		restored: PatternTypes = PatternTypes.read(json_path)

		assert restored.assignments == original.assignments
		assert restored.meta.n_clusters == original.meta.n_clusters
		assert len(restored.types) == len(original.types)


# ---------------------------------------------------------------------------
# PatternTypes — queries
# ---------------------------------------------------------------------------


class TestPatternTypesQueries:
	def test_get_type_found(self) -> None:
		pt: PatternTypes = _make_pattern_types()
		result: PatternType | None = pt.get_type("gpt2:L0:H0")
		assert result is not None
		assert result.id == 0
		assert result.name == "induction"

	def test_get_type_not_found(self) -> None:
		pt: PatternTypes = _make_pattern_types()
		assert pt.get_type("nonexistent:L9:H9") is None

	def test_get_type_name(self) -> None:
		pt: PatternTypes = _make_pattern_types()
		assert pt.get_type_name("gpt2:L1:H0") == "previous"
		assert pt.get_type_name("missing:L0:H0") == "none"

	def test_heads_by_type(self) -> None:
		pt: PatternTypes = _make_pattern_types()
		heads: list[str] = pt.heads_by_type(0)
		assert set(heads) == {"gpt2:L0:H0", "gpt2:L0:H1"}

	def test_heads_by_type_empty(self) -> None:
		pt: PatternTypes = _make_pattern_types()
		assert pt.heads_by_type(99) == []

	def test_type_counts(self) -> None:
		pt: PatternTypes = _make_pattern_types()
		counts: dict[int, int] = pt.type_counts()
		assert counts == {0: 2, 1: 2}


# ---------------------------------------------------------------------------
# PatternTypes.from_clustering
# ---------------------------------------------------------------------------


def _make_clustering(n_heads: int = 6) -> HierarchicalClusteringResult:
	"""Create a synthetic clustering from a distance matrix."""
	rng: np.random.Generator = np.random.default_rng(42)
	# two tight clusters + noise
	distances: np.ndarray = np.zeros((n_heads, n_heads))
	for i in range(n_heads):
		for j in range(i + 1, n_heads):
			# heads 0-2 close together, 3-5 close together, far apart between groups
			same_group: bool = (i < 3 and j < 3) or (i >= 3 and j >= 3)
			d: float = rng.uniform(0.1, 0.5) if same_group else rng.uniform(2.0, 3.0)
			distances[i, j] = d
			distances[j, i] = d

	cls_values: list[str] = [f"m:L0:H{i}" for i in range(n_heads)]
	return HierarchicalClusteringResult.from_distance_matrix(
		distances, cls_values, method="average"
	)


class TestPatternTypesFromClustering:
	def test_basic(self) -> None:
		clustering: HierarchicalClusteringResult = _make_clustering()
		pt: PatternTypes = PatternTypes.from_clustering(
			clustering, n_clusters=2
		)
		assert pt.meta.n_clusters == 2
		assert pt.stats.n_heads == 6
		assert len(pt.assignments) == 6
		# all assignments are 0-indexed
		assert all(v >= 0 for v in pt.assignments.values())

	def test_min_cluster_size_merges_small(self) -> None:
		"""Small clusters get merged into misc (id=-1)."""
		clustering: HierarchicalClusteringResult = _make_clustering()
		# request many clusters so some are small
		pt: PatternTypes = PatternTypes.from_clustering(
			clustering, n_clusters=5, min_cluster_size=2
		)
		# should have misc cluster for any singleton clusters
		cluster_ids: set[int] = set(pt.assignments.values())
		if PatternTypes.MISC_CLUSTER_ID in cluster_ids:
			# misc type should exist and have correct description
			misc_type: PatternType | None = None
			for t in pt.types:
				if t.id == PatternTypes.MISC_CLUSTER_ID:
					misc_type = t
					break
			assert misc_type is not None
			assert misc_type.name == "misc"
			assert "Merged from clusters" in misc_type.description

	def test_from_clustering_cut_height(self) -> None:
		clustering: HierarchicalClusteringResult = _make_clustering()
		pt: PatternTypes = PatternTypes.from_clustering(
			clustering, cut_height=1.0
		)
		assert pt.meta.cut_height == 1.0
		assert pt.stats.n_heads == 6
		assert len(pt.assignments) == 6
