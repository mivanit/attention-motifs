"""Pattern types system for labeling hierarchical clustering cuts.

This module provides a way to export clustering cuts at specific heights
and assign human-readable labels to the resulting clusters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TypedDict

from attention_motifs.features.clustering import HierarchicalClusteringResult


class PatternTypeDict(TypedDict):
	"""JSON schema for a single pattern type."""

	id: int
	name: str
	description: str


class PatternTypesMetaDict(TypedDict):
	"""JSON schema for pattern types metadata."""

	cut_height: float | None
	n_clusters: int
	linkage_method: str
	clustering_path: str
	created_at: str


class PatternTypesStatsDict(TypedDict):
	"""JSON schema for pattern types statistics."""

	n_heads: int
	cluster_sizes: dict[str, int]


class PatternTypesDict(TypedDict):
	"""JSON schema for the full pattern types file."""

	meta: PatternTypesMetaDict
	stats: PatternTypesStatsDict
	types: list[PatternTypeDict]
	assignments: dict[str, int]


@dataclass
class PatternType:
	"""A single pattern type (cluster label).

	Attributes:
		id: Cluster index (0-indexed)
		name: Human-readable name, "none" if unlabeled
		description: Description of the pattern type, "none" if unlabeled
	"""

	id: int
	name: str = "none"
	description: str = "none"

	def serialize(self) -> PatternTypeDict:
		"""Convert to JSON-compatible dict."""
		return PatternTypeDict(
			id=self.id,
			name=self.name,
			description=self.description,
		)

	@classmethod
	def load(cls, data: PatternTypeDict) -> "PatternType":
		"""Create instance from dict."""
		return cls(
			id=data["id"],
			name=data.get("name", "none"),
			description=data.get("description", "none"),
		)


@dataclass
class PatternTypesMeta:
	"""Metadata about how the clustering was generated.

	Attributes:
		cut_height: Height at which dendrogram was cut (None if n_clusters used)
		n_clusters: Number of clusters
		linkage_method: Linkage method used (e.g., "average", "ward")
		clustering_path: Path to source clustering data
		created_at: ISO timestamp of when this was created
	"""

	cut_height: float | None
	n_clusters: int
	linkage_method: str
	clustering_path: str
	created_at: str

	def serialize(self) -> PatternTypesMetaDict:
		"""Convert to JSON-compatible dict."""
		return PatternTypesMetaDict(
			cut_height=self.cut_height,
			n_clusters=self.n_clusters,
			linkage_method=self.linkage_method,
			clustering_path=self.clustering_path,
			created_at=self.created_at,
		)

	@classmethod
	def load(cls, data: PatternTypesMetaDict) -> "PatternTypesMeta":
		"""Create instance from dict."""
		return cls(
			cut_height=data.get("cut_height"),
			n_clusters=data["n_clusters"],
			linkage_method=data["linkage_method"],
			clustering_path=data["clustering_path"],
			created_at=data["created_at"],
		)


@dataclass
class PatternTypesStats:
	"""Statistics about the clustering.

	Attributes:
		n_heads: Total number of heads
		cluster_sizes: Mapping from cluster ID (as string) to count
	"""

	n_heads: int
	cluster_sizes: dict[str, int]

	def serialize(self) -> PatternTypesStatsDict:
		"""Convert to JSON-compatible dict."""
		return PatternTypesStatsDict(
			n_heads=self.n_heads,
			cluster_sizes=self.cluster_sizes,
		)

	@classmethod
	def load(cls, data: PatternTypesStatsDict) -> "PatternTypesStats":
		"""Create instance from dict."""
		return cls(
			n_heads=data["n_heads"],
			cluster_sizes=data["cluster_sizes"],
		)


@dataclass
class PatternTypes:
	"""Collection of pattern types with head assignments.

	Attributes:
		meta: Metadata about the clustering
		stats: Statistics about cluster sizes
		types: List of pattern type definitions
		assignments: Mapping from head_id to cluster_id
	"""

	meta: PatternTypesMeta
	stats: PatternTypesStats
	types: list[PatternType]
	assignments: dict[str, int]

	# --- File I/O ---

	def save(self, path: Path | str) -> None:
		"""Write to JSON file."""
		output_path: Path = Path(path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with open(output_path, "w") as f:
			json.dump(self.serialize(), f, indent="\t")

	@classmethod
	def read(cls, path: Path | str) -> "PatternTypes":
		"""Load from JSON file."""
		input_path: Path = Path(path)
		with open(input_path, "r") as f:
			data: PatternTypesDict = json.load(f)
		return cls.load(data)

	# --- Dict conversion ---

	def serialize(self) -> PatternTypesDict:
		"""Convert to JSON-compatible dict."""
		return PatternTypesDict(
			meta=self.meta.serialize(),
			stats=self.stats.serialize(),
			types=[t.serialize() for t in self.types],
			assignments=self.assignments,
		)

	@classmethod
	def load(cls, data: PatternTypesDict) -> "PatternTypes":
		"""Create instance from dict."""
		return cls(
			meta=PatternTypesMeta.load(data["meta"]),
			stats=PatternTypesStats.load(data["stats"]),
			types=[PatternType.load(t) for t in data["types"]],
			assignments=data["assignments"],
		)

	# --- Queries ---

	def get_type(self, head_id: str) -> PatternType | None:
		"""Get pattern type for a head, or None if head not found."""
		cluster_id: int | None = self.assignments.get(head_id)
		if cluster_id is None:
			return None
		for t in self.types:
			if t.id == cluster_id:
				return t
		return None

	def get_type_name(self, head_id: str) -> str:
		"""Get pattern type name for a head. Returns 'none' if not found."""
		pattern_type: PatternType | None = self.get_type(head_id)
		if pattern_type is None:
			return "none"
		return pattern_type.name

	def heads_by_type(self, type_id: int) -> list[str]:
		"""Get all head IDs assigned to a specific cluster."""
		return [h for h, c in self.assignments.items() if c == type_id]

	def type_counts(self) -> dict[int, int]:
		"""Get count of heads for each cluster ID."""
		counts: dict[int, int] = {}
		for cluster_id in self.assignments.values():
			counts[cluster_id] = counts.get(cluster_id, 0) + 1
		return counts

	# --- Factory ---

	@classmethod
	def from_clustering(
		cls,
		clustering: HierarchicalClusteringResult,
		cut_height: float | None = None,
		n_clusters: int | None = None,
		clustering_path: str = "",
	) -> "PatternTypes":
		"""Create PatternTypes from a clustering result.

		Args:
			clustering: The hierarchical clustering result
			cut_height: Height at which to cut the dendrogram
			n_clusters: Number of clusters (alternative to cut_height)
			clustering_path: Path to source clustering data (for metadata)

		Returns:
			PatternTypes with unlabeled types and cluster assignments
		"""
		# Get cluster assignments
		assignments: dict[str, int] = clustering.get_clusters(
			n_clusters=n_clusters,
			cut_height=cut_height,
		)

		# Compute stats
		n_heads: int = len(assignments)
		cluster_ids: set[int] = set(assignments.values())
		actual_n_clusters: int = len(cluster_ids)

		cluster_sizes: dict[str, int] = {}
		for cluster_id in sorted(cluster_ids):
			count: int = sum(1 for c in assignments.values() if c == cluster_id)
			cluster_sizes[str(cluster_id)] = count

		# Create metadata
		meta: PatternTypesMeta = PatternTypesMeta(
			cut_height=cut_height,
			n_clusters=actual_n_clusters,
			linkage_method=clustering.linkage_method,
			clustering_path=clustering_path,
			created_at=datetime.now(timezone.utc).isoformat(),
		)

		# Create stats
		stats: PatternTypesStats = PatternTypesStats(
			n_heads=n_heads,
			cluster_sizes=cluster_sizes,
		)

		# Create unlabeled types for each cluster
		types: list[PatternType] = [
			PatternType(id=i, name="none", description="none")
			for i in sorted(cluster_ids)
		]

		return cls(
			meta=meta,
			stats=stats,
			types=types,
			assignments=assignments,
		)
