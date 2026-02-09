"""Hierarchical clustering for attention head distances."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


LinkageMethod = Literal["ward", "average", "complete", "single"]


@dataclass
class HierarchicalClusteringResult:
	"""Result of hierarchical clustering on head distances.

	The linkage matrix follows scipy's format: each row [i, j, dist, count]
	represents a merge of clusters i and j at distance dist, creating a
	cluster with count original observations.
	"""

	cls_values: list[str]
	linkage_matrix: Float[np.ndarray, "n_merges 4"]
	linkage_method: LinkageMethod

	def serialize(self) -> dict:
		"""Convert to JSON-compatible dict."""
		return dict(
			cls_values=self.cls_values,
			linkage_matrix=self.linkage_matrix.tolist(),
			linkage_method=self.linkage_method,
		)

	@classmethod
	def load(cls, data: dict) -> "HierarchicalClusteringResult":
		"""Create instance from dict."""
		return cls(
			cls_values=data["cls_values"],
			linkage_matrix=np.array(data["linkage_matrix"]),
			linkage_method=data["linkage_method"],
		)

	def save(self, path: Path | str) -> None:
		"""Write to file (directory with JSON + NPY files).

		Creates:
		- clustering_meta.json: metadata (cls_values, linkage_method)
		- linkage.npy: numpy array for Python
		- linkage.json: JSON array for browser consumption
		"""
		path: Path = Path(path)
		path.mkdir(parents=True, exist_ok=True)

		# Save metadata as JSON
		meta: dict = dict(
			cls_values=self.cls_values,
			linkage_method=self.linkage_method,
		)
		meta_path: Path = path / "clustering_meta.json"
		with open(meta_path, "w") as f:
			json.dump(meta, f, indent=2)

		# Save linkage matrix as numpy (efficient for Python)
		npy_path: Path = path / "linkage.npy"
		np.save(npy_path, self.linkage_matrix)

		# Save linkage matrix as JSON (for browser)
		json_path: Path = path / "linkage.json"
		with open(json_path, "w") as f:
			json.dump(self.linkage_matrix.tolist(), f)

	@classmethod
	def read(cls, path: Path | str) -> "HierarchicalClusteringResult":
		"""Load from file (directory with JSON + NPY files)."""
		path: Path = Path(path)

		meta_path: Path = path / "clustering_meta.json"
		with open(meta_path, "r") as f:
			meta: dict = json.load(f)

		npy_path: Path = path / "linkage.npy"
		linkage_matrix: Float[np.ndarray, "n_merges 4"] = np.load(npy_path)

		return cls(
			cls_values=meta["cls_values"],
			linkage_matrix=linkage_matrix,
			linkage_method=meta["linkage_method"],
		)

	def get_clusters(
		self,
		n_clusters: int | None = None,
		cut_height: float | None = None,
	) -> dict[str, int]:
		"""Get cluster assignments for each head.

		Args:
			n_clusters: Number of clusters (takes precedence if both provided)
			cut_height: Height at which to cut the dendrogram

		Returns:
			Dict mapping head ID to cluster index (0-indexed)
		"""
		labels: Int[np.ndarray, " n_heads"]
		if n_clusters is not None:
			labels = fcluster(self.linkage_matrix, n_clusters, criterion="maxclust")
		elif cut_height is not None:
			labels = fcluster(self.linkage_matrix, cut_height, criterion="distance")
		else:
			raise ValueError("Must specify n_clusters or cut_height")

		# Convert to 0-indexed
		assignments: dict[str, int] = {
			cls: int(label - 1) for cls, label in zip(self.cls_values, labels)
		}
		return assignments

	def get_max_height(self) -> float:
		"""Get the maximum height in the dendrogram (root merge distance)."""
		max_height: float = float(self.linkage_matrix[-1, 2])
		return max_height

	@classmethod
	def from_distance_matrix(
		cls,
		distances: Float[np.ndarray, "n_heads n_heads"],
		cls_values: list[str],
		method: LinkageMethod = "average",
	) -> "HierarchicalClusteringResult":
		"""Compute hierarchical clustering from a distance matrix.

		Args:
			distances: Square distance matrix (n_heads x n_heads)
			cls_values: List of head IDs in the same order as the distance matrix
			method: Linkage method ("ward", "average", "complete", "single")

		Returns:
			HierarchicalClusteringResult with computed linkage matrix
		"""
		# Convert square distance matrix to condensed form for scipy
		# squareform expects a symmetric matrix with zeros on the diagonal
		condensed: Float[np.ndarray, " n_pairs"] = squareform(distances)

		# Compute linkage
		Z: Float[np.ndarray, "n_merges 4"] = linkage(condensed, method=method)

		return cls(
			cls_values=cls_values,
			linkage_matrix=Z,
			linkage_method=method,
		)
