"""Clustering methods for attention head distances.

Provides hierarchical clustering (existing) plus HDBSCAN and Leiden
community detection as alternative methods that better respect manifold
structure.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


logger: logging.Logger = logging.getLogger(__name__)

LinkageMethod = Literal["ward", "average", "complete", "single"]
ClusteringMethod = Literal["hierarchical", "hdbscan", "leiden"]


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
		output_dir: Path = Path(path)
		output_dir.mkdir(parents=True, exist_ok=True)

		# Save metadata as JSON
		meta: dict = dict(
			cls_values=self.cls_values,
			linkage_method=self.linkage_method,
		)
		meta_path: Path = output_dir / "clustering_meta.json"
		with open(meta_path, "w") as f:
			json.dump(meta, f, indent=2)

		# Save linkage matrix as numpy (efficient for Python)
		npy_path: Path = output_dir / "linkage.npy"
		np.save(npy_path, self.linkage_matrix)

		# Save linkage matrix as JSON (for browser)
		json_path: Path = output_dir / "linkage.json"
		with open(json_path, "w") as f:
			json.dump(self.linkage_matrix.tolist(), f)

	@classmethod
	def read(cls, path: Path | str) -> "HierarchicalClusteringResult":
		"""Load from file (directory with JSON + NPY files)."""
		input_dir: Path = Path(path)

		meta_path: Path = input_dir / "clustering_meta.json"
		with open(meta_path, "r") as f:
			meta: dict = json.load(f)

		npy_path: Path = input_dir / "linkage.npy"
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

	def generate_default_labels(
		self,
		n_clusters_list: list[int],
	) -> dict[str, dict[str, dict[str, str | None | list[str]]]]:
		"""Generate default cluster labels with null name/desc for each K.

		For each K, computes the cut height from the linkage matrix
		and creates entries with null name, null desc, and the heads list.
		"""
		labels: dict[str, dict[str, dict[str, str | None | list[str]]]] = {}
		n: int = len(self.cls_values)
		for k in n_clusters_list:
			if k >= n:
				continue
			# Cut height for K clusters: midpoint between last-kept and first-skipped merge
			low: float = float(self.linkage_matrix[n - 1 - k, 2])
			high: float = float(self.linkage_matrix[n - k, 2])
			cut_height: float = (low + high) / 2
			height_key: str = f"{cut_height:.3f}"

			assignments: dict[str, int] = self.get_clusters(n_clusters=k)

			# Group heads by cluster
			clusters: dict[int, list[str]] = {}
			for head_id, cluster_id in assignments.items():
				clusters.setdefault(cluster_id, []).append(head_id)

			labels[height_key] = {
				str(cluster_id): {"name": None, "desc": None, "heads": sorted(heads)}
				for cluster_id, heads in sorted(clusters.items())
			}
		return labels

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


@dataclass
class FlatClusteringResult:
	"""Result of a flat (non-hierarchical) clustering method.

	Stores precomputed cluster assignments for multiple parameter values.
	Used by both HDBSCAN and Leiden community detection.
	"""

	cls_values: list[str]
	method: Literal["hdbscan", "leiden"]
	param_name: str  # "min_cluster_size" for HDBSCAN, "resolution" for Leiden
	param_keys: list[str]  # sorted parameter value keys
	# param_key -> {head_id -> cluster_id}
	partitions: dict[str, dict[str, int]]
	# param_key -> {n_clusters, n_outliers, ...}
	partition_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

	def get_clusters(self, param_key: str) -> dict[str, int]:
		"""Get cluster assignments for a specific parameter value.

		Args:
			param_key: String key for the parameter value (e.g. "10" or "0.500")

		Returns:
			Dict mapping head ID to cluster index (0-indexed, -1 for outliers)
		"""
		if param_key not in self.partitions:
			raise KeyError(
				f"Unknown param_key '{param_key}', available: {self.param_keys}"
			)
		return dict(self.partitions[param_key])

	def generate_default_labels(
		self,
	) -> dict[str, dict[str, dict[str, str | None | list[str]]]]:
		"""Generate default cluster labels with null name/desc for each parameter.

		Returns labels keyed by "{method}:{param_name}={value}" with the same
		structure as HierarchicalClusteringResult.generate_default_labels().
		"""
		labels: dict[str, dict[str, dict[str, str | None | list[str]]]] = {}
		for param_key in self.param_keys:
			label_key: str = f"{self.method}:{self.param_name}={param_key}"
			assignments: dict[str, int] = self.partitions[param_key]

			# Group heads by cluster
			clusters: dict[int, list[str]] = {}
			for head_id, cluster_id in assignments.items():
				clusters.setdefault(cluster_id, []).append(head_id)

			labels[label_key] = {
				str(cluster_id): {
					"name": None,
					"desc": None,
					"heads": sorted(heads),
				}
				for cluster_id, heads in sorted(clusters.items())
			}
		return labels

	def save(self, path: Path | str) -> None:
		"""Write to directory with JSON files.

		Creates:
		- clustering_meta.json: metadata (method, param_name, cls_values, param_keys, partition_meta)
		- partitions.json: all assignments for browser consumption
		"""
		output_dir: Path = Path(path)
		output_dir.mkdir(parents=True, exist_ok=True)

		meta: dict = dict(
			method=self.method,
			param_name=self.param_name,
			cls_values=self.cls_values,
			param_keys=self.param_keys,
			partition_meta=self.partition_meta,
		)
		meta_path: Path = output_dir / "clustering_meta.json"
		with open(meta_path, "w") as f:
			json.dump(meta, f, indent=2)

		partitions_path: Path = output_dir / "partitions.json"
		with open(partitions_path, "w") as f:
			json.dump(self.partitions, f)

	@classmethod
	def read(cls, path: Path | str) -> "FlatClusteringResult":
		"""Load from directory with JSON files."""
		input_dir: Path = Path(path)

		meta_path: Path = input_dir / "clustering_meta.json"
		with open(meta_path, "r") as f:
			meta: dict = json.load(f)

		partitions_path: Path = input_dir / "partitions.json"
		with open(partitions_path, "r") as f:
			partitions: dict[str, dict[str, int]] = json.load(f)

		return cls(
			cls_values=meta["cls_values"],
			method=meta["method"],
			param_name=meta["param_name"],
			param_keys=meta["param_keys"],
			partitions=partitions,
			partition_meta=meta.get("partition_meta", {}),
		)

	def serialize(self) -> dict:
		"""Convert to JSON-compatible dict."""
		return dict(
			cls_values=self.cls_values,
			method=self.method,
			param_name=self.param_name,
			param_keys=self.param_keys,
			partitions=self.partitions,
			partition_meta=self.partition_meta,
		)

	@classmethod
	def load(cls, data: dict) -> "FlatClusteringResult":
		"""Create instance from dict."""
		return cls(
			cls_values=data["cls_values"],
			method=data["method"],
			param_name=data["param_name"],
			param_keys=data["param_keys"],
			partitions=data["partitions"],
			partition_meta=data.get("partition_meta", {}),
		)


def _relabel_contiguous(labels: Int[np.ndarray, " n"]) -> dict[str, int]:
	"""Relabel cluster assignments to be contiguous 0-indexed, preserving -1 for outliers.

	Args:
		labels: Raw cluster labels (may have gaps, may use -1 for noise)

	Returns:
		Dict ready to zip with cls_values
	"""
	unique_labels: list[int] = sorted(set(int(x) for x in labels if x != -1))
	label_map: dict[int, int] = {old: new for new, old in enumerate(unique_labels)}
	label_map[-1] = -1
	return {str(i): label_map[int(labels[i])] for i in range(len(labels))}


def compute_hdbscan(
	distances: Float[np.ndarray, "n_heads n_heads"],
	cls_values: list[str],
	min_cluster_sizes: list[int],
) -> FlatClusteringResult:
	"""Compute HDBSCAN clustering at multiple min_cluster_size values.

	Args:
		distances: Square distance matrix (n_heads x n_heads)
		cls_values: List of head IDs in the same order as the distance matrix
		min_cluster_sizes: List of min_cluster_size values to compute

	Returns:
		FlatClusteringResult with partitions for each min_cluster_size
	"""
	from sklearn.cluster import HDBSCAN

	n_heads: int = len(cls_values)
	partitions: dict[str, dict[str, int]] = {}
	partition_meta: dict[str, dict[str, Any]] = {}

	for mcs in sorted(min_cluster_sizes):
		param_key: str = str(mcs)

		if mcs > n_heads // 2:
			logger.warning(
				f"HDBSCAN min_cluster_size={mcs} > n_heads/2={n_heads // 2}, skipping"
			)
			continue

		clusterer: HDBSCAN = HDBSCAN(
			min_cluster_size=mcs,
			metric="precomputed",
		)
		labels: Int[np.ndarray, " n_heads"] = clusterer.fit_predict(distances)

		# Build assignments dict
		assignments: dict[str, int] = {}
		unique_labels: list[int] = sorted(set(int(x) for x in labels if x != -1))
		label_map: dict[int, int] = {old: new for new, old in enumerate(unique_labels)}
		label_map[-1] = -1
		for i, cls_val in enumerate(cls_values):
			assignments[cls_val] = label_map[int(labels[i])]

		n_clusters: int = len(unique_labels)
		n_outliers: int = int(np.sum(labels == -1))

		partitions[param_key] = assignments
		partition_meta[param_key] = {
			"n_clusters": n_clusters,
			"n_outliers": n_outliers,
		}

		logger.info(
			f"HDBSCAN min_cluster_size={mcs}: "
			f"{n_clusters} clusters, {n_outliers} outliers"
		)

	param_keys: list[str] = sorted(partitions.keys(), key=lambda x: int(x))

	return FlatClusteringResult(
		cls_values=cls_values,
		method="hdbscan",
		param_name="min_cluster_size",
		param_keys=param_keys,
		partitions=partitions,
		partition_meta=partition_meta,
	)


def compute_leiden(
	distances: Float[np.ndarray, "n_heads n_heads"],
	cls_values: list[str],
	resolutions: list[float],
	n_neighbors: int = 10,
) -> FlatClusteringResult:
	"""Compute Leiden community detection at multiple resolution values.

	Builds a k-NN graph from the distance matrix with Gaussian kernel weights,
	then runs Leiden community detection at each resolution.

	Args:
		distances: Square distance matrix (n_heads x n_heads)
		cls_values: List of head IDs in the same order as the distance matrix
		resolutions: List of resolution values to compute
		n_neighbors: Number of nearest neighbors for the k-NN graph

	Returns:
		FlatClusteringResult with partitions for each resolution
	"""
	import igraph as ig

	n_heads: int = len(cls_values)
	k: int = min(n_neighbors, n_heads - 1)

	# Build mutual k-NN graph with Gaussian kernel weights
	# sigma = median of all k-NN distances for scale invariance
	knn_dists: list[float] = []
	neighbors: list[set[int]] = []
	for i in range(n_heads):
		sorted_indices: Int[np.ndarray, " n_heads"] = np.argsort(distances[i])
		# skip self (index 0 in sorted)
		nn: set[int] = set(int(x) for x in sorted_indices[1 : k + 1])
		neighbors.append(nn)
		for j in nn:
			knn_dists.append(float(distances[i, j]))

	sigma: float = float(np.median(knn_dists)) if knn_dists else 1.0
	if sigma < 1e-10:
		sigma = 1.0

	# Build edge list (mutual k-NN: edge if both are in each other's k-NN)
	edges: list[tuple[int, int]] = []
	weights: list[float] = []
	for i in range(n_heads):
		for j in neighbors[i]:
			if j > i and i in neighbors[j]:  # mutual
				w: float = float(np.exp(-(distances[i, j] ** 2) / (2 * sigma**2)))
				edges.append((i, j))
				weights.append(w)

	# Fallback: if mutual k-NN produces a disconnected or too-sparse graph,
	# use non-mutual k-NN
	if len(edges) < n_heads - 1:
		logger.info(
			f"Mutual k-NN graph too sparse ({len(edges)} edges), "
			f"falling back to non-mutual k-NN"
		)
		edges = []
		weights = []
		seen: set[tuple[int, int]] = set()
		for i in range(n_heads):
			for j in neighbors[i]:
				edge: tuple[int, int] = (min(i, j), max(i, j))
				if edge not in seen:
					seen.add(edge)
					w = float(np.exp(-(distances[i, j] ** 2) / (2 * sigma**2)))
					edges.append(edge)
					weights.append(w)

	graph: ig.Graph = ig.Graph(n=n_heads, edges=edges)
	graph.es["weight"] = weights

	partitions: dict[str, dict[str, int]] = {}
	partition_meta: dict[str, dict[str, Any]] = {}

	for res in sorted(resolutions):
		param_key: str = f"{res:.3f}"

		membership: ig.clustering.VertexClustering = graph.community_leiden(
			objective_function="modularity",
			weights="weight",
			resolution=res,
		)
		labels: list[int] = membership.membership

		# Build assignments dict (already 0-indexed from igraph)
		assignments: dict[str, int] = {}
		for i, cls_val in enumerate(cls_values):
			assignments[cls_val] = int(labels[i])

		n_clusters: int = len(set(labels))

		partitions[param_key] = assignments
		partition_meta[param_key] = {
			"n_clusters": n_clusters,
			"n_outliers": 0,
		}

		logger.info(f"Leiden resolution={res:.3f}: {n_clusters} clusters")

	param_keys: list[str] = sorted(partitions.keys(), key=lambda x: float(x))

	return FlatClusteringResult(
		cls_values=cls_values,
		method="leiden",
		param_name="resolution",
		param_keys=param_keys,
		partitions=partitions,
		partition_meta=partition_meta,
	)
