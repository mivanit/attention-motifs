"""Pipeline step 4c: Compute hierarchical clustering from head distances."""

import sys

from attention_motifs.features.analysis import DistanceTensorResult
from attention_motifs.features.clustering import HierarchicalClusteringResult
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def head_clustering(cfg: PipelineConfig) -> None:
	"""Compute hierarchical clustering from head distances.

	Loads the distance matrix from step 4 and computes hierarchical clustering
	using scipy's linkage function. The result is saved as both numpy and JSON
	for Python and browser consumption respectively.
	"""
	pipeline_step_major("pipeline step 4c: hierarchical clustering")

	# Load head distances
	head_dists: DistanceTensorResult = DistanceTensorResult.read_raw(
		cfg.data_path("head_dists_raw")
	)

	# Compute clustering
	clustering: HierarchicalClusteringResult = (
		HierarchicalClusteringResult.from_distance_matrix(
			distances=head_dists.mean_dists,
			cls_values=head_dists.cls_values,
			method=cfg.clustering_linkage_method,
		)
	)

	# Save results
	clustering.save(cfg.data_path("clustering"))

	if cfg.verbose > 0:
		n_heads: int = len(clustering.cls_values)
		print(f"Computed hierarchical clustering for {n_heads} heads")
		print(f"Linkage method: {clustering.linkage_method}")
		print(f"Linkage matrix shape: {clustering.linkage_matrix.shape}")
		print(f"Max dendrogram height: {clustering.get_max_height():.4f}")

		# Show example cluster assignments
		example_clusters: dict[str, int] = clustering.get_clusters(n_clusters=10)
		cluster_sizes: dict[int, int] = {}
		for cluster_id in example_clusters.values():
			cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
		print(f"Example with 10 clusters: {dict(sorted(cluster_sizes.items()))}")


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	head_clustering(cfg)
