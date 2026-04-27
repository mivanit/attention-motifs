"""Pipeline step 4c: Compute clustering from head distances.

Supports hierarchical, HDBSCAN, and Leiden community detection methods.
Methods are selected via cfg.clustering_methods.
"""

import json
import logging
import sys
from pathlib import Path

from attention_motifs.features.analysis import DistanceTensorResult
from attention_motifs.features.clustering import (
	FlatClusteringResult,
	HierarchicalClusteringResult,
	compute_hdbscan,
	compute_leiden,
)
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major

logger: logging.Logger = logging.getLogger(__name__)


def _save_default_labels(result: FlatClusteringResult, output_dir: Path) -> None:
	"""Generate and save default cluster labels for a flat clustering result."""
	default_labels: dict = result.generate_default_labels()
	labels_path: Path = output_dir / "cluster_labels.json"
	with open(labels_path, "w") as f:
		json.dump(default_labels, f, indent=2)


def head_clustering(cfg: PipelineConfig) -> None:
	"""Compute clustering from head distances using configured methods.

	Loads the distance matrix from step 4 and computes clustering
	using each method in cfg.clustering_methods.
	"""
	pipeline_step_major("pipeline step 4c: clustering")

	# Load head distances
	head_dists: DistanceTensorResult = DistanceTensorResult.read_raw(
		cfg.data_path("head_dists_raw")
	)

	if cfg.verbose > 0:
		print(
			f"Loaded head distances: {len(head_dists.cls_values)} heads, "
			f"methods: {cfg.clustering_methods}"
		)

	# --- Hierarchical ---
	if "hierarchical" in cfg.clustering_methods:
		clustering: HierarchicalClusteringResult = (
			HierarchicalClusteringResult.from_distance_matrix(
				distances=head_dists.mean_dists,
				cls_values=head_dists.cls_values,
				method=cfg.clustering_hierarchical_linkage_method,
			)
		)
		clustering.save(cfg.data_path("clustering"))

		default_labels: dict = clustering.generate_default_labels(
			cfg.clustering_hierarchical_n_clusters_list
		)
		labels_path: Path = Path(cfg.data_path("clustering")) / "cluster_labels.json"
		with open(labels_path, "w") as f:
			json.dump(default_labels, f, indent=2)

		if cfg.verbose > 0:
			n_heads: int = len(clustering.cls_values)
			print(f"Hierarchical: {n_heads} heads, method={clustering.linkage_method}")
			print(f"  Max dendrogram height: {clustering.get_max_height():.4f}")

	# --- HDBSCAN ---
	if "hdbscan" in cfg.clustering_methods:
		try:
			hdbscan_result: FlatClusteringResult = compute_hdbscan(
				distances=head_dists.mean_dists,
				cls_values=head_dists.cls_values,
				min_cluster_sizes=cfg.clustering_hdbscan_min_cluster_sizes,
			)
			hdbscan_result.save(cfg.data_path("clustering_hdbscan"))
			_save_default_labels(
				hdbscan_result, Path(cfg.data_path("clustering_hdbscan"))
			)

			if cfg.verbose > 0:
				print(f"HDBSCAN: {len(hdbscan_result.param_keys)} parameter values")
				for pk in hdbscan_result.param_keys:
					meta: dict = hdbscan_result.partition_meta[pk]
					print(
						f"  min_cluster_size={pk}: "
						f"{meta['n_clusters']} clusters, {meta['n_outliers']} outliers"
					)
		except ImportError:
			logger.warning(
				"scikit-learn not installed, skipping HDBSCAN clustering. "
				"Install with: pip install scikit-learn>=1.3"
			)

	# --- Leiden ---
	if "leiden" in cfg.clustering_methods:
		try:
			leiden_result: FlatClusteringResult = compute_leiden(
				distances=head_dists.mean_dists,
				cls_values=head_dists.cls_values,
				resolutions=cfg.clustering_leiden_resolutions,
				n_neighbors=cfg.clustering_leiden_n_neighbors,
			)
			leiden_result.save(cfg.data_path("clustering_leiden"))
			_save_default_labels(
				leiden_result, Path(cfg.data_path("clustering_leiden"))
			)

			if cfg.verbose > 0:
				print(f"Leiden: {len(leiden_result.param_keys)} parameter values")
				for pk in leiden_result.param_keys:
					meta = leiden_result.partition_meta[pk]
					print(f"  resolution={pk}: {meta['n_clusters']} clusters")
		except ImportError:
			logger.warning(
				"igraph not installed, skipping Leiden clustering. "
				"Install with: pip install igraph>=0.11"
			)

	# Write methods manifest (so frontend knows what's available)
	methods_manifest: dict = {"methods": cfg.clustering_methods}
	manifest_path: Path = cfg.features_dir / "clustering_methods.json"
	with open(manifest_path, "w") as f:
		json.dump(methods_manifest, f, indent=2)


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	head_clustering(cfg)
