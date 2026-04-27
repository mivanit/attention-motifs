"""Pipeline step 6: Create clustered head embeddings with all clustering methods."""

import sys
from pathlib import Path

import polars as pl

from attention_motifs.features.clustering import (
	FlatClusteringResult,
	HierarchicalClusteringResult,
)
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def clustered_embed(cfg: PipelineConfig) -> None:
	"""Create head_embed_clustered.jsonl with cluster columns from all methods.

	Loads the head embeddings from s5 and clustering results from s4c,
	adds cluster assignment columns for each configured method and parameter,
	and saves a new JSONL file.
	"""
	pipeline_step_major("pipeline step 6: clustered head embeddings")

	# Load head embeddings from s5
	head_embed_df: pl.DataFrame = pl.read_ndjson(cfg.data_path("head_embed"))

	if cfg.verbose > 0:
		print(f"Loaded head embeddings: {head_embed_df.shape}")

	cls_values: list[str] = head_embed_df["cls"].to_list()

	# --- Hierarchical clustering columns ---
	if "hierarchical" in cfg.clustering_methods:
		clustering_path: Path = cfg.data_path("clustering")
		if clustering_path.exists():
			clustering: HierarchicalClusteringResult = (
				HierarchicalClusteringResult.read(clustering_path)
			)

			if cfg.verbose > 0:
				print(
					f"Loaded hierarchical clustering: "
					f"{len(clustering.cls_values)} heads, "
					f"method={clustering.linkage_method}"
				)

			k: int
			for k in cfg.clustering_hierarchical_n_clusters_list:
				assignments: dict[str, int] = clustering.get_clusters(n_clusters=k)
				cluster_ids: list[int] = [
					assignments.get(cls, -1) for cls in cls_values
				]
				col_name: str = f"cluster.k{k}"
				head_embed_df = head_embed_df.with_columns(
					pl.Series(col_name, cluster_ids).cast(pl.Int32)
				)

	# --- HDBSCAN clustering columns ---
	if "hdbscan" in cfg.clustering_methods:
		hdbscan_path: Path = cfg.data_path("clustering_hdbscan")
		if hdbscan_path.exists():
			hdbscan_result: FlatClusteringResult = FlatClusteringResult.read(
				hdbscan_path
			)

			if cfg.verbose > 0:
				print(
					f"Loaded HDBSCAN clustering: "
					f"{len(hdbscan_result.param_keys)} parameter values"
				)

			for param_key in hdbscan_result.param_keys:
				partition: dict[str, int] = hdbscan_result.partitions[param_key]
				cluster_ids = [partition.get(cls, -1) for cls in cls_values]
				col_name = f"cluster.hdbscan.mcs{param_key}"
				head_embed_df = head_embed_df.with_columns(
					pl.Series(col_name, cluster_ids).cast(pl.Int32)
				)

	# --- Leiden clustering columns ---
	if "leiden" in cfg.clustering_methods:
		leiden_path: Path = cfg.data_path("clustering_leiden")
		if leiden_path.exists():
			leiden_result: FlatClusteringResult = FlatClusteringResult.read(leiden_path)

			if cfg.verbose > 0:
				print(
					f"Loaded Leiden clustering: "
					f"{len(leiden_result.param_keys)} parameter values"
				)

			for param_key in leiden_result.param_keys:
				partition = leiden_result.partitions[param_key]
				cluster_ids = [partition.get(cls, -1) for cls in cls_values]
				col_name = f"cluster.leiden.r{param_key}"
				head_embed_df = head_embed_df.with_columns(
					pl.Series(col_name, cluster_ids).cast(pl.Int32)
				)

	# Save clustered embeddings
	head_embed_df.write_ndjson(cfg.data_path("head_embed_clustered"))

	if cfg.verbose > 0:
		cluster_cols: list[str] = [
			c for c in head_embed_df.columns if c.startswith("cluster.")
		]
		print(
			f"Saved clustered embeddings: {head_embed_df.shape} "
			f"({len(cluster_cols)} cluster columns)"
		)


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	clustered_embed(cfg)
