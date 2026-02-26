"""Pipeline step 6: Create clustered head embeddings and write clustering frontend."""

import sys

import polars as pl

from attention_motifs.features.clustering import HierarchicalClusteringResult
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def clustered_embed(cfg: PipelineConfig) -> None:
	"""Create head_embed_clustered.jsonl and write clustering frontend.

	Loads the head embeddings from s5 and the clustering result from s4c,
	adds cluster assignment columns for each configured K, and saves
	a new JSONL file. Also copies the clustering frontend to vis/.
	"""
	pipeline_step_major("pipeline step 6: clustered head embeddings")

	# Load head embeddings from s5
	head_embed_df: pl.DataFrame = pl.read_ndjson(cfg.data_path("head_embed"))

	if cfg.verbose > 0:
		print(f"Loaded head embeddings: {head_embed_df.shape}")

	# Load clustering from s4c
	clustering: HierarchicalClusteringResult = HierarchicalClusteringResult.read(
		cfg.data_path("clustering")
	)

	if cfg.verbose > 0:
		print(
			f"Loaded clustering: {len(clustering.cls_values)} heads, "
			f"method={clustering.linkage_method}"
		)

	# Add cluster columns for each configured K
	cls_values: list[str] = head_embed_df["cls"].to_list()
	k: int
	for k in cfg.clustering_n_clusters_list:
		assignments: dict[str, int] = clustering.get_clusters(n_clusters=k)
		cluster_ids: list[int] = [assignments.get(cls, -1) for cls in cls_values]
		col_name: str = f"cluster.k{k}"
		head_embed_df = head_embed_df.with_columns(
			pl.Series(col_name, cluster_ids).cast(pl.Int32)
		)

		if cfg.verbose > 1:
			n_assigned: int = sum(1 for c in cluster_ids if c >= 0)
			print(f"  {col_name}: {n_assigned}/{len(cls_values)} heads assigned")

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
