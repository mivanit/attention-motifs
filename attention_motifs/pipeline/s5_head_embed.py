import polars as pl
import json
from pathlib import Path
from typing import Any

# attention-motifs
from attention_motifs.attnpedia.attnpedia import AttentionPedia
from attention_motifs.features.analysis import DistanceTensorResult
from attention_motifs.features.head_analysis import create_embedding_df_multi
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major
from attention_motifs.pipeline.model_table import ModelInfo, fetch_model_table
from attention_motifs.pipeline.s6b_cluster_trends import get_model_family


def get_embedding_prefixes(df: pl.DataFrame) -> list[str]:
	"""Extract all embedding prefixes from DataFrame columns."""
	prefixes = set()
	for col in df.columns:
		if col.startswith("embed.") and ".dim." in col:
			# Remove .dim.{n} suffix to get prefix
			prefix = col.rsplit(".dim.", 1)[0]
			prefixes.add(prefix)
	return sorted(prefixes)


def parse_prefix_info(prefix: str) -> dict[str, str]:
	"""Parse embedding prefix to extract method, dimensions, and neighbors.

	Args:
		prefix: Embedding prefix in format embed.{method}.d{n_components}.b{n_neighbors}

	Returns:
		Dictionary with method, n_components, and n_neighbors as strings

	Raises:
		ValueError: If prefix format is invalid
	"""
	# Format: embed.{method}.d{n_components}.b{n_neighbors}
	parts = prefix.split(".")
	if len(parts) != 4 or parts[0] != "embed":
		raise ValueError(f"Invalid prefix format: {prefix}")

	method = parts[1]
	n_components = parts[2][1:]  # Remove 'd' prefix
	n_neighbors = parts[3][1:]  # Remove 'b' prefix

	return {
		"method": method,
		"n_components": n_components,
		"n_neighbors": n_neighbors,
	}


def create_plots_metadata(prefixes: list[str]) -> dict[str, Any]:
	"""Create JSON metadata for embedding plots.

	Args:
		prefixes: List of embedding prefixes from DataFrame columns

	Returns:
		Dictionary containing plots metadata and summary statistics
	"""
	plots = []

	for prefix in prefixes:
		info = parse_prefix_info(prefix)

		# Create URL for 3D visualization with method-specific parameters
		embed_url = f"../vis/embeds/heads/index.html?axes.x=0&axes.y=1&axes.z=2&defaultColorColumn=type.group&numericalPrefix={prefix}.dim."

		plot_data = {
			"prefix": prefix,
			"method": info["method"],
			"n_components": info["n_components"],
			"n_neighbors": info["n_neighbors"],
			"svg_filename": f"head-embed-svgs/{prefix}.svg",
			"embed_url": embed_url,
		}

		plots.append(plot_data)

	return {
		"plots": plots,
		"metadata": {
			"total_plots": len(plots),
			"methods": list(set(plot["method"] for plot in plots)),
			"dimensions": list(set(plot["n_components"] for plot in plots)),
			"neighbor_counts": list(set(plot["n_neighbors"] for plot in plots)),
		},
	}


def add_model_metadata_columns(df: pl.DataFrame) -> pl.DataFrame:
	"""Add model_family, layer_depth, and model_size columns to a head embedding DataFrame.

	Drops existing metadata columns if present (idempotent), then adds fresh ones
	from the model table. Returns a new DataFrame with columns reordered:
	cls, model, layer, head, model_family, layer_depth, model_size, type.*, embed.*

	Args:
		df: DataFrame with at least 'model' and 'layer' columns.

	Returns:
		DataFrame with metadata columns added and reordered.
	"""
	# Drop existing metadata columns if present (makes this idempotent)
	existing_meta: list[str] = [
		c for c in ("model_family", "layer_depth", "model_size") if c in df.columns
	]
	if existing_meta:
		df = df.drop(existing_meta)

	model_table: dict[str, ModelInfo] = fetch_model_table()
	model_n_layers: dict[str, int] = {
		name: info.n_layers for name, info in model_table.items()
	}
	model_n_params: dict[str, int] = {
		name: info.n_params for name, info in model_table.items()
	}

	df = df.with_columns(
		pl.col("model")
		.map_elements(
			lambda m: get_model_family(m, except_on_missing=False), return_dtype=pl.Utf8
		)
		.alias("model_family"),
		(
			pl.col("layer")
			/ pl.col("model").replace_strict(
				model_n_layers, default=None, return_dtype=pl.Int64
			)
		).alias("layer_depth"),
		pl.col("model")
		.replace_strict(model_n_params, default=None, return_dtype=pl.Int64)
		.alias("model_size"),
	)

	# Reorder: put new columns right after "head", before "type.*"
	base_cols: list[str] = [
		"cls",
		"model",
		"layer",
		"head",
		"model_family",
		"layer_depth",
		"model_size",
	]
	type_cols: list[str] = [c for c in df.columns if c.startswith("type.")]
	embed_cols_ordered: list[str] = [
		c for c in df.columns if c.startswith("embed.")
	]
	return df.select(base_cols + type_cols + embed_cols_ordered)


def add_metadata_to_existing(path: str | Path) -> None:
	"""Load an existing head_embed.jsonl, add/refresh metadata columns, write back.

	This is a fast alternative to re-running the full s5 pipeline step when only
	the metadata columns (model_family, layer_depth, model_size) need updating.

	Args:
		path: Path to the head_embed.jsonl file.
	"""
	jsonl_path: Path = Path(path)
	df: pl.DataFrame = pl.read_ndjson(jsonl_path)
	print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols from {jsonl_path}")

	df = add_model_metadata_columns(df)
	df.write_ndjson(jsonl_path)
	print(f"Written {df.shape[0]} rows, {df.shape[1]} cols to {jsonl_path}")


def head_embed(cfg: PipelineConfig) -> None:
	"""Generate head embeddings from distance matrix.

	Args:
		cfg: Pipeline configuration containing embedding parameters
	"""
	pipeline_step_major("pipeline step 5: generate head embeddings")

	# Load head distances from previous step
	head_dists: DistanceTensorResult = DistanceTensorResult.read(
		cfg.data_path("head_dists_zanj")
	)

	# Load AttentionPedia for head type information
	attentionpedia: AttentionPedia = AttentionPedia()

	# Create embeddings using multiple methods and parameters from config
	head_embed_df: pl.DataFrame = create_embedding_df_multi(
		head_dists=head_dists,
		attnpedia=attentionpedia,
		embedding_methods=cfg.embedding_methods,
		n_components_list=cfg.embedding_n_components_list,
		n_neighbors_list=cfg.embedding_n_neighbors_list,
		match_model=None,  # Include all models
		save_path=None,  # We'll save manually to follow pipeline conventions
	)

	head_embed_df = add_model_metadata_columns(head_embed_df)

	# Save embeddings for frontend visualization
	head_embed_df.write_ndjson(cfg.data_path("head_embed"))

	# Generate plot metadata for figures (if figures enabled)
	if cfg.do_figures:
		assert cfg.figures_dir is not None

		prefixes = get_embedding_prefixes(head_embed_df)
		plots_metadata = create_plots_metadata(prefixes)

		# Save plot metadata as JSON
		plots_json_path = cfg.figures_dir / "head_embed_plots.json"
		cfg.figures_dir.mkdir(parents=True, exist_ok=True)

		with open(plots_json_path, "w") as f:
			json.dump(plots_metadata, f, indent=2)

		if cfg.verbose > 0:
			print(f"Plots metadata saved to: {plots_json_path}")
			print(f"Total embedding combinations: {len(prefixes)}")

	if cfg.verbose > 0:
		print(f"Generated head embeddings: {head_embed_df.shape}")

		# Extract method names from embedding column names
		embed_cols = [col for col in head_embed_df.columns if col.startswith("embed.")]
		methods = set()
		for col in embed_cols:
			# Column format: embed.{method}.d{n_components}.b{n_neighbors}.dim.{i}
			method = col.split(".")[1]
			methods.add(method)

		print(f"Embedding methods included: {sorted(list(methods))}")
		print(f"Models included: {sorted(head_embed_df['model'].unique().to_list())}")


if __name__ == "__main__":
	import sys

	if len(sys.argv) >= 2 and sys.argv[1] == "--add-metadata":
		# Fast path: just add metadata columns to existing JSONL
		if len(sys.argv) > 3:
			print(
				"Usage: python -m attention_motifs.pipeline.s5_head_embed --add-metadata [path]",
				file=sys.stderr,
			)
			sys.exit(1)
		path: str = sys.argv[2] if len(sys.argv) > 2 else "data/features/head_embed.jsonl"
		if path.startswith("-"):
			print(
				f"Error: expected a file path, got flag '{path}'\n"
				"Usage: python -m attention_motifs.pipeline.s5_head_embed --add-metadata [path]",
				file=sys.stderr,
			)
			sys.exit(1)
		add_metadata_to_existing(path)
	else:
		cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
		head_embed(cfg)
