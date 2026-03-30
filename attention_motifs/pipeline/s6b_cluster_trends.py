"""Pipeline step 6b: Compute cluster trend data for visualization.

Reads clustered head embeddings and model metadata, then computes:
- Per (model, layer, cluster) fractions with normalized layer depth
- Per (model, cluster) fractions
- Per (model, layer) Shannon entropy of cluster distribution
Outputs a compact JSON file for the cluster_trends frontend.
Supports hierarchical, HDBSCAN, and Leiden clustering methods.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import polars as pl

from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def get_model_family(model_name: str, except_on_missing: bool = True) -> str:
	"""Extract model family from model name.

	Uses simple heuristics to group models into families.

	If `except_on_missing` is True (default), raises ValueError when no
	known family matches.  If False, returns ``"unknown"`` instead.
	"""
	if model_name.startswith("gpt2") or model_name == "distillgpt2":
		return "gpt2"
	if model_name.startswith("pythia"):
		return "pythia"
	if model_name.startswith("tiny-stories"):
		return "tiny-stories"
	if model_name.startswith("gemma-3"):
		return "gemma-3"
	if model_name.startswith("gemma"):
		return "gemma"
	if "llama" in model_name.lower():
		return "llama"
	if except_on_missing:
		raise ValueError(f"Unknown model family for {model_name!r}")
	return "unknown"


def _shannon_entropy(counts: list[int]) -> float:
	"""Compute Shannon entropy from a list of counts."""
	total: int = sum(counts)
	if total == 0:
		return 0.0
	entropy: float = 0.0
	c: int
	for c in counts:
		if c > 0:
			p: float = c / total
			entropy -= p * math.log2(p)
	return entropy


def _compute_trends_for_columns(
	df: pl.DataFrame,
	cluster_cols: list[str],
	models_meta: dict[str, dict[str, Any]],
) -> tuple[
	dict[str, list[dict[str, Any]]],
	dict[str, list[dict[str, Any]]],
	dict[str, list[dict[str, Any]]],
]:
	"""Compute by_layer, by_model, and entropy_by_layer trends for a set of cluster columns.

	Args:
		df: DataFrame with cluster columns
		cluster_cols: List of column names to compute trends for
		models_meta: Model metadata dict

	Returns:
		Tuple of (by_layer, by_model, entropy_by_layer) dicts keyed by column suffix
	"""
	by_layer: dict[str, list[dict[str, Any]]] = {}
	by_model: dict[str, list[dict[str, Any]]] = {}
	entropy_by_layer: dict[str, list[dict[str, Any]]] = {}

	col: str
	for col in cluster_cols:
		# Extract a short key from the column name (strip "cluster." prefix)
		key: str = col.removeprefix("cluster.")

		# --- by_layer: per (model, layer, cluster) ---
		layer_records: list[dict[str, Any]] = []
		entropy_records: list[dict[str, Any]] = []

		model_name: str
		for model_name in df["model"].unique().sort().to_list():
			meta: dict[str, Any] | None = models_meta.get(model_name)
			if meta is None:
				continue

			n_layers: int = meta["n_layers"]
			n_heads_per_layer: int = meta["n_heads"]
			model_df: pl.DataFrame = df.filter(pl.col("model") == model_name)

			layer_idx: int
			for layer_idx in range(n_layers):
				layer_df: pl.DataFrame = model_df.filter(pl.col("layer") == layer_idx)
				if layer_df.height == 0:
					continue

				depth: float = layer_idx / max(n_layers - 1, 1)

				# Count heads per cluster at this layer
				cluster_counts: dict[int, int] = {}
				cluster_id: int
				for cluster_id in layer_df[col].to_list():
					cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

				for cluster_id, count in cluster_counts.items():
					frac: float = count / n_heads_per_layer
					layer_records.append(
						{
							"model": model_name,
							"layer": layer_idx,
							"depth": round(depth, 4),
							"cluster": cluster_id,
							"count": count,
							"frac": round(frac, 4),
						}
					)

				# Shannon entropy at this layer
				counts_list: list[int] = list(cluster_counts.values())
				ent: float = _shannon_entropy(counts_list)
				entropy_records.append(
					{
						"model": model_name,
						"layer": layer_idx,
						"depth": round(depth, 4),
						"entropy": round(ent, 4),
					}
				)

		by_layer[key] = layer_records
		entropy_by_layer[key] = entropy_records

		# --- by_model: per (model, cluster) ---
		model_records: list[dict[str, Any]] = []
		for model_name in df["model"].unique().sort().to_list():
			meta = models_meta.get(model_name)
			if meta is None:
				continue

			model_df = df.filter(pl.col("model") == model_name)
			total_heads: int = model_df.height

			cluster_counts = {}
			for cluster_id in model_df[col].to_list():
				cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

			for cluster_id, count in cluster_counts.items():
				frac = count / total_heads
				model_records.append(
					{
						"model": model_name,
						"cluster": cluster_id,
						"count": count,
						"frac": round(frac, 4),
					}
				)

		by_model[key] = model_records

	return by_layer, by_model, entropy_by_layer


def cluster_trends(cfg: PipelineConfig) -> None:
	"""Compute cluster trend data and write cluster_trends frontend."""
	pipeline_step_major("pipeline step 6b: cluster trends")

	# Load clustered head embeddings from s6
	embed_path: Path = cfg.data_path("head_embed_clustered")
	df: pl.DataFrame = pl.read_ndjson(embed_path)

	if cfg.verbose > 0:
		print(f"Loaded clustered embeddings: {df.shape}")

	# Load model configs from models.jsonl
	models_jsonl_path: Path = cfg.patterns_dir / "models.jsonl"
	model_configs_raw: list[dict[str, Any]] = []
	with open(models_jsonl_path, "r") as f:
		line: str
		for line in f:
			line = line.strip()
			if line:
				model_configs_raw.append(json.loads(line))

	# Build model metadata dict
	models_meta: dict[str, dict[str, Any]] = {}
	mc: dict[str, Any]
	for mc in model_configs_raw:
		name: str = mc["model_name"]
		models_meta[name] = {
			"n_layers": mc["n_layers"],
			"n_heads": mc["n_heads"],
			"n_params": mc.get("n_params", 0),
			"family": get_model_family(name),
		}

	if cfg.verbose > 0:
		print(f"Loaded model metadata for {len(models_meta)} models")

	# --- Hierarchical trends (backward-compatible top-level keys) ---
	hierarchical_cols: list[str] = sorted(
		c for c in df.columns if c.startswith("cluster.k")
	)
	k_values: list[int] = sorted(
		int(c.removeprefix("cluster.k")) for c in hierarchical_cols
	)

	by_layer: dict[str, list[dict[str, Any]]]
	by_model: dict[str, list[dict[str, Any]]]
	entropy_by_layer: dict[str, list[dict[str, Any]]]

	if hierarchical_cols:
		by_layer, by_model, entropy_by_layer = _compute_trends_for_columns(
			df, hierarchical_cols, models_meta
		)
	else:
		by_layer, by_model, entropy_by_layer = {}, {}, {}

	# Assemble output (backward-compatible structure at top level)
	output: dict[str, Any] = {
		"models": models_meta,
		"methods": list(cfg.clustering_methods),
		"k_values": k_values,
		"by_layer": by_layer,
		"by_model": by_model,
		"entropy_by_layer": entropy_by_layer,
	}

	# --- HDBSCAN trends ---
	hdbscan_cols: list[str] = sorted(
		c for c in df.columns if c.startswith("cluster.hdbscan.")
	)
	if hdbscan_cols:
		h_by_layer, h_by_model, h_entropy = _compute_trends_for_columns(
			df, hdbscan_cols, models_meta
		)
		param_values: list[int] = sorted(
			int(c.removeprefix("cluster.hdbscan.mcs")) for c in hdbscan_cols
		)
		output["hdbscan"] = {
			"param_name": "min_cluster_size",
			"param_values": param_values,
			"by_layer": h_by_layer,
			"by_model": h_by_model,
			"entropy_by_layer": h_entropy,
		}

	# --- Leiden trends ---
	leiden_cols: list[str] = sorted(
		c for c in df.columns if c.startswith("cluster.leiden.")
	)
	if leiden_cols:
		l_by_layer, l_by_model, l_entropy = _compute_trends_for_columns(
			df, leiden_cols, models_meta
		)
		leiden_param_values: list[float] = sorted(
			float(c.removeprefix("cluster.leiden.r")) for c in leiden_cols
		)
		output["leiden"] = {
			"param_name": "resolution",
			"param_values": leiden_param_values,
			"by_layer": l_by_layer,
			"by_model": l_by_model,
			"entropy_by_layer": l_entropy,
		}

	if cfg.verbose > 0:
		print(
			f"Computed trends: hierarchical={len(hierarchical_cols)}, "
			f"hdbscan={len(hdbscan_cols)}, leiden={len(leiden_cols)} columns"
		)

	# Write JSON
	output_path: Path = cfg.data_path("cluster_trends")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(output, f)

	if cfg.verbose > 0:
		print(f"Wrote cluster trends to {output_path}")


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	cluster_trends(cfg)
