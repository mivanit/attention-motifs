"""Pipeline step 6b: Compute cluster trend data for visualization.

Reads clustered head embeddings and model metadata, then computes:
- Per (model, layer, cluster) fractions with normalized layer depth
- Per (model, cluster) fractions
- Per (model, layer) Shannon entropy of cluster distribution
Outputs a compact JSON file for the cluster_trends frontend.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import polars as pl

from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def get_model_family(model_name: str) -> str:
	"""Extract model family from model name.

	Uses simple heuristics to group models into families.
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
	# fallback: first segment before dash
	return model_name.split("-")[0]


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

	# Identify cluster columns and K values
	cluster_cols: list[str] = [c for c in df.columns if c.startswith("cluster.k")]
	k_values: list[int] = sorted(int(c.removeprefix("cluster.k")) for c in cluster_cols)

	if cfg.verbose > 0:
		print(f"K values: {k_values}")

	# Compute trends for each K
	by_layer: dict[str, list[dict[str, Any]]] = {}
	by_model: dict[str, list[dict[str, Any]]] = {}
	entropy_by_layer: dict[str, list[dict[str, Any]]] = {}

	k: int
	for k in k_values:
		col: str = f"cluster.k{k}"
		key: str = f"k{k}"

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

		if cfg.verbose > 1:
			print(
				f"  {key}: {len(layer_records)} layer records, "
				f"{len(model_records)} model records, "
				f"{len(entropy_records)} entropy records"
			)

	# Assemble output
	output: dict[str, Any] = {
		"models": models_meta,
		"k_values": k_values,
		"by_layer": by_layer,
		"by_model": by_model,
		"entropy_by_layer": entropy_by_layer,
	}

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
