"""Pipeline step 7: Run ablation experiments on clustered heads.

Builds CandidateHeads from clustering results (produced by s4c),
filters to pipeline models, and runs evaluate_induction_scores()
per cluster. Writes per-model JSON results and ablation frontend.

Clusters can be specified by:
- ``heads``: list of head IDs (e.g. ``["gpt2-small:L5:H5"]``); each head's
  cluster is looked up automatically
- ``cluster_ids``: explicit cluster indices
- Neither: ablate all clusters at the given cut height

Skipped automatically if [ablation] section is absent or empty in
the pipeline config, or if cut params are not set.
"""

import json
import sys
from pathlib import Path
from typing import Any

from attention_motifs.ablation.ablate import AblationMethod
from attention_motifs.ablation.candidates import CandidateHeads
from attention_motifs.ablation.experiment import (
	AblationConfig,
	AblationResults,
	evaluate_induction_scores,
)
from attention_motifs.features.clustering import HierarchicalClusteringResult
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


# AblationConfig field names (for cherry-picking from TOML dict)
_ABLATION_CONFIG_KEYS: set[str] = {
	"n_sequences",
	"seq_length",
	"n_repetitions",
	"n_calibration_prompts",
	"seed",
	"micro_batch_size",
	"ablation_methods",
	"icl_prompts_file",
	"n_icl_prompts",
}


def _build_ablation_config(ablation_dict: dict[str, Any]) -> AblationConfig:
	"""Build AblationConfig from the [ablation] TOML section.

	Only passes through keys that AblationConfig accepts,
	ignoring pipeline-level keys like cluster_ids, cut_height, etc.
	"""
	config_kwargs: dict[str, Any] = {
		k: v for k, v in ablation_dict.items() if k in _ABLATION_CONFIG_KEYS
	}
	# Handle ablation_methods string→enum conversion
	if "ablation_methods" in config_kwargs:
		config_kwargs["ablation_methods"] = [
			AblationMethod(m) for m in config_kwargs["ablation_methods"]
		]
	return AblationConfig(**config_kwargs)


def _cluster_dir_name(
	cut_height: float | None,
	n_clusters: int | None,
	cluster_id: int,
) -> str:
	"""Build directory name encoding the cut parameters and cluster ID.

	Examples: ``"h5.00_c3"``, ``"n20_c3"``
	"""
	if cut_height is not None:
		return f"h{cut_height:.2f}_c{cluster_id}"
	else:
		return f"n{n_clusters}_c{cluster_id}"


def _write_cluster_info(
	output_dir: Path,
	cluster_id: int,
	candidates: CandidateHeads,
	cut_height: float | None,
	n_clusters: int | None,
	seed_heads: list[str] | None = None,
) -> None:
	"""Write cluster_info.json with metadata about the ablated cluster."""
	output_dir.mkdir(parents=True, exist_ok=True)

	info: dict[str, Any] = {
		"cluster_id": cluster_id,
		"cut_height": cut_height,
		"n_clusters": n_clusters,
		"total_heads": candidates.n_heads,
		"models": {
			model: [f"L{layer}:H{head}" for layer, head in heads]
			for model, heads in sorted(candidates.heads_by_model.items())
		},
	}
	if seed_heads is not None:
		info["seed_heads"] = seed_heads
	info_path: Path = output_dir / "cluster_info.json"
	info_path.write_text(json.dumps(info, indent=2))


def _resolve_cluster_ids(
	ablation_dict: dict[str, Any],
	assignments: dict[str, int],
) -> dict[int, list[str]]:
	"""Resolve which cluster IDs to ablate from config.

	Supports three modes (checked in order):
	1. ``heads``: look up each head's cluster in assignments
	2. ``cluster_ids``: use explicit cluster indices
	3. Neither: default to ``["gpt2-small:L5:H5"]``

	Returns
	-------
	dict[int, list[str]]
		Mapping of cluster_id → list of seed head IDs that selected it
		(empty list if selected via cluster_ids or all-clusters mode).
	"""
	heads: list[str] | None = ablation_dict.get("heads")
	cluster_ids_cfg: list[int] | None = ablation_dict.get("cluster_ids")

	if heads is not None:
		# Mode 1: look up cluster for each head
		cluster_to_seeds: dict[int, list[str]] = {}
		for head_id in heads:
			if head_id not in assignments:
				raise ValueError(
					f"Head '{head_id}' not found in clustering assignments. "
					f"Check that the head exists and the model is in the pipeline."
				)
			cid: int = assignments[head_id]
			cluster_to_seeds.setdefault(cid, []).append(head_id)
		return cluster_to_seeds

	elif cluster_ids_cfg is not None:
		# Mode 2: explicit cluster IDs
		return {cid: [] for cid in cluster_ids_cfg}

	else:
		# Mode 3: default to gpt2-small:L5:H5 (known induction head)
		default_heads: list[str] = ["gpt2-small:L5:H5"]
		cluster_to_seeds = {}
		for head_id in default_heads:
			if head_id not in assignments:
				raise ValueError(
					f"Default head '{head_id}' not found in clustering assignments. "
					f"Specify 'heads' or 'cluster_ids' explicitly in [ablation]."
				)
			cid = assignments[head_id]
			cluster_to_seeds.setdefault(cid, []).append(head_id)
		return cluster_to_seeds


def run_ablation(cfg: PipelineConfig) -> None:
	"""Run ablation experiments on specified clusters.

	Reads from ``cfg.ablation`` dict (``[ablation]`` TOML section):

	**Routing parameters** (consumed by this step):

	- ``cut_height`` (float): dendrogram cut height
	- ``n_clusters`` (int): number of clusters (alternative to cut_height)
	- ``heads`` (list[str]): head IDs to look up clusters for
	  (e.g. ``["gpt2-small:L5:H5"]``)
	- ``cluster_ids`` (list[int] | None): explicit cluster indices
	- ``output_dir`` (str): where to write results (default: ``"data/ablations"``)

	**Experiment parameters** (forwarded to AblationConfig):

	- ``n_sequences``, ``seq_length``, ``n_repetitions``, ``seed``,
	  ``micro_batch_size``, ``n_calibration_prompts``, ``ablation_methods``,
	  ``icl_prompts_file``, ``n_icl_prompts``
	"""
	pipeline_step_major("pipeline step 7: ablation experiments")

	ablation_dict: dict[str, Any] = cfg.ablation

	# --- Skip logic ---
	if not ablation_dict:
		print("[skip] s7_ablation: no [ablation] section in config")
		return

	cut_height: float | None = ablation_dict.get("cut_height")
	n_clusters: int | None = ablation_dict.get("n_clusters")
	if cut_height is None and n_clusters is None:
		raise ValueError("[ablation] must specify either 'cut_height' or 'n_clusters'")

	output_dir: Path = Path(ablation_dict.get("output_dir", "data/ablations"))

	# --- Load clustering and get assignments ---
	clustering_path: Path = cfg.data_path("clustering")
	clustering: HierarchicalClusteringResult = HierarchicalClusteringResult.read(
		clustering_path
	)
	assignments: dict[str, int] = clustering.get_clusters(
		cut_height=cut_height,
		n_clusters=n_clusters,
	)

	# --- Determine which clusters to run ---
	cluster_map: dict[int, list[str]] = _resolve_cluster_ids(ablation_dict, assignments)

	# --- Build AblationConfig ---
	config: AblationConfig = _build_ablation_config(ablation_dict)

	print(f"Ablation: {len(cluster_map)} cluster(s), output → {output_dir}")

	# --- Iterate clusters ---
	for cluster_id, seed_heads in sorted(cluster_map.items()):
		if seed_heads:
			print(
				f"\n--- Ablating cluster {cluster_id} "
				f"(from heads: {', '.join(seed_heads)}) ---"
			)
		else:
			print(f"\n--- Ablating cluster {cluster_id} ---")

		# Build candidates from assignments
		candidates: CandidateHeads = CandidateHeads._from_assignments(
			assignments, cluster_id
		)

		# Filter to pipeline models only
		candidates = candidates.filter_models(cfg.models)

		if candidates.n_heads == 0:
			print(f"  Cluster {cluster_id}: no heads in pipeline models, skipping")
			continue

		# Per-cluster output directory
		dir_name: str = _cluster_dir_name(cut_height, n_clusters, cluster_id)
		cluster_output: Path = output_dir / dir_name

		# Save cluster metadata
		_write_cluster_info(
			cluster_output,
			cluster_id,
			candidates,
			cut_height,
			n_clusters,
			seed_heads=seed_heads or None,
		)

		# Delegate to evaluate_induction_scores
		# (handles per-model results, resume, frontend writing)
		results: dict[str, AblationResults] = evaluate_induction_scores(
			candidates=candidates,
			config=config,
			device=cfg.device,
			output_dir=cluster_output,
			show_progress=cfg.verbose > 0,
		)

		print(f"  Cluster {cluster_id}: completed {len(results)} model(s)")


if __name__ == "__main__":
	pipeline_cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	run_ablation(pipeline_cfg)
