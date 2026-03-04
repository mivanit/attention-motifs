"""CLI for ablation experiments.

Usage:
    # From pattern types file:
    python -m attention_motifs.ablation cluster \
        --pattern-types pattern_types.json --cluster-id 3

    # From explicit clustering path:
    python -m attention_motifs.ablation cluster \
        --clustering-path data/features/clustering \
        --cut-height 5.0 --cluster-id 3

    # Infer clustering path from pipeline config:
    python -m attention_motifs.ablation cluster \
        --cut-height 5.0 --cluster-id 3

    # Find a head's cluster and ablate all heads in it:
    python -m attention_motifs.ablation head \
        --head gpt2-small:L5:H5 --cut-height 5.0
"""

import argparse
import sys
from pathlib import Path

from attention_motifs.ablation.candidates import CandidateHeads
from attention_motifs.ablation.experiment import (
	ExperimentConfig,
	ExperimentResults,
	evaluate_induction_scores,
)


# ============================================================
# Shared argument helpers
# ============================================================


def _add_source_args(parser: argparse.ArgumentParser) -> None:
	"""Add input-source and dendrogram-cut arguments shared across subcommands."""
	source_group: argparse._MutuallyExclusiveGroup = (
		parser.add_mutually_exclusive_group()
	)
	source_group.add_argument(
		"--pattern-types",
		type=str,
		default=None,
		help="Path to pattern_types.json (pre-computed assignments)",
	)
	source_group.add_argument(
		"--clustering-path",
		type=str,
		default=None,
		help="Path to clustering directory (contains linkage.npy, clustering_meta.json)",
	)

	parser.add_argument(
		"--cut-height",
		type=float,
		default=None,
		help="Height at which to cut the dendrogram (for --clustering-path or pipeline config)",
	)
	parser.add_argument(
		"--n-clusters",
		type=int,
		default=None,
		help="Number of clusters (alternative to --cut-height)",
	)
	parser.add_argument(
		"--pipeline-cfg",
		type=str,
		default="pipeline_cfg.toml",
		help="Path to pipeline TOML config for inferring clustering path (default: pipeline_cfg.toml)",
	)


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
	"""Add runtime and experiment-config arguments shared across subcommands."""
	parser.add_argument(
		"--models",
		type=str,
		default=None,
		help="Comma-separated list of models to restrict to (default: all models in cluster)",
	)
	parser.add_argument(
		"--model-family",
		type=str,
		default=None,
		help="Comma-separated family prefixes to include (e.g. 'gpt2,pythia')",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda",
		help="Device to run on (default: cuda)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Directory to save results (default: no save)",
	)

	parser.add_argument(
		"--n-sequences",
		type=int,
		default=100,
		help="Number of test sequences (default: 100)",
	)
	parser.add_argument(
		"--seq-length",
		type=int,
		default=25,
		help="Base pattern length (default: 25)",
	)
	parser.add_argument(
		"--n-repetitions",
		type=int,
		default=4,
		help="Number of pattern repetitions (default: 4)",
	)
	parser.add_argument(
		"--n-calibration-prompts",
		type=int,
		default=50,
		help="Number of prompts for mean ablation calibration (default: 50)",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed (default: 42)",
	)


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
	"""Build ExperimentConfig from parsed CLI args."""
	return ExperimentConfig(
		n_sequences=args.n_sequences,
		seq_length=args.seq_length,
		n_repetitions=args.n_repetitions,
		n_calibration_prompts=args.n_calibration_prompts,
		seed=args.seed,
	)


def _run_and_print(
	candidates: CandidateHeads,
	args: argparse.Namespace,
) -> None:
	"""Filter candidates, run ablation, and print summary."""
	if args.models is not None:
		models: list[str] = [m.strip() for m in args.models.split(",") if m.strip()]
		candidates = candidates.filter_models(models)

	if args.model_family is not None:
		prefixes: list[str] = [
			p.strip() for p in args.model_family.split(",") if p.strip()
		]
		matching: list[str] = [
			m
			for m in candidates.models
			if any(p in m for p in prefixes)
		]
		candidates = candidates.filter_models(matching)

	if candidates.n_heads == 0:
		print("Error: No heads found for the specified cluster", file=sys.stderr)
		sys.exit(1)

	config: ExperimentConfig = _build_config(args)

	results: dict[str, ExperimentResults] = evaluate_induction_scores(
		candidates=candidates,
		config=config,
		device=args.device,
		output_dir=args.output_dir,
		show_progress=True,
	)

	print(f"\nCompleted ablation for {len(results)} model(s)")
	for model_name, experiment_results in results.items():
		n_results: int = len(experiment_results.results)
		print(f"  {model_name}: {n_results} ablation results")


def _require_cut_params(args: argparse.Namespace, context: str) -> None:
	"""Exit with error if neither --cut-height nor --n-clusters is set."""
	if args.cut_height is None and args.n_clusters is None:
		print(
			f"Error: Must specify --cut-height or --n-clusters {context}",
			file=sys.stderr,
		)
		sys.exit(1)


def _get_assignments(args: argparse.Namespace) -> dict[str, int]:
	"""Load clustering and return head_id -> cluster_id assignments.

	Works with --clustering-path or pipeline-config fallback.
	Exits with error if --pattern-types is used (not supported for lookup).
	"""
	from attention_motifs.features.clustering import HierarchicalClusteringResult
	from attention_motifs.pipeline.cfg import PipelineConfig

	if args.clustering_path is not None:
		_require_cut_params(args, "when using --clustering-path")
		clustering: HierarchicalClusteringResult = HierarchicalClusteringResult.read(
			args.clustering_path
		)
	else:
		_require_cut_params(args, "when inferring from pipeline config")
		cfg: PipelineConfig = PipelineConfig.read(Path(args.pipeline_cfg))
		clustering_path: Path = cfg.data_path("clustering")
		clustering = HierarchicalClusteringResult.read(clustering_path)

	assignments: dict[str, int] = clustering.get_clusters(
		n_clusters=args.n_clusters,
		cut_height=args.cut_height,
	)
	return assignments


# ============================================================
# Subcommands
# ============================================================


def cluster_command(args: argparse.Namespace) -> None:
	"""Run ablation on all heads in a specific cluster."""
	candidates: CandidateHeads
	if args.pattern_types is not None:
		candidates = CandidateHeads.from_pattern_types(
			args.pattern_types, args.cluster_id
		)
	elif args.clustering_path is not None:
		_require_cut_params(args, "when using --clustering-path")
		candidates = CandidateHeads.from_clustering(
			args.clustering_path,
			args.cluster_id,
			cut_height=args.cut_height,
			n_clusters=args.n_clusters,
		)
	else:
		_require_cut_params(args, "when inferring from pipeline config")
		candidates = CandidateHeads.from_pipeline_config(
			args.cluster_id,
			cut_height=args.cut_height,
			n_clusters=args.n_clusters,
			pipeline_cfg_path=args.pipeline_cfg,
		)

	_run_and_print(candidates, args)


def head_command(args: argparse.Namespace) -> None:
	"""Find which cluster a head belongs to, then ablate all heads in that cluster."""
	from attention_motifs.attnpedia import parse_cls

	head_id: str = args.head

	# Validate the head string parses correctly
	try:
		model_name: str
		layer: int
		head: int
		model_name, layer, head = parse_cls(head_id)
	except ValueError:
		print(
			f"Error: Invalid head format '{head_id}'. "
			"Expected 'model:L{{layer}}:H{{head}}' (e.g. 'gpt2-small:L5:H5')",
			file=sys.stderr,
		)
		sys.exit(1)

	if args.pattern_types is not None:
		# For pattern types: look up which type the head belongs to
		from attention_motifs.pattern_types.pattern_types import PatternTypes

		pt: PatternTypes = PatternTypes.read(args.pattern_types)
		cluster_id: int | None = pt.head_to_type.get(head_id)
		if cluster_id is None:
			print(
				f"Error: Head '{head_id}' not found in pattern types file",
				file=sys.stderr,
			)
			sys.exit(1)
		candidates: CandidateHeads = CandidateHeads.from_pattern_types(
			args.pattern_types, cluster_id
		)
	else:
		# For clustering: load assignments and look up cluster
		assignments: dict[str, int] = _get_assignments(args)
		if head_id not in assignments:
			print(
				f"Error: Head '{head_id}' not found in clustering assignments",
				file=sys.stderr,
			)
			sys.exit(1)
		cluster_id = assignments[head_id]
		candidates = CandidateHeads._from_assignments(assignments, cluster_id)

	print(f"Head '{head_id}' belongs to cluster {cluster_id}")
	_run_and_print(candidates, args)


# ============================================================
# CLI entry point
# ============================================================


def main() -> None:
	"""Main CLI entry point."""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Ablation experiment CLI",
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	# --- cluster subcommand ---
	cluster_parser: argparse.ArgumentParser = subparsers.add_parser(
		"cluster",
		help="Run ablation on heads in a specific cluster",
	)
	_add_source_args(cluster_parser)
	cluster_parser.add_argument(
		"--cluster-id",
		type=int,
		required=True,
		help="Cluster index to ablate",
	)
	_add_runtime_args(cluster_parser)
	cluster_parser.set_defaults(func=cluster_command)

	# --- head subcommand ---
	head_parser: argparse.ArgumentParser = subparsers.add_parser(
		"head",
		help="Find a head's cluster and ablate all heads in it",
	)
	_add_source_args(head_parser)
	head_parser.add_argument(
		"--head",
		type=str,
		required=True,
		help="Head identifier (e.g. 'gpt2-small:L5:H5')",
	)
	_add_runtime_args(head_parser)
	head_parser.set_defaults(func=head_command)

	args: argparse.Namespace = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
