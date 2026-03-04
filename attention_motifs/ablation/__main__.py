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
"""

import argparse
import sys

from attention_motifs.ablation.candidates import CandidateHeads
from attention_motifs.ablation.experiment import (
	ExperimentConfig,
	evaluate_induction_scores,
)


def cluster_command(args: argparse.Namespace) -> None:
	"""Run ablation on all heads in a specific cluster."""
	# Build CandidateHeads from the appropriate source
	candidates: CandidateHeads
	if args.pattern_types is not None:
		candidates = CandidateHeads.from_pattern_types(
			args.pattern_types, args.cluster_id
		)
	elif args.clustering_path is not None:
		# Validate cut parameters when using raw clustering
		if args.cut_height is None and args.n_clusters is None:
			print(
				"Error: Must specify --cut-height or --n-clusters"
				" when using --clustering-path",
				file=sys.stderr,
			)
			sys.exit(1)
		candidates = CandidateHeads.from_clustering(
			args.clustering_path,
			args.cluster_id,
			cut_height=args.cut_height,
			n_clusters=args.n_clusters,
		)
	else:
		# Infer from pipeline config
		if args.cut_height is None and args.n_clusters is None:
			print(
				"Error: Must specify --cut-height or --n-clusters"
				" when inferring from pipeline config",
				file=sys.stderr,
			)
			sys.exit(1)
		candidates = CandidateHeads.from_pipeline_config(
			args.cluster_id,
			cut_height=args.cut_height,
			n_clusters=args.n_clusters,
			pipeline_cfg_path=args.pipeline_cfg,
		)

	# Filter to specific models if requested
	if args.models is not None:
		models: list[str] = [m.strip() for m in args.models.split(",") if m.strip()]
		candidates = candidates.filter_models(models)

	if candidates.n_heads == 0:
		print("Error: No heads found for the specified cluster", file=sys.stderr)
		sys.exit(1)

	# Build experiment config
	config: ExperimentConfig = ExperimentConfig(
		n_sequences=args.n_sequences,
		seq_length=args.seq_length,
		n_repetitions=args.n_repetitions,
		n_calibration_prompts=args.n_calibration_prompts,
		seed=args.seed,
	)

	results: dict = evaluate_induction_scores(
		candidates=candidates,
		config=config,
		device=args.device,
		output_dir=args.output_dir,
		show_progress=True,
	)

	# Print summary
	print(f"\nCompleted ablation for {len(results)} model(s)")
	for model_name, experiment_results in results.items():
		n_results: int = len(experiment_results.results)
		print(f"  {model_name}: {n_results} ablation results")


def main() -> None:
	"""Main CLI entry point."""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Ablation experiment CLI",
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	# cluster subcommand
	cluster_parser: argparse.ArgumentParser = subparsers.add_parser(
		"cluster",
		help="Run ablation on heads in a specific cluster",
	)

	# --- Input source (mutually exclusive) ---
	source_group = cluster_parser.add_mutually_exclusive_group()
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

	# --- Cluster selection ---
	cluster_parser.add_argument(
		"--cluster-id",
		type=int,
		required=True,
		help="Cluster index to ablate",
	)
	cluster_parser.add_argument(
		"--cut-height",
		type=float,
		default=None,
		help="Height at which to cut the dendrogram (for --clustering-path or pipeline config)",
	)
	cluster_parser.add_argument(
		"--n-clusters",
		type=int,
		default=None,
		help="Number of clusters (alternative to --cut-height)",
	)

	# --- Pipeline config fallback ---
	cluster_parser.add_argument(
		"--pipeline-cfg",
		type=str,
		default="pipeline_cfg.toml",
		help="Path to pipeline TOML config for inferring clustering path (default: pipeline_cfg.toml)",
	)

	# --- Filtering and runtime ---
	cluster_parser.add_argument(
		"--models",
		type=str,
		default=None,
		help="Comma-separated list of models to restrict to (default: all models in cluster)",
	)
	cluster_parser.add_argument(
		"--device",
		type=str,
		default="cuda",
		help="Device to run on (default: cuda)",
	)
	cluster_parser.add_argument(
		"--output-dir",
		type=str,
		default=None,
		help="Directory to save results (default: no save)",
	)

	# --- Experiment config ---
	cluster_parser.add_argument(
		"--n-sequences",
		type=int,
		default=100,
		help="Number of test sequences (default: 100)",
	)
	cluster_parser.add_argument(
		"--seq-length",
		type=int,
		default=25,
		help="Base pattern length (default: 25)",
	)
	cluster_parser.add_argument(
		"--n-repetitions",
		type=int,
		default=4,
		help="Number of pattern repetitions (default: 4)",
	)
	cluster_parser.add_argument(
		"--n-calibration-prompts",
		type=int,
		default=50,
		help="Number of prompts for mean ablation calibration (default: 50)",
	)
	cluster_parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed (default: 42)",
	)
	cluster_parser.set_defaults(func=cluster_command)

	args: argparse.Namespace = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
