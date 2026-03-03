"""CLI for ablation experiments.

Usage:
    python -m attention_motifs.ablation cluster \
        --clustering-path data/features/clustering \
        --cut-height 5.0 \
        --cluster-id 3 \
        --device cuda \
        --output-dir data/ablation/cluster_3
"""

import argparse
import sys

from attention_motifs.ablation.experiment import (
	ExperimentConfig,
	run_cluster_ablation,
)


def cluster_command(args: argparse.Namespace) -> None:
	"""Run ablation on all heads in a specific cluster."""
	# Validate cut parameters
	if args.cut_height is None and args.n_clusters is None:
		print("Error: Must specify --cut-height or --n-clusters", file=sys.stderr)
		sys.exit(1)

	# Parse models filter
	models: list[str] | None = None
	if args.models is not None:
		models = [m.strip() for m in args.models.split(",") if m.strip()]

	# Build experiment config
	config: ExperimentConfig = ExperimentConfig(
		n_sequences=args.n_sequences,
		seq_length=args.seq_length,
		n_repetitions=args.n_repetitions,
		n_calibration_prompts=args.n_calibration_prompts,
		seed=args.seed,
	)

	results: dict = run_cluster_ablation(
		clustering_path=args.clustering_path,
		cluster_id=args.cluster_id,
		cut_height=args.cut_height,
		n_clusters=args.n_clusters,
		config=config,
		models=models,
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
	cluster_parser.add_argument(
		"--clustering-path",
		type=str,
		required=True,
		help="Path to clustering directory (contains linkage.npy, clustering_meta.json)",
	)
	cluster_parser.add_argument(
		"--cluster-id",
		type=int,
		required=True,
		help="Cluster index (0-indexed) to ablate",
	)
	cluster_parser.add_argument(
		"--cut-height",
		type=float,
		default=None,
		help="Height at which to cut the dendrogram",
	)
	cluster_parser.add_argument(
		"--n-clusters",
		type=int,
		default=None,
		help="Number of clusters (alternative to --cut-height)",
	)
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
