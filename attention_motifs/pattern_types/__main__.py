"""CLI for pattern types export.

Usage:
    python -m attention_motifs.pattern_types export \
        --clustering-path data/features/clustering \
        --cut-height 0.35 \
        --output attention_motifs/pattern_types/pattern_types.json
"""

import argparse
import sys
from pathlib import Path

from attention_motifs.features.clustering import HierarchicalClusteringResult
from attention_motifs.pattern_types.pattern_types import PatternTypes


def export_command(args: argparse.Namespace) -> None:
	"""Export clustering cut to pattern types JSON."""
	clustering_path: Path = Path(args.clustering_path)
	output_path: Path = Path(args.output)

	# Validate clustering path exists
	if not clustering_path.exists():
		print(f"Error: Clustering path does not exist: {clustering_path}", file=sys.stderr)
		sys.exit(1)

	# Load clustering
	print(f"Loading clustering from: {clustering_path}", file=sys.stderr)
	clustering: HierarchicalClusteringResult = HierarchicalClusteringResult.read(
		clustering_path
	)

	# Validate cut parameters
	if args.cut_height is None and args.n_clusters is None:
		print("Error: Must specify --cut-height or --n-clusters", file=sys.stderr)
		sys.exit(1)

	if args.cut_height is not None and args.n_clusters is not None:
		print("Warning: Both --cut-height and --n-clusters specified, using --n-clusters", file=sys.stderr)

	# Create pattern types
	pattern_types: PatternTypes = PatternTypes.from_clustering(
		clustering=clustering,
		cut_height=args.cut_height if args.n_clusters is None else None,
		n_clusters=args.n_clusters,
		clustering_path=str(clustering_path),
	)

	# Print stats
	print(f"Created {pattern_types.meta.n_clusters} clusters from {pattern_types.stats.n_heads} heads", file=sys.stderr)
	print("Cluster sizes:", file=sys.stderr)
	for cluster_id, size in sorted(pattern_types.stats.cluster_sizes.items(), key=lambda x: int(x[0])):
		print(f"  Cluster {cluster_id}: {size} heads", file=sys.stderr)

	# Save
	pattern_types.save(output_path)
	print(f"Saved to: {output_path}", file=sys.stderr)


def main() -> None:
	"""Main CLI entry point."""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Pattern types CLI for exporting clustering cuts"
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	# Export command
	export_parser: argparse.ArgumentParser = subparsers.add_parser(
		"export",
		help="Export clustering cut to pattern types JSON",
	)
	export_parser.add_argument(
		"--clustering-path",
		type=str,
		required=True,
		help="Path to clustering directory (contains linkage.npy, clustering_meta.json)",
	)
	export_parser.add_argument(
		"--cut-height",
		type=float,
		default=None,
		help="Height at which to cut the dendrogram",
	)
	export_parser.add_argument(
		"--n-clusters",
		type=int,
		default=None,
		help="Number of clusters (alternative to --cut-height)",
	)
	export_parser.add_argument(
		"--output",
		"-o",
		type=str,
		required=True,
		help="Output path for pattern types JSON",
	)
	export_parser.set_defaults(func=export_command)

	args: argparse.Namespace = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
