import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

# attention-motifs
from attention_motifs.features.head_analysis import (
	plot_head_embeddings_multi,
	plot_head_embeddings,
)
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def get_embedding_prefixes(df: pl.DataFrame) -> list[str]:
	"""Extract all embedding prefixes from DataFrame columns.

	Args:
		df: DataFrame with embedding columns in format embed.{method}.d{n_components}.b{n_neighbors}.dim.{i}

	Returns:
		Sorted list of unique embedding prefixes
	"""
	prefixes = set()
	for col in df.columns:
		if col.startswith("embed.") and ".dim." in col:
			# Remove .dim.{n} suffix to get prefix
			prefix = col.rsplit(".dim.", 1)[0]
			prefixes.add(prefix)
	return sorted(prefixes)


def head_embed_plots(cfg: PipelineConfig) -> None:
	"""Generate head embedding plots and HTML frontend.

	Args:
		cfg: Pipeline configuration with plotting and output settings
	"""
	pipeline_step_major("pipeline step 5b: generate head embedding plots")

	# Load head embedding data
	head_embed_df: pl.DataFrame = pl.read_ndjson(cfg.data_path("head_embed"))

	if cfg.verbose > 0:
		print(f"Loaded head embeddings: {head_embed_df.shape}")

	# Generate figures if enabled
	if cfg.do_figures:
		# Create subdirectory for SVGs
		head_embed_svgs_dir: Path = cfg.figures_dir / "head-embed-svgs"
		head_embed_svgs_dir.mkdir(parents=True, exist_ok=True)

		try:
			# Create multi-plot PDF
			print("Generating multi-plot PDF...")
			fig, _ = plot_head_embeddings_multi(
				head_embed_df,
				color_by="type.group",
				methods=cfg.embedding_methods[
					:3
				],  # Limit to first 3 methods to avoid overcrowding
				sizes=(12, 6),
				figsize=(24, 32),
			)

			# Save the multi-plot PDF
			fig.savefig(
				cfg.figure_path("head_embed"),
				bbox_inches="tight",
				pad_inches=0.1,
			)

			# Clean up
			plt.close(fig)

			# Generate individual SVG plots for each method/parameter combination
			print("Generating individual SVG plots...")
			prefixes = get_embedding_prefixes(head_embed_df)

			if cfg.verbose > 0:
				print(f"Generating {len(prefixes)} individual SVG plots...")

			for i, prefix in enumerate(prefixes):
				try:
					if cfg.verbose > 1:
						print(f"  [{i + 1}/{len(prefixes)}] Generating {prefix}")

					# Create individual plot
					fig, _ = plot_head_embeddings(
						head_embed_df,
						prefix=prefix,
						color_by="type.group",
						sizes=(12, 6),
						figsize=(8, 6),
						title=f"Head Embeddings: {prefix}",
					)

					# Save as SVG
					svg_path = head_embed_svgs_dir / f"{prefix}.svg"
					fig.savefig(
						svg_path,
						format="svg",
						bbox_inches="tight",
						pad_inches=0.1,
					)

					# Clean up
					plt.close(fig)

				except Exception as e:
					print(f"Warning: Could not generate plot for {prefix}: {e}")

			# Check that plots metadata exists (should be created by s5)
			plots_json_source = cfg.figures_dir / "head_embed_plots.json"
			if plots_json_source.exists():
				if cfg.verbose > 0:
					print(
						f"SVG plots generated successfully, metadata at: {plots_json_source}"
					)
					print("HTML frontend created by s4b_write_frontend.py")
			else:
				print(
					"Warning: head_embed_plots.json not found. Make sure s5_head_embed ran successfully."
				)

		except Exception as e:
			print(f"Error generating head embedding plots: {e}")
	else:
		print("Figures disabled, skipping plot generation")


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	head_embed_plots(cfg)
