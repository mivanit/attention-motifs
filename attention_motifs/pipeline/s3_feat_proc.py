import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from muutils.dbg import dbg, dbg_tensor
from sklearn.decomposition import PCA

from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


# attention-motifs
from attention_motifs.features.analysis import (
	filter_data,
	normalize_data,
	null_stats,
	pca_importance_table,
)
from attention_motifs.features.plotting import (
	apply_pca,
)


def _sample_prompts(
	df: pl.DataFrame,
	n_prompts: int,
	seed: int,
	prompt_col: str = "activation.prompt",
) -> pl.DataFrame:
	"""Deterministically sample a subset of prompts, keeping all rows for each sampled prompt.

	All models share the same prompt subset. Models with fewer prompts than
	`n_prompts` keep all their rows.
	"""
	unique_prompts: list[str] = sorted(df[prompt_col].unique().to_list())
	if len(unique_prompts) <= n_prompts:
		return df
	rng: random.Random = random.Random(seed)
	sampled: list[str] = rng.sample(unique_prompts, n_prompts)
	return df.filter(pl.col(prompt_col).is_in(sampled))


def compute_normalization(cfg: PipelineConfig) -> tuple[pl.DataFrame, list[str]]:
	# raw data from file
	data_raw: pl.DataFrame = pl.read_ndjson(cfg.data_path("raw"))

	# this will filter all-nan rows
	data_filtered: pl.DataFrame = filter_data(data_raw)

	if cfg.verbose > 1:
		dbg(null_stats(data_filtered))

	# normalize the data
	feature_cols: list[str] = [
		col for col in data_filtered.columns if col.startswith("feat.")
	]

	if cfg.verbose > 0:
		dbg_tensor(data_filtered[feature_cols].to_numpy())

	data_scaled: pl.DataFrame
	_data_norms: pl.DataFrame
	data_scaled, _data_norms = normalize_data(data_filtered, feature_cols)

	# save the normalized data, and what we normalized by
	_data_norms.write_ndjson(cfg.data_path("norms"))
	data_scaled.write_ndjson(cfg.data_path("scaled"))

	# print some debug information
	if cfg.verbose > 1:
		dbg(null_stats(data_scaled))
	if cfg.verbose > 0:
		dbg_tensor(data_scaled[feature_cols].to_numpy())

	return data_scaled, feature_cols


def compute_pca(
	cfg: PipelineConfig,
	data_scaled: pl.DataFrame,
	feature_cols: list[str],
) -> tuple[pl.DataFrame, np.ndarray]:
	if data_scaled is None:
		data_scaled = pl.read_ndjson(cfg.data_path("scaled"))

	meta_cols: list[str] = [
		col for col in data_scaled.columns if col.startswith("activation.")
	]

	# compute PCA
	pca_data: np.ndarray
	pca_obj: PCA
	pca_data, pca_obj = apply_pca(
		data_scaled,
		n_components=cfg.pca_n_components,
		feature_cols=feature_cols,
		plot_variance=cfg.do_figures,
	)
	# save raw PCA data
	np.save(
		cfg.data_path("pca_npy"),
		pca_data,
	)
	# PCA importance figure
	if cfg.do_figures:
		plt.savefig(cfg.figure_path("pca"), bbox_inches="tight", pad_inches=0.01)

	# debug printing
	if cfg.verbose > 0:
		dbg_tensor(pca_data)
		dbg_tensor(pca_obj.components_)

	# pca and meta in one dataframe
	df_pca: pl.DataFrame = pl.concat(
		[
			# metadata -- "activation.*"
			data_scaled[meta_cols],
			# pca cols
			pl.DataFrame(
				pca_data, schema=[f"pc.{i}" for i in range(pca_data.shape[1])]
			),
		],
		how="horizontal",
	)
	# save PCA as table
	df_pca.write_ndjson(cfg.data_path("pca"))
	df_pca.write_parquet(cfg.data_path("pca").with_suffix(".parquet"))
	# write a CSV version with less precision, for the web interface
	df_pca.write_csv(
		cfg.data_path("pca").with_suffix(".csv"),
		float_precision=6,
	)

	# write a reduced CSV for the web visualization (sampled prompts)
	df_pca_web: pl.DataFrame = df_pca
	if cfg.web_pca_n_prompts is not None:
		df_pca_web = _sample_prompts(
			df_pca,
			n_prompts=cfg.web_pca_n_prompts,
			seed=cfg.web_pca_seed,
		)
	df_pca_web.write_csv(
		cfg.data_path("pca_web"),
		float_precision=6,
	)

	# importance table
	df_importance: pl.DataFrame = pca_importance_table(
		pca_obj,
		feature_names=feature_cols,
	)
	# df_importance.sort(pl.col("PC0").abs(), descending=True)
	df_importance.write_ndjson(cfg.data_path("importance"))

	return df_importance, pca_data


def feat_proc(cfg: PipelineConfig) -> None:
	"""Main function to run the pipeline."""
	pipeline_step_major("pipeline step 3: process attention features")

	if cfg.do_figures:
		assert cfg.figures_dir is not None
		cfg.figures_dir.mkdir(parents=True, exist_ok=True)

	# compute normalization
	data_scaled: pl.DataFrame
	feature_cols: list[str]
	data_scaled, feature_cols = compute_normalization(cfg=cfg)

	# compute PCA
	compute_pca(
		cfg=cfg,
		data_scaled=data_scaled,
		feature_cols=feature_cols,
	)


def add_metadata_to_pattern_files(data_dir: str | Path) -> None:
	"""Add/refresh activation.model_family and activation.model_size on existing pattern files.

	Patches all pattern embedding data files in *data_dir* (JSONL, CSV, Parquet)
	without re-running the full s3 pipeline step.

	Args:
		data_dir: Directory containing the pattern embedding files
			(default ``data/features/``).
	"""
	from attention_motifs.features.feature_table import add_pattern_metadata_columns

	data_dir_path: Path = Path(data_dir)

	# Patch JSONL files
	for filename in ("raw.jsonl", "scaled.jsonl", "pca.jsonl"):
		path: Path = data_dir_path / filename
		if not path.exists():
			print(f"  Skipping {path} (not found)")
			continue
		df: pl.DataFrame = pl.read_ndjson(path)
		print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} cols from {path}")
		df = add_pattern_metadata_columns(df)
		df.write_ndjson(path)
		print(f"  Written {df.shape[0]} rows, {df.shape[1]} cols to {path}")

	# Patch CSV files
	for filename in ("pca.csv", "pca_web.csv"):
		path = data_dir_path / filename
		if not path.exists():
			print(f"  Skipping {path} (not found)")
			continue
		df = pl.read_csv(path)
		print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} cols from {path}")
		df = add_pattern_metadata_columns(df)
		df.write_csv(path, float_precision=6)
		print(f"  Written {df.shape[0]} rows, {df.shape[1]} cols to {path}")

	# Patch Parquet
	pca_parquet: Path = data_dir_path / "pca.parquet"
	if pca_parquet.exists():
		df = pl.read_parquet(pca_parquet)
		print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} cols from {pca_parquet}")
		df = add_pattern_metadata_columns(df)
		df.write_parquet(pca_parquet)
		print(f"  Written {df.shape[0]} rows, {df.shape[1]} cols to {pca_parquet}")
	else:
		print(f"  Skipping {pca_parquet} (not found)")


if __name__ == "__main__":
	import sys

	if len(sys.argv) >= 2 and sys.argv[1] == "--add-metadata":
		# Fast path: just add metadata columns to existing pattern files
		if len(sys.argv) > 3:
			print(
				"Usage: python -m attention_motifs.pipeline.s3_feat_proc --add-metadata [dir]",
				file=sys.stderr,
			)
			sys.exit(1)
		path_arg: str = sys.argv[2] if len(sys.argv) > 2 else "data/features/"
		if path_arg.startswith("-"):
			print(
				f"Error: expected a directory path, got flag '{path_arg}'\n"
				"Usage: python -m attention_motifs.pipeline.s3_feat_proc --add-metadata [dir]",
				file=sys.stderr,
			)
			sys.exit(1)
		add_metadata_to_pattern_files(path_arg)
	else:
		cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
		feat_proc(cfg)
