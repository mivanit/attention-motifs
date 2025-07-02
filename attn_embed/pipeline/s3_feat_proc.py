
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from muutils.dbg import dbg, dbg_tensor
from sklearn.decomposition import PCA

from attn_embed.util.pipeline_cfg import PipelineConfig, pipeline_step_major


# attention-motifs
from attn_embed.features.analysis import (
	filter_data,
	normalize_data,
	null_stats,
	pca_importance_table,
)
from attn_embed.features.plotting import (
	apply_pca,
)


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
		data_scaled: pl.DataFrame = pl.read_ndjson(cfg.data_path("scaled"))

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


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	feat_proc(cfg)
