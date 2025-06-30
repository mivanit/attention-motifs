from pathlib import Path

import matplotlib.pyplot as plt
import muutils.tensor_info
import numpy as np
import polars as pl
from jaxtyping import Float
from muutils.dbg import dbg_tensor
from sklearn.decomposition import PCA

from attn_embed.util.pipeline_cfg import PipelineConfig

from attn_embed.features.features import (
	scalar_feature_table,
	compute_scalar_features,
)


# attention-motifs
from attn_embed.features.analysis import (
	aggregate_feature_stats,
	filter_data,
	normalize_data,
	null_stats,
	pca_importance_table,
	plot_importance_covariance,
)
from attn_embed.features.plotting import (
	apply_pca,
	plot_correlation_matrix,
	plot_embedding,
)
from muutils.dbg import dbg

def main(cfg: PipelineConfig) -> None:
	# raw data from file instead of regenerating
	data_raw: pl.DataFrame = pl.read_ndjson(cfg.data_path("raw"))

	# this will filter all-nan rows
	data_filtered: pl.DataFrame = filter_data(data_raw)
	
	# dbg(null_stats(data_filtered))

	# normalize the data
	feature_cols: list[str] = [
		col for col in data_filtered.columns if col.startswith("feat.")
	]
	data_scaled: pl.DataFrame
	_data_norms: pl.DataFrame
	data_scaled, _data_norms = normalize_data(data_filtered, feature_cols)
	_data_norms.write_ndjson(cfg.data_path("norms"))
	data_scaled.write_ndjson(cfg.data_path("scaled"))
	
	# dbg(null_stats(data_scaled))
	# dbg_tensor(DATA_SCALED[FEATURE_COLS].to_numpy())

	meta_cols: list[str] = [
		col for col in data_scaled.columns if col.startswith("activation.")
	]
	pca_data: np.ndarray
	pca_obj: PCA
	pca_data, pca_obj = apply_pca(data_scaled, n_components=16, feature_cols=feature_cols)
	plt.savefig(cfg.figure_path("pca"), bbox_inches="tight", pad_inches=0.01)
	dbg_tensor(pca_data)
	dbg_tensor(pca_obj.components_)

	# pca and meta in one dataframe
	df_pca: pl.DataFrame = pl.concat(
		[
			# metadata -- "activation.*"
			data_scaled[meta_cols],
			# pca cols
			pl.DataFrame(pca_data, schema=[f"pc.{i}" for i in range(pca_data.shape[1])]),
		],
		how="horizontal",
	)
	df_pca.write_ndjson(cfg.data_path("pca"))

	df_importance: pl.DataFrame = pca_importance_table(
		pca_obj,
		feature_names=feature_cols,
	)
	df_importance.sort(pl.col("PC0").abs(), descending=True)



	display(aggregate_feature_stats(df_importance, side=1).sort("max", descending=True))


	cov_feats: list[str]
	cov_mat: Float[np.ndarray, "n_features n_features"]

	# full
	cov_feats, cov_mat = plot_importance_covariance(
		data_scaled,
		df_importance,
		# feature_order=sorted(FEATURE_COLS, key=lambda x: x.split(".")[-1]),
		metrics=["abs_max", "abs_mean"],
		descending=True,
		cmap="coolwarm",
		figsize=(20, 20),
		tick_pad=100,
		imp_legend_align=-0.15,
	)
	plt.savefig(cfg.figure_path("cov_full"), bbox_inches="tight", pad_inches=0.01)

	# reduced
	plot_importance_covariance(
		data_scaled,
		df_importance,
		# feature_order=sorted(FEATURE_COLS, key=lambda x: x.split(".")[-1]),
		metrics=["abs_max", "abs_mean"],
		descending=True,
		cmap="coolwarm",
		figsize=(10, 10),
		tick_pad=100,
		importance_threshold=0.2,
	)
	plt.savefig(
		cfg.figure_path("cov_reduced"),
		bbox_inches="tight",
		pad_inches=0.01,
	)


	n_dims: int = 5
	embed_fig, embed_ax = plt.subplots(
		n_dims - 1,
		n_dims - 1,
		figsize=(15, 15),
	)
	for i in range(n_dims - 1):
		for j in range(i + 1, n_dims):
			handles = plot_embedding(
				embedding=pca_data,
				labels=data_scaled["activation.model"],
				dims=(i, j),
				alpha=0.1,
				marker_size=1,
				title=f"({i}, {j})",
				ax=embed_ax[i, j - 1],
				do_legend=False,
			)
			embed_ax[i, j - 1].set_ylabel(None)

		for k in range(i):
			embed_ax[i, k].axis("off")

	plt.legend(
		handles=handles,
		loc="lower left",
		bbox_to_anchor=(-3, 1),
		title="Models",
		fontsize=16,
		title_fontsize=20,
	)
	# fig_pca_all_png: Path = PATH_FIGURES / "pca-all.png"
	# print(f"saving to {fig_pca_all_png}")
	# plt.savefig(fig_pca_all_png, dpi=250, bbox_inches="tight", pad_inches=0.01)
	fig_pca_all_jpg: Path = cfg.figure_path("pca_all")
	print(f"saving to {fig_pca_all_jpg}")
	plt.savefig(fig_pca_all_jpg, bbox_inches="tight", pad_inches=0.01, dpi=500)

if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	main(cfg)