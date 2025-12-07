from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from jaxtyping import Float

from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


# attention-motifs
from attention_motifs.features.analysis import (
	plot_importance_covariance,
)
from attention_motifs.features.plotting import (
	plot_embedding,
)


def plot_feat_covariance(
	cfg: PipelineConfig,
	data_scaled: pl.DataFrame,
	df_importance: pl.DataFrame,
) -> None:
	cov_feats: list[str]
	cov_mat: Float[np.ndarray, "n_features n_features"]

	# full covariance matrix
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


def plot_pca_all(
	cfg: PipelineConfig,
	data_scaled: pl.DataFrame,
	pca_data: np.ndarray,
) -> None:
	n_dims: int = cfg.plot_kwargs.get("n_dims", 5)
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
	plt.savefig(
		fig_pca_all_jpg,
		bbox_inches="tight",
		pad_inches=0.01,
		dpi=cfg.plot_kwargs.get("pca_all_dpi", 500),
	)


def feat_figures(cfg: PipelineConfig) -> None:
	"""Main function to run the pipeline."""
	# compute covariance
	if cfg.do_figures:
		data_scaled: pl.DataFrame = pl.read_ndjson(cfg.data_path("scaled"))
		df_importance: pl.DataFrame = pl.read_ndjson(cfg.data_path("importance"))
		pca_data: np.ndarray = np.load(cfg.data_path("pca_npy"))
		pipeline_step_major("pipeline step 3b: process attention features")
		plot_feat_covariance(
			cfg=cfg,
			data_scaled=data_scaled,
			df_importance=df_importance,
		)

		# plot PCA all
		plot_pca_all(
			cfg=cfg,
			data_scaled=data_scaled,
			pca_data=pca_data,
		)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	feat_figures(cfg)
