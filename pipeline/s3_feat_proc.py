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



def filter_and_normalize(
	data: pl.DataFrame,
)


# raw data from file instead of regenerating
DATA_RAW: pl.DataFrame = pl.read_ndjson(PATH_BASE / "raw-m2-p16-c101.jsonl")

# this will filter all-nan rows
DATA_FILTERED: pl.DataFrame = filter_data(
	DATA_RAW,
	# tinystories and small pythia models look very different in embedding space, so we get rid of them
	remove_models=["tiny-stories-1M", "pythia-14m"],
)
null_stats(DATA_FILTERED)



FEATURE_COLS: list[str] = [
	col for col in DATA_FILTERED.columns if col.startswith("feat.")
]
DATA_SCALED: pl.DataFrame
_data_norms: pl.DataFrame
DATA_SCALED, _data_norms = normalize_data(DATA_FILTERED, FEATURE_COLS)
_data_norms.write_ndjson(PATH_BASE / "norms.jsonl")
DATA_SCALED.write_ndjson(PATH_BASE / "scaled.jsonl")
display(null_stats(DATA_SCALED))
META_COLS: list[str] = [
	col for col in DATA_SCALED.columns if col.startswith("activation.")
]

# dbg_tensor(DATA_SCALED[FEATURE_COLS].to_numpy())
PCA_DATA: np.ndarray
PCA_OBJ: PCA
PCA_DATA, PCA_OBJ = apply_pca(DATA_SCALED, n_components=16, feature_cols=FEATURE_COLS)
plt.savefig(PATH_FIGURES / "pca-variance.pdf", bbox_inches="tight", pad_inches=0.01)
dbg_tensor(PCA_DATA)
dbg_tensor(PCA_OBJ.components_)

# pca and meta in one dataframe
DF_PCA: pl.DataFrame = pl.concat(
	[
		# metadata -- "activation.*"
		DATA_SCALED[META_COLS],
		# pca cols
		pl.DataFrame(PCA_DATA, schema=[f"pc.{i}" for i in range(PCA_DATA.shape[1])]),
	],
	how="horizontal",
)
DF_PCA.write_ndjson(PATH_BASE / "pca.jsonl")

df_importance: pl.DataFrame = pca_importance_table(
	PCA_OBJ,
	feature_names=FEATURE_COLS,
)
df_importance.sort(pl.col("PC0").abs(), descending=True)



display(aggregate_feature_stats(df_importance, side=1).sort("max", descending=True))


COV_FEATS: list[str]
COV_MAT: Float[np.ndarray, "n_features n_features"]

# full
COV_FEATS, COV_MAT = plot_importance_covariance(
	DATA_SCALED,
	df_importance,
	# feature_order=sorted(FEATURE_COLS, key=lambda x: x.split(".")[-1]),
	metrics=["abs_max", "abs_mean"],
	descending=True,
	cmap="coolwarm",
	figsize=(20, 20),
	tick_pad=100,
	imp_legend_align=-0.15,
)
plt.savefig(PATH_FIGURES / "feature-cov-full.pdf", bbox_inches="tight", pad_inches=0.01)

# reduced
plot_importance_covariance(
	DATA_SCALED,
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
	PATH_FIGURES / "feature-cov-reduced.pdf",
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
			embedding=PCA_DATA,
			labels=DATA_SCALED["activation.model"],
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
fig_pca_all_png: Path = PATH_FIGURES / "pca-all.png"
print(f"saving to {fig_pca_all_png}")
plt.savefig(fig_pca_all_png, dpi=250, bbox_inches="tight", pad_inches=0.01)
fig_pca_all_jpg: Path = PATH_FIGURES / "pca-all.jpg"
print(f"saving to {fig_pca_all_jpg}")
plt.savefig(fig_pca_all_jpg, bbox_inches="tight", pad_inches=0.01, dpi=500)

if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	print(f"Using configuration:\n{cfg}")
	scalar_feature_table(
		features_func=compute_scalar_features,
		act_path=cfg.patterns_dir,
		models=cfg.models,
		out_path=cfg.features_dir,
		processes=cfg.n_proc,
	)