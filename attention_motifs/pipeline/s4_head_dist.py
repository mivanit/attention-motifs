import polars as pl
from muutils.dbg import dbg_tensor

# attention-motifs
from attention_motifs.features.analysis import (
	DistanceTensorResult,
)

from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def head_dists(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 4: compute head distances")
	df_pca: pl.DataFrame = pl.read_ndjson(cfg.data_path("pca"))

	n_proc: int = cfg.s4_n_proc if cfg.s4_n_proc is not None else cfg.n_proc
	head_dists: DistanceTensorResult = DistanceTensorResult.build_distance_tensor(
		df_pca,
		feature_prefix="pc.",
		parallel=n_proc > 1,
		n_proc=n_proc,
	)
	dbg_tensor(head_dists.distances)
	head_dists.save(cfg.data_path("head_dists_zanj"))
	cfg.data_path("head_dists_raw").mkdir(parents=True, exist_ok=True)
	head_dists.save_raw(cfg.data_path("head_dists_raw"))

	if cfg.do_figures:
		fig = head_dists.plot_heatmap(show=False)
		fig.savefig(
			cfg.figure_path("head_dists"),
			bbox_inches="tight",
			pad_inches=0.1,
		)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	head_dists(cfg)
