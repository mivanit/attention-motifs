from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major

from attention_motifs.features.feature_table import scalar_feature_table
from attention_motifs.features.features import compute_scalar_features


def compute_features(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 2: compute attention features")
	scalar_feature_table(
		features_func=compute_scalar_features,
		act_path=cfg.patterns_dir,
		models=cfg.models,
		out_path=cfg.data_path("raw"),
		processes=cfg.n_proc,
		chunksize=cfg.s2_chunksize,
		verbose=cfg.verbose > 0,
	)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	compute_features(cfg)
