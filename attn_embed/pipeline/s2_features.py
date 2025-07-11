from attn_embed.pipeline.cfg import PipelineConfig, pipeline_step_major

from attn_embed.features.features import (
	scalar_feature_table,
	compute_scalar_features,
)


def compute_features(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 2: compute attention features")
	scalar_feature_table(
		features_func=compute_scalar_features,
		act_path=cfg.patterns_dir,
		models=cfg.models,
		out_path=cfg.data_path("raw"),
		processes=cfg.n_proc,
		verbose=cfg.verbose > 0,
	)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	compute_features(cfg)
