from attn_embed.util.pipeline_cfg import PipelineConfig

from attn_embed.features.features import (
	scalar_feature_table,
	compute_scalar_features,
)

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
		verbose=cfg.verbose > 0,
	)
