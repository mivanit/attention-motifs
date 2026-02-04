from attention_motifs.figure_funcs import _ensure_register
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major, pipeline_model_progress
from pattern_lens.figures import figures_main


def render_patterns(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 1.b: render patterns")
	n_models: int = len(cfg.models)
	for idx, model in enumerate(cfg.models):
		pipeline_model_progress(idx, n_models, model)
		figures_main(
			model_name=model,
			save_path=cfg.patterns_dir,
			n_samples=cfg.prompts_n_samples,
			force=cfg.force_overwrite,
			figure_funcs_select={"attn"},
		)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	_ensure_register()
	render_patterns(cfg)
