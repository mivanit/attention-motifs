from pattern_lens.activations import activations_main
from attention_motifs.pipeline.cfg import (
	PipelineConfig,
	pipeline_step_major,
	pipeline_model_progress,
)


def generate_activations(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 1: generate activations")
	n_models: int = len(cfg.models)
	idx: int
	model_name: str
	for idx, model_name in enumerate(cfg.models):
		pipeline_model_progress(idx, n_models, model_name)
		activations_main(
			model_name=model_name,
			save_path=cfg.patterns_dir,
			prompts_path=cfg.prompts_file,
			raw_prompts=True,
			min_chars=cfg.prompts_min_chars,
			max_chars=cfg.prompts_max_chars,
			force=cfg.force_overwrite,
			n_samples=cfg.prompts_n_samples,
			no_index_html=False,
			shuffle=False,
			stacked_heads=False,
			device=cfg.device,
		)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	generate_activations(cfg)
