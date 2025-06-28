from attn_embed.util.pipeline_cfg import PipelineConfig
from pattern_lens.figures import figures_main


def render_patterns(cfg: PipelineConfig) -> None:
	n_models: int = len(cfg.models)
	for idx, model in enumerate(cfg.models):
		print(f"processing model {idx + 1} / {n_models}: {model}")
		figures_main(
			model_name=model,
			save_path=cfg.patterns_dir,
			n_samples=cfg.prompts_n_samples,
			force=cfg.force_overwrite,
			figure_funcs_select=None,
		)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	print(f"Using configuration:\n{cfg}")
	render_patterns(cfg)
