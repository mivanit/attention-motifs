from transformer_lens import HookedTransformer

from attention_motifs.pipeline.cfg import (
	PipelineConfig,
	pipeline_step_major,
	pipeline_model_progress,
)


def download_models(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 0: download models")
	print(f"Using configuration:\n{cfg}")
	print(f"# Will download {len(cfg.models)} models: {cfg.models}")
	for idx, model_name in enumerate(cfg.models):
		pipeline_model_progress(idx, len(cfg.models), model_name)
		model = HookedTransformer.from_pretrained(model_name)
		del model  # Free memory after downloading


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	download_models(cfg)
