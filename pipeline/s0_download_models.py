from transformer_lens import HookedTransformer

from attn_embed.util.pipeline_cfg import PipelineConfig, pipeline_step_major


def download_models(cfg: PipelineConfig) -> None:
	print(f"# Will download {len(cfg.models)} models: {cfg.models}")
	for idx, model_name in enumerate(cfg.models):
		print(f"\t # Downloading model {idx + 1}/{len(cfg.models)}: {model_name}")
		model = HookedTransformer.from_pretrained(model_name)
		del model  # Free memory after downloading


if __name__ == "__main__":
	pipeline_step_major("pipeline step 0: download models")
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	download_models(cfg)
