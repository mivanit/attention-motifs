import gc

import torch
from transformer_lens import HookedTransformer
from pattern_lens.load_model import load_model

# import consts to load HF_TOKEN into os.environ before downloading gated models
from attention_motifs.consts import HF_TOKEN as _HF_TOKEN

assert isinstance(_HF_TOKEN, str)

from attention_motifs.pipeline.cfg import (  # noqa: E402
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
		model: HookedTransformer = load_model(model_name)
		del model
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	download_models(cfg)
