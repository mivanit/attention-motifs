"""Verification: load each model into memory to confirm weights are valid.

This is the old ``s0_download_models`` behavior, preserved as an optional
verification step.  It actually instantiates ``HookedTransformer`` for each
model, which requires GPU memory and is slow.
"""

import gc
import sys

import torch
from pattern_lens.load_model import load_model
from transformer_lens import HookedTransformer

# Import consts to load HF_TOKEN into os.environ before loading gated models.
from attention_motifs.consts import HF_TOKEN as _HF_TOKEN

assert isinstance(_HF_TOKEN, str)

from attention_motifs.pipeline.cfg import (  # noqa: E402
	PipelineConfig,
	pipeline_model_progress,
	pipeline_step_major,
)


def test_model_load(cfg: PipelineConfig) -> None:
	"""Load each model into memory to verify weights are correct."""
	pipeline_step_major("test model load: loading models into memory")
	print(f"Using configuration:\n{cfg}")
	print(f"# Will load {len(cfg.models)} models: {cfg.models}")
	for idx, model_name in enumerate(cfg.models):
		pipeline_model_progress(idx, len(cfg.models), model_name)
		model: HookedTransformer = load_model(model_name)
		print(f"  Successfully loaded {model_name} ({model.cfg.n_params:,} params)")
		del model
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	print("All models loaded successfully!")


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	test_model_load(cfg)
