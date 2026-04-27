"""Pipeline step 0: download model weights from HuggingFace Hub.

Uses ``huggingface_hub.snapshot_download`` to cache model weights locally
without loading them into GPU memory.  The HF Hub client automatically
skips files that are already cached (etag-based check).
"""

import sys

from huggingface_hub import snapshot_download

# Import consts to load HF_TOKEN into os.environ before downloading gated models.
from attention_motifs.consts import HF_TOKEN as _HF_TOKEN

assert isinstance(_HF_TOKEN, str)

from attention_motifs.pipeline.cfg import (  # noqa: E402
	PipelineConfig,
	pipeline_model_progress,
	pipeline_step_major,
)
from attention_motifs.util.model_name import cached_get_hf_repo_id  # noqa: E402


def download_models(cfg: PipelineConfig) -> None:
	"""Download model weights for all models in the config.

	Uses ``huggingface_hub.snapshot_download`` which:
	- Caches to ``~/.cache/huggingface/hub/`` (standard HF cache)
	- Skips already-cached files based on etag
	- Supports gated models when ``HF_TOKEN`` is set
	"""
	pipeline_step_major("pipeline step 0: download models")
	print(f"Using configuration:\n{cfg}")
	print(f"# Will download {len(cfg.models)} models: {cfg.models}")

	token: str | None = _HF_TOKEN if _HF_TOKEN else None

	for idx, model_name in enumerate(cfg.models):
		pipeline_model_progress(idx, len(cfg.models), model_name)
		hf_repo_id: str = cached_get_hf_repo_id(model_name)
		print(f"  HF repo: {hf_repo_id}")
		try:
			cache_path: str = snapshot_download(
				repo_id=hf_repo_id,
				token=token,
			)
			print(f"  Cached at: {cache_path}")
		except Exception as exc:
			print(f"  WARNING: failed to download {hf_repo_id}: {exc}")
			print("  (The model may still be loadable if previously cached)")


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	download_models(cfg)
