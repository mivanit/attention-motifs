import attention_motifs.consts  # noqa: F401  # Load HF_TOKEN into os.environ early

import time
from pathlib import Path
from typing import Callable

from attention_motifs.pipeline.s1_activations import generate_activations
from attention_motifs.pipeline.s1b_render_patterns import render_patterns
from attention_motifs.pipeline.s1c_write_idxs import write_idxs
from attention_motifs.pipeline.s2_features import compute_features
from attention_motifs.pipeline.s3b_feat_fig import feat_figures
from attention_motifs.pipeline.s4_head_dist import head_dists
from attention_motifs.pipeline.s4b_write_frontend import write_frontend
from attention_motifs.pipeline.s4c_clustering import head_clustering
from attention_motifs.pipeline.s5_head_embed import head_embed
from attention_motifs.pipeline.s5b_head_embed_plots import head_embed_plots
from attention_motifs.pipeline.cfg import PipelineConfig
from attention_motifs.pipeline.s3_feat_proc import feat_proc
from attention_motifs.pipeline.state import PipelineState, StepName

# Step registry: maps step names to their functions
PIPELINE_STEPS: list[tuple[StepName, Callable[[PipelineConfig], None]]] = [
	("s1_activations", generate_activations),
	("s1b_render_patterns", render_patterns),
	("s1c_write_idxs", write_idxs),
	("s2_features", compute_features),
	("s3_feat_proc", feat_proc),
	("s3b_feat_fig", feat_figures),
	("s4_head_dist", head_dists),
	("s4b_write_frontend", write_frontend),
	("s4c_clustering", head_clustering),
	("s5_head_embed", head_embed),
	("s5b_head_embed_plots", head_embed_plots),
]


def _get_state_path(cfg: PipelineConfig) -> Path:
	"""Get the path to the pipeline state file."""
	return cfg.features_dir.parent / "pipeline_state.json"


def _load_or_create_state(cfg: PipelineConfig) -> PipelineState:
	"""Load existing state or create new one. Invalidate if config changed."""
	state_path: Path = _get_state_path(cfg)
	current_hash: str = cfg.compute_hash()

	if state_path.exists():
		state: PipelineState = PipelineState.read(state_path)
		if state.config_hash != current_hash:
			print(
				f"[smart] Config changed (hash: {current_hash[:8]}...), "
				f"invalidating all completed steps"
			)
			state = PipelineState(config_hash=current_hash)
		else:
			print(
				f"[smart] Config unchanged (hash: {current_hash[:8]}...), "
				f"{len(state.completed_steps)} steps already complete"
			)
	else:
		print("[smart] No previous state found, starting fresh")
		state = PipelineState(config_hash=current_hash)

	return state


def full_pipeline(cfg: PipelineConfig) -> None:
	"""Run the full attention motifs pipeline."""
	print(f"Running full pipeline with config:\n{cfg}")

	state: PipelineState | None = None
	if cfg.smart_mode:
		state = _load_or_create_state(cfg)

	step_name: StepName
	step_func: Callable[[PipelineConfig], None]
	for step_name, step_func in PIPELINE_STEPS:
		# Smart mode: check if step already complete
		if state is not None and state.is_step_complete(step_name):
			record = state.completed_steps[step_name]
			print(f"[smart] Skipping {step_name} (completed at {record.completed_at})")
			continue

		# Run the step
		start_time: float = time.time()
		step_func(cfg)
		duration: float = time.time() - start_time

		# Smart mode: record completion
		if state is not None:
			state.mark_step_complete(step_name, duration)
			state.save(_get_state_path(cfg))
			print(f"[smart] Saved state after {step_name}")


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	full_pipeline(cfg)
