import os

from pattern_lens.activations import activations_main

from attention_motifs.pipeline.cfg import (
	PipelineConfig,
	pipeline_step_major,
	pipeline_model_progress,
)


def generate_activations(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 1: generate activations")

	if cfg.parallel_models:
		_generate_activations_parallel(cfg)
	else:
		_generate_activations_sequential(cfg)


def _generate_activations_sequential(cfg: PipelineConfig) -> None:
	"""Original sequential path — one model at a time, in-process."""
	n_models: int = len(cfg.models)
	idx: int
	model_name: str
	for idx, model_name in enumerate(cfg.models):
		pipeline_model_progress(idx, n_models, model_name)
		activations_main(
			model_name=model_name,
			save_path=str(cfg.patterns_dir),
			prompts_path=str(cfg.prompts_file),
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


def _generate_activations_parallel(cfg: PipelineConfig) -> None:
	"""VRAM-aware parallel path — multiple models via subprocesses."""
	from attention_motifs.pipeline.model_table import (
		ModelInfo,
		fetch_model_table,
		get_model_params,
	)
	from attention_motifs.pipeline.model_scheduler import (
		ModelScheduler,
		ScheduledModel,
		estimate_vram_bytes,
	)

	model_table: dict[str, ModelInfo] = fetch_model_table()

	scheduled: list[ScheduledModel] = []
	skipped: list[str] = []
	for model_name in cfg.models:
		try:
			n_params: int = get_model_params(model_name, model_table)
		except KeyError:
			print(
				f"\033[93m[scheduler] warning: {model_name!r} not in model table, "
				f"will run sequentially as fallback\033[m"
			)
			skipped.append(model_name)
			continue
		estimated_vram: int = estimate_vram_bytes(n_params, cfg.vram_safety_factor)
		scheduled.append(
			ScheduledModel(
				name=model_name,
				n_params=n_params,
				estimated_vram=estimated_vram,
			)
		)

	# run schedulable models in parallel
	if scheduled:
		scheduler: ModelScheduler = ModelScheduler(
			models=scheduled,
			devices=cfg.devices,
			prompts_path=str(cfg.prompts_file),
			save_path=str(cfg.patterns_dir),
			n_samples=cfg.prompts_n_samples,
			min_chars=cfg.prompts_min_chars,
			max_chars=cfg.prompts_max_chars,
			force=cfg.force_overwrite,
			total_cpu_cores=os.cpu_count(),
		)
		scheduler.run_all()

	# fall back to sequential for models not in the table
	if skipped:
		print(
			f"\033[93m[scheduler] running {len(skipped)} models sequentially "
			f"(not in model table)\033[m"
		)
		for model_name in skipped:
			activations_main(
				model_name=model_name,
				save_path=str(cfg.patterns_dir),
				prompts_path=str(cfg.prompts_file),
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
