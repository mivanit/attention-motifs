import gc
import os
import shutil

import torch
from pattern_lens.activations import activations_main

from attention_motifs.pipeline.cfg import (
	PipelineConfig,
	pipeline_step_major,
	pipeline_model_progress,
)
from attention_motifs.util.estimate_batch_size import main as estimate_batch_sizes


def _flush_gpu_memory() -> None:
	"""Force-free GPU memory between sequential model runs.

	HookedTransformer has circular references that prevent refcount-based
	deallocation, so ``gc.collect()`` is needed to trigger the cyclic GC.
	``torch.cuda.empty_cache()`` then returns freed blocks from PyTorch's
	caching allocator back to the CUDA driver.
	"""
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def _warn_batch_size(cfg: PipelineConfig) -> None:
	"""Print a prominent warning if configured batch_size may cause OOM."""
	try:
		report: dict = estimate_batch_sizes(
			models=cfg.models,
			device=cfg.device,
			max_seq_len=cfg.prompts_max_chars,
		)
	except Exception:
		return  # don't block the pipeline if estimation fails

	overall_max: int = report["overall_max_batch"]
	if cfg.batch_size <= overall_max:
		return

	# find the tightest model
	tightest_name: str = ""
	tightest_batch: int = overall_max
	name: str
	info: dict
	for name, info in report["models"].items():
		if "max_batch" in info and info["max_batch"] <= tightest_batch:
			tightest_name = name
			tightest_batch = info["max_batch"]

	suggested: int = report["overall_max_batch_pow2"]
	term_width: int = shutil.get_terminal_size((80, 20)).columns
	border: str = "!" * term_width
	print(f"\033[93m{border}\033[m")
	print(
		f"\033[93m  WARNING: batch_size={cfg.batch_size} exceeds estimated"
		f" max safe batch size={overall_max}"
		f" (pow2={suggested})\033[m"
	)
	print(
		f"\033[93m  tightest model: {tightest_name} (max_batch={tightest_batch})\033[m"
	)
	print(f"\033[93m  consider: --batch-size {suggested}\033[m")
	print(f"\033[93m{border}\033[m")


def generate_activations(cfg: PipelineConfig) -> None:
	pipeline_step_major("pipeline step 1: generate activations")
	_warn_batch_size(cfg)

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
			batch_size=cfg.batch_size,
			compress_level=cfg.compress_level,
		)
		_flush_gpu_memory()


def _generate_activations_parallel(cfg: PipelineConfig) -> None:
	"""VRAM-aware parallel path — multiple models via subprocesses."""
	from attention_motifs.pipeline.model_table import (
		MODEL_TABLE_CACHE,
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
				f"\033[93m[scheduler] could not find model {model_name!r} in cached model table "
				f"{MODEL_TABLE_CACHE}, will run serially\033[m"
			)
			skipped.append(model_name)
			continue
		estimated_vram: int = estimate_vram_bytes(
			n_params,
			cfg.vram_safety_factor,
			cuda_context_bytes=cfg.cuda_context_bytes,
		)
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
			batch_size=cfg.batch_size,
			compress_level=cfg.compress_level,
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
				batch_size=cfg.batch_size,
				compress_level=cfg.compress_level,
			)
			_flush_gpu_memory()


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	generate_activations(cfg)
