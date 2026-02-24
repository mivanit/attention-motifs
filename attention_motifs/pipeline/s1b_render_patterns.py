"""Pipeline step 1b: render attention patterns from .npz activations to .png heatmaps.

Features:
- Deterministic random sampling of prompts (same subset across all models)
- Fault-tolerant: skips malformed npz files, processes the rest, reports failures
- Writes error files for easy cleanup of malformed activations
"""

import functools
import json
import multiprocessing
import random
import traceback
from pathlib import Path

from muutils.parallel import run_maybe_parallel
from pattern_lens.figure_util import AttentionMatrixFigureFunc
from pattern_lens.figures import (
	HTConfigMock,
	process_prompt,
	select_attn_figure_funcs,
)
from pattern_lens.indexes import (
	generate_functions_jsonl,
	generate_models_jsonl,
)
from pattern_lens.load_activations import augment_prompt_with_hash

from attention_motifs.figure_funcs import _ensure_register
from attention_motifs.pipeline.cfg import (
	PipelineConfig,
	pipeline_model_progress,
	pipeline_step_major,
)


def _safe_process_prompt(
	prompt: dict,
	model_cfg: HTConfigMock,
	save_path: Path,
	figure_funcs: list[AttentionMatrixFigureFunc],
	force_overwrite: bool,
) -> tuple[str, str, bool, str | None]:
	"""Wrapper around process_prompt that never raises.

	Returns:
		(prompt_hash, npz_path, success, error_string_or_none)
	"""
	augment_prompt_with_hash(prompt)
	prompt_hash: str = prompt["hash"]
	npz_path: str = str(
		save_path
		/ model_cfg.model_name
		/ "prompts"
		/ prompt_hash
		/ "activations.npz"
	)
	try:
		process_prompt(
			prompt=prompt,
			model_cfg=model_cfg,
			save_path=save_path,
			figure_funcs=figure_funcs,
			force_overwrite=force_overwrite,
		)
		return (prompt_hash, npz_path, True, None)
	except Exception as exc:  # noqa: BLE001
		tb: str = traceback.format_exc()
		error_str: str = f"{type(exc).__name__}: {exc}\n{tb}"
		return (prompt_hash, npz_path, False, error_str)


def _select_render_prompts(
	all_prompts: list[dict],
	render_n_samples: int | None,
	render_seed: int,
) -> list[dict]:
	"""Select prompts to render, with deterministic random sampling.

	Args:
		all_prompts: full list of prompts (already truncated to prompts_n_samples)
		render_n_samples: how many to render, or None for all
		render_seed: seed for reproducible random selection

	Returns:
		selected subset (or all if render_n_samples is None or >= len)
	"""
	if render_n_samples is None or render_n_samples >= len(all_prompts):
		return all_prompts
	rng: random.Random = random.Random(render_seed)
	selected: list[dict] = rng.sample(all_prompts, render_n_samples)
	return selected


def _write_rendered_prompts(
	patterns_dir: Path,
	selected_prompts: list[dict],
) -> Path:
	"""Write the list of selected prompts to rendered_prompts.jsonl."""
	out_path: Path = patterns_dir / "rendered_prompts.jsonl"
	with open(out_path, "w") as f:
		for prompt in selected_prompts:
			f.write(json.dumps(prompt) + "\n")
	return out_path


def _write_error_files(
	patterns_dir: Path,
	all_failures: list[tuple[str, str, str]],
) -> tuple[Path, Path]:
	"""Write malformed npz list and detailed error log.

	Args:
		patterns_dir: root patterns directory
		all_failures: list of (model_name, npz_path, error_string)

	Returns:
		(malformed_txt_path, error_log_path)
	"""
	txt_path: Path = patterns_dir / "render_malformed_npz.txt"
	log_path: Path = patterns_dir / "render_errors.log"

	with open(txt_path, "w") as f:
		for _model, npz_path, _error in all_failures:
			f.write(npz_path + "\n")

	with open(log_path, "w") as f:
		for model, npz_path, error in all_failures:
			f.write(f"[{model}] {npz_path}\n")
			# indent each line of the error for readability
			for line in error.splitlines():
				f.write(f"  {line}\n")
			f.write("\n")

	return txt_path, log_path


def _cleanup_stale_error_files(patterns_dir: Path) -> None:
	"""Remove error files from previous runs if they exist."""
	for fname in ("render_malformed_npz.txt", "render_errors.log"):
		p: Path = patterns_dir / fname
		if p.exists():
			p.unlink()


def _print_error_banner(
	n_failed: int,
	n_total: int,
	n_models_failed: int,
	txt_path: Path,
	log_path: Path,
) -> None:
	"""Print a bold red error banner to stderr-like stdout."""
	red: str = "\033[1;31m"
	reset: str = "\033[0m"
	bar: str = "=" * 80
	print(f"{red}{bar}")
	print(f"  ERROR: {n_failed}/{n_total} prompt renders failed across {n_models_failed} model(s)")
	print(f"  malformed npz list: {txt_path}")
	print(f"  detailed errors:    {log_path}")
	print()
	print("  To delete malformed files and regenerate:")
	print(f"    xargs rm < {txt_path}")
	print("    make am-pipeline  # re-run s1 to regenerate")
	print(f"{bar}{reset}")


def render_patterns(cfg: PipelineConfig) -> None:
	"""Render attention pattern PNGs from npz activations.

	Handles prompt sampling, fault tolerance, and error reporting.
	"""
	pipeline_step_major("pipeline step 1.b: render patterns")

	n_models: int = len(cfg.models)
	if n_models == 0:
		print("No models configured, skipping s1b")
		return

	# -- Load canonical prompt list from first model --
	first_model_path: Path = cfg.patterns_dir / cfg.models[0]
	prompts_jsonl: Path = first_model_path / "prompts.jsonl"
	with open(prompts_jsonl, "r") as f:
		all_prompts: list[dict] = [json.loads(line) for line in f]
	# truncate to prompts_n_samples (same truncation s1 and figures_main use)
	all_prompts = all_prompts[: cfg.prompts_n_samples]
	print(f"{len(all_prompts)} prompts loaded from {prompts_jsonl}")

	# -- Deterministic sampling (once for all models) --
	selected_prompts: list[dict] = _select_render_prompts(
		all_prompts, cfg.render_n_samples, cfg.render_seed
	)
	if cfg.render_n_samples is not None and cfg.render_n_samples < len(all_prompts):
		print(
			f"Randomly selected {len(selected_prompts)}/{len(all_prompts)} prompts "
			f"(seed={cfg.render_seed})"
		)

	rendered_path: Path = _write_rendered_prompts(cfg.patterns_dir, selected_prompts)
	print(f"Selected prompts written to {rendered_path}")

	# -- Resolve figure functions --
	figure_funcs: list[AttentionMatrixFigureFunc] = select_attn_figure_funcs({"attn"})
	print(
		f"{len(figure_funcs)} figure functions: "
		+ ", ".join(getattr(fn, "__name__", "<unknown>") for fn in figure_funcs)
	)

	# -- Process each model --
	all_failures: list[tuple[str, str, str]] = []  # (model, npz_path, error)
	total_prompts_across_models: int = 0
	models_with_failures: set[str] = set()

	chunksize: int = int(max(1, len(selected_prompts) // (5 * multiprocessing.cpu_count())))

	for idx, model in enumerate(cfg.models):
		pipeline_model_progress(idx, n_models, model)

		# load model config
		model_path: Path = cfg.patterns_dir / model
		with open(model_path / "model_cfg.json", "r") as f:
			model_cfg: HTConfigMock = HTConfigMock.load(json.load(f))

		# process prompts with error handling
		safe_func = functools.partial(
			_safe_process_prompt,
			model_cfg=model_cfg,
			save_path=cfg.patterns_dir,
			figure_funcs=figure_funcs,
			force_overwrite=cfg.force_overwrite,
		)

		results: list[tuple[str, str, bool, str | None]] = list(
			run_maybe_parallel(
				func=safe_func,
				iterable=selected_prompts,
				parallel=True,
				chunksize=chunksize,
				pbar="tqdm",
				pbar_kwargs=dict(
					desc=f"Rendering {model}",
					unit="prompt",
				),
			),
		)

		# tally results
		n_ok: int = sum(1 for _, _, ok, _ in results if ok)
		n_fail: int = sum(1 for _, _, ok, _ in results if not ok)
		total_prompts_across_models += len(results)

		if n_fail > 0:
			models_with_failures.add(model)
			for prompt_hash, npz_path, ok, error in results:
				if not ok:
					assert error is not None
					all_failures.append((model, npz_path, error))
			print(
				f"\033[33m  {model}: {n_ok}/{len(results)} OK, "
				f"{n_fail} failed\033[0m"
			)
		else:
			print(f"  {model}: {n_ok}/{len(results)} OK")

		# update index files
		generate_models_jsonl(cfg.patterns_dir)
		generate_functions_jsonl(cfg.patterns_dir)

	# -- Report results --
	if all_failures:
		txt_path: Path
		log_path: Path
		txt_path, log_path = _write_error_files(cfg.patterns_dir, all_failures)
		_print_error_banner(
			n_failed=len(all_failures),
			n_total=total_prompts_across_models,
			n_models_failed=len(models_with_failures),
			txt_path=txt_path,
			log_path=log_path,
		)
		raise SystemExit(1)
	else:
		_cleanup_stale_error_files(cfg.patterns_dir)
		print(
			f"\033[92mAll {total_prompts_across_models} prompt renders succeeded "
			f"across {n_models} model(s)\033[0m"
		)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	_ensure_register()
	render_patterns(cfg)
