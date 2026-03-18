import os

# HuggingFace's Rust tokenizers use internal thread pools. This module uses mp.Pool
# (fork-based), and forking a process with active threads risks deadlocks from
# inherited locked mutexes. Disabling tokenizer parallelism before import prevents
# the thread pool from ever being created.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from collections import Counter
from pathlib import Path
from typing import Callable
import functools
import multiprocessing as mp

import numpy as np
from jaxtyping import Float
import polars as pl
import tqdm

# custom utils
from muutils.collect_warnings import CollateWarnings
from muutils.spinner import SpinnerContext

# pattern_lens
from pattern_lens.consts import (
	SPINNER_KWARGS,
)
from pattern_lens.load_activations import load_activations
from pattern_lens.figures import HTConfigMock

from attention_motifs.util.model_name import cached_sanitize_model_name
from attention_motifs.util.util import prefix_dict

_SPINNER_INTERVAL: float = float(os.environ.get("SPINNER_UPDATE_INTERVAL", "0.1"))


_WarningCounts = Counter[tuple[str, int, str, str]]


def process_prompt(
	prompt: dict,
	model_name: str,
	save_path: Path,
	features_func: Callable[
		[Float[np.ndarray, "n_ctx n_ctx"]],
		dict[str, int | float],
	],
) -> tuple[list[dict[str, int | float | str]], _WarningCounts]:
	with CollateWarnings(print_on_exit=False) as cw:
		activations_path: Path
		cache: dict[str, Float[np.ndarray, "batch n_heads d_head"]]
		activations_path, cache = load_activations(
			model_name=model_name,
			prompt=prompt,
			save_path=save_path,
		)

		output: list[dict[str, int | float | str]] = list()

		for cache_key, head_batch in cache.items():
			layer_idx: int = int(cache_key.split(".")[1])
			for head_idx, A in enumerate(head_batch[0]):
				output.append(
					{
						**prefix_dict(
							dict(
								model=model_name,
								layer=str(layer_idx),
								cache_key=str(cache_key),
								head=int(head_idx),
								cls=f"{model_name}:L{layer_idx}:H{head_idx}",
								prompt=str(prompt["hash"]),
								n_ctx=int(A.shape[0]),
							),
							prefix="activation",
						),
						**prefix_dict(
							# returns dict[str, int|float], but type checker expects dict[str, int|float|str] (dict is invariant)
							features_func(A),  # type: ignore[arg-type]
							prefix="feat",
						),
					}
				)

	return output, cw.counts


def get_layer_depth(row: dict, model_configs: dict[str, HTConfigMock]) -> float:
	model: str = row["activation.model"]
	layer_idx: int = row["activation.layer"]
	model_n_layers: int = model_configs[model].n_layers
	return float(layer_idx) / float(model_n_layers - 1)


def _checkpoint_path(out_path: Path, model: str) -> Path:
	"""Per-model checkpoint file path for scalar_feature_table."""
	return out_path.parent / f"{out_path.stem}.checkpoint.{model}.jsonl"


def scalar_feature_table(
	features_func: Callable[
		[Float[np.ndarray, "n_ctx n_ctx"]],
		dict[str, float],
	],
	act_path: Path,
	out_path: Path,
	models: list[str] | None = None,
	processes: int | None = None,
	chunksize: int = 1,
	verbose: bool = True,
) -> pl.DataFrame:
	if models is None:
		models = [
			json.loads(cfg)["model_name"]
			for cfg in (act_path / "models.jsonl").read_text().splitlines()
		]
		for m in models:
			sanitized: str = cached_sanitize_model_name(m)
			if sanitized != m:
				print(
					f"\033[93m  WARNING: model name {m!r} from models.jsonl"
					f" is not sanitized (expected {sanitized!r})\033[m"
				)

	print_log = (lambda *args, **kwargs: print(*args, flush=True, **kwargs)) if verbose else lambda *args, **kwargs: None

	print_log(f"# models: {models}")

	out_path.parent.mkdir(parents=True, exist_ok=True)

	# --- Per-model processing with checkpointing ---
	failed_models: list[tuple[str, Exception]] = []
	for idx, model in enumerate(models):
		ckpt_path: Path = _checkpoint_path(out_path, model)

		if ckpt_path.exists():
			print_log(f"  # model '{model}': checkpoint exists, skipping")
			continue

		try:
			print_log(f"  # model: '{model}'")
			with SpinnerContext(
				message="setting up paths",
				update_interval=_SPINNER_INTERVAL,
				**SPINNER_KWARGS,
			):
				model_path: Path = act_path / model
				with open(model_path / "model_cfg.json", "r") as f:
					model_cfg: HTConfigMock = HTConfigMock.load(json.load(f))

			with SpinnerContext(
				message="loading prompts",
				update_interval=_SPINNER_INTERVAL,
				**SPINNER_KWARGS,
			):
				# load prompts
				with open(model_path / "prompts.jsonl", "r") as f:
					prompts: list[dict] = [json.loads(line) for line in f.readlines()]

			print_log(f"  # {len(prompts)} prompts loaded")

			processes = processes or mp.cpu_count()
			print_log(f"  # using {processes} processes, chunksize={chunksize}")
			with mp.Pool(processes=processes) as pool:
				# process each prompt in parallel
				prompt_func: Callable[
					[dict],
					tuple[list[dict[str, int | float | str]], _WarningCounts],
				] = functools.partial(
					process_prompt,
					model_name=model,
					save_path=act_path,
					features_func=features_func,
				)
				model_out = tqdm.tqdm(
					pool.imap(prompt_func, prompts, chunksize=chunksize),
					total=len(prompts),
				)
				model_rows: list[dict[str, int | float | str]] = []
				model_warnings: _WarningCounts = Counter()
				for rows, warn_counts in model_out:
					model_rows.extend(rows)
					model_warnings += warn_counts

			if model_warnings:
				for (
					filename,
					lineno,
					category,
					message,
				), count in model_warnings.items():
					print_log(
						f"  # ({count}x) {filename}:{lineno} {category}: {message}"
					)

			# Build per-model DataFrame with layer_depth
			df_model: pl.DataFrame = pl.DataFrame(model_rows)
			n_layers: int = model_cfg.n_layers
			df_model = df_model.with_columns(
				(
					pl.col("activation.layer").cast(pl.Float64) / float(n_layers - 1)
				).alias("activation.layer_depth")
			)

			df_model.write_ndjson(ckpt_path)
			print_log(f"  # checkpoint saved: {ckpt_path}")
		except Exception as e:
			print_log(f"  # ERROR: model '{model}' failed: {e}")
			failed_models.append((model, e))
			continue

	# --- Bail out if any models failed (don't combine partial results) ---
	if failed_models:
		failed_names: list[str] = [name for name, _ in failed_models]
		raise RuntimeError(
			f"s2 feature extraction failed for {len(failed_models)} model(s): {failed_names}. "
			f"Checkpoints for successful models are preserved; re-run to retry only the failed models."
		)

	# --- Concatenate all checkpoints into final output ---
	checkpoint_paths: list[Path] = [_checkpoint_path(out_path, m) for m in models]

	missing: list[str] = [m for m, p in zip(models, checkpoint_paths) if not p.exists()]
	if missing:
		raise FileNotFoundError(f"Missing checkpoint files for models: {missing}")

	print_log(f"# all {len(models)} model checkpoints found, concatenating into {out_path} ...")
	dfs: list[pl.DataFrame] = [pl.read_ndjson(p) for p in checkpoint_paths]
	df: pl.DataFrame = pl.concat(dfs)

	print_log(f"# output shape: {df.shape}")
	print_log(f"# saving to {out_path}")
	df.write_ndjson(out_path)

	return df
