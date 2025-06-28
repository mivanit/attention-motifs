import itertools
import json
from pathlib import Path
from typing import Callable
import functools
import multiprocessing as mp


import torch
import numpy as np
from jaxtyping import Float
import polars as pl
import tqdm

# custom utils
from muutils.spinner import SpinnerContext

# pattern_lens
from pattern_lens.consts import (
	SPINNER_KWARGS,
)
from pattern_lens.load_activations import load_activations
from pattern_lens.figures import HTConfigMock

from attn_embed.util import prefix_dict
from attn_embed.util.bins import Bins
from attn_embed.features.vec_features import vec_features
from attn_embed.math.cos_sim import cosine_similarity_matrix
from attn_embed.math.math import skew_lt


def process_prompt(
	prompt: dict,
	model_name: str,
	save_path: Path,
	features_func: Callable[
		[Float[torch.Tensor, "n_ctx n_ctx"]],
		dict[str, float],
	],
) -> list[dict[str, int | float | str]]:
	activations_path, cache = load_activations(
		model_name=model_name,
		prompt=prompt,
		save_path=save_path,
		return_fmt="numpy",
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
							layer=layer_idx,
							cache_key=cache_key,
							head=head_idx,
							cls=f"{model_name}:L{layer_idx}:H{head_idx}",
							prompt=prompt["hash"],
							n_ctx=A.shape[0],
						),
						prefix="activation",
					),
					**prefix_dict(
						features_func(A),
						prefix="feat",
					),
				}
			)

	return output


def get_layer_depth(row: dict, model_configs: dict[str, HTConfigMock]) -> float:
	model: str = row["activation.model"]
	layer_idx: int = row["activation.layer"]
	model_n_layers: int = model_configs[model].n_layers
	return float(layer_idx) / float(model_n_layers - 1)


def scalar_feature_table(
	features_func: Callable[
		[Float[torch.Tensor, "n_ctx n_ctx"]],
		dict[str, float],
	],
	act_path: Path = Path("../docs/temp"),
	models: list[str] | None = None,
	out_path: Path = Path("data/features/"),
	processes: int | None = None,
	chunksize: int | None = None,
) -> pl.DataFrame:
	if models is None:
		models = [
			json.loads(cfg)["model_name"]
			for cfg in (act_path / "models.jsonl").read_text().splitlines()
		]

	print(f"models: {models}")

	output: list[dict[str, int | float | str]] = list()
	model_configs: dict[str, HTConfigMock] = dict()

	for idx, model in enumerate(models):
		print(f"model: '{model}'")
		with SpinnerContext(message="setting up paths", **SPINNER_KWARGS):
			model_path: Path = act_path / model
			with open(model_path / "model_cfg.json", "r") as f:
				model_cfg = HTConfigMock.load(json.load(f))
			model_configs[model] = model_cfg

		with SpinnerContext(message="loading prompts", **SPINNER_KWARGS):
			# load prompts
			with open(model_path / "prompts.jsonl", "r") as f:
				prompts: list[dict] = [json.loads(line) for line in f.readlines()]
			# truncate to n_samples
			prompts = prompts

		print(f"{len(prompts)} prompts loaded")

		# for prompt in tqdm.tqdm(prompts, desc="prompts", total=len(prompts)):
		processes = processes or mp.cpu_count()
		print(f"using {processes} processes")
		# chunksize = 1
		with mp.Pool(processes=processes) as pool:
			# process each prompt in parallel
			prompt_func: Callable[[dict], list[dict[str, int | float | str]]] = (
				functools.partial(
					process_prompt,
					model_name=model,
					save_path=act_path,
					features_func=features_func,
				)
			)
			model_out: list[dict] = tqdm.tqdm(
				pool.imap(prompt_func, prompts),
				total=len(prompts),
			)
			output.extend(itertools.chain.from_iterable(model_out))

	# turn everything into a DataFrame
	df: pl.DataFrame = pl.DataFrame(output)


	# add a activation.layer_depth column by applying get_layer_depth to each row
	df = df.with_columns(
		pl.col("activation.layer").apply(
			get_layer_depth,
			model_configs=model_configs,
			return_dtype=pl.Float64,
		).alias("activation.layer_depth"),
	)

	# n models, n prompts, n features
	out_fname: str = f"raw-m{len(models)}-p{len(prompts)}-c{len(df.columns)}.jsonl"
	print(f"output shape: {df.shape}")
	print(f"saving to {out_path / out_fname}")
	df.write_ndjson(out_path / out_fname)

	return df


def gram_features(A: Float[np.ndarray, "n_ctx n_ctx"]) -> dict[str, float]:
	# dbg_tensor(A)
	bins: Bins = Bins(n_bins=32, start=0.0, stop=1.0)
	x_hist, _ = np.histogram(A.flatten(), bins.edges, density=True)
	return prefix_dict(
		vec_features(x_hist),
		prefix="hist",
	)
	# TODO: mass as a function of distance from diagonal


def compute_scalar_features(
	A: Float[np.ndarray, "n_ctx n_ctx"],
) -> dict[str, float]:
	# dbg_tensor(A)
	A_log: Float[np.ndarray, "n_ctx n_ctx"] = np.nan_to_num(np.log(A + 1e-9), nan=-10)
	# dbg_tensor(A_log)

	A_skew: Float[np.ndarray, "n_ctx n_ctx"] = skew_lt(A)
	# dbg_tensor(A_skew)
	A_log_skew: Float[np.ndarray, "n_ctx n_ctx"] = skew_lt(A_log)

	return dict(
		# diagonal: standard features, fit diff to beta dist
		**prefix_dict(vec_features(A.diagonal(), reduced=False), prefix="diag"),
		# off-diagonal: standard features, fit diff to beta dist
		**prefix_dict(vec_features(A[:, 0], reduced=False), prefix="first_tok"),
		# transition tensor: standard features, standard features on diff, linear envelope on transition time
		# 	TODO: standard features on decay rate
		# markov transition not that important?
		# **prefix_dict(
		# 	tt_features(A),
		# 	prefix="markov_transition",
		# ),
		# # {log, raw} gram matrix of {rows, cols, rows of skewed}: beta fit hist
		# # 	TODO: fit fft in `gram_features`, but this is expensive
		**prefix_dict(
			gram_features(A @ A.T),
			prefix=["gram", "row"],
		),
		**prefix_dict(
			gram_features(A.T @ A),
			prefix=["gram", "col"],
		),
		**prefix_dict(
			gram_features(A_skew.T @ A_skew),
			prefix=["gram", "skew"],
		),
		**prefix_dict(
			gram_features(cosine_similarity_matrix(A_log)),
			prefix=["log", "gram", "row"],
		),
		**prefix_dict(
			gram_features(cosine_similarity_matrix(A_log, col=True)),
			prefix=["log", "gram", "col"],
		),
		**prefix_dict(
			gram_features(cosine_similarity_matrix(A_log_skew)),
			prefix=["log", "gram", "skew"],
		),
	)
