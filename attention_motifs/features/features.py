import itertools
import json
from pathlib import Path
from typing import Callable
import functools
import multiprocessing as mp


import torch
from jaxtyping import Float
import pandas as pd
import tqdm

# custom utils
from muutils.spinner import SpinnerContext

# pattern_lens
from pattern_lens.consts import (
	SPINNER_KWARGS,
)
from pattern_lens.load_activations import load_activations
from pattern_lens.figures import HTConfigMock

from attention_motifs.util import prefix_dict


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


def scalar_feature_table(
	features_func: Callable[
		[Float[torch.Tensor, "n_ctx n_ctx"]],
		dict[str, float],
	],
	save_path: Path = Path("../docs/temp"),
	models: list[str] | None = None,
) -> pd.DataFrame:
	if models is None:
		models = [
			json.loads(cfg)["model_name"]
			for cfg in (save_path / "models.jsonl").read_text().splitlines()
		]

	print(f"models: {models}")

	# output has cols:
	# model, prompt, layer_idx, head_idx, feature_name, feature_value
	output: list[dict[str, int | float | str]] = list()

	for idx, model in enumerate(models):
		print(f"model: '{model}'")
		with SpinnerContext(message="setting up paths", **SPINNER_KWARGS):
			model_path: Path = save_path / model
			with open(model_path / "model_cfg.json", "r") as f:
				model_cfg = HTConfigMock.load(json.load(f))

		with SpinnerContext(message="loading prompts", **SPINNER_KWARGS):
			# load prompts
			with open(model_path / "prompts.jsonl", "r") as f:
				prompts: list[dict] = [json.loads(line) for line in f.readlines()]
			# truncate to n_samples
			prompts = prompts

		print(f"{len(prompts)} prompts loaded")

		# for prompt in tqdm.tqdm(prompts, desc="prompts", total=len(prompts)):

		with mp.Pool(processes=mp.cpu_count()) as pool:
			# process each prompt in parallel
			prompt_func: Callable[[dict], list[dict[str, int | float | str]]] = (
				functools.partial(
					process_prompt,
					model_name=model,
					save_path=save_path,
					features_func=features_func,
				)
			)
			model_out: list[dict] = tqdm.tqdm(
				pool.imap(prompt_func, prompts), total=len(prompts)
			)
			output.extend(itertools.chain.from_iterable(model_out))

	return pd.DataFrame(output)
