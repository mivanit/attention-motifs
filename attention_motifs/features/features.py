import json
from pathlib import Path
from typing import Callable


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



def scalar_feature_table(
	features_func: Callable[
		[Float[torch.Tensor, "batch n_ctx n_ctx"]],
		dict[str, Float[torch.Tensor, " batch"]],
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
	output: list[dict] = list()

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

		for prompt in tqdm.tqdm(prompts, desc="prompts", total=len(prompts)):
			activations_path, cache = load_activations(
				model_name=model_cfg.model_name,
				prompt=prompt,
				save_path=save_path,
				return_fmt="numpy",
			)

			for cache_key, head_batch in cache.items():
				layer_idx: int = int(cache_key.split(".")[1])
				for head_idx, A in enumerate(head_batch[0]):
					output.append(
						{
							**prefix_dict(
								dict(
									model=model,
									layer=layer_idx,
									cache_key=cache_key,
									head=head_idx,
									cls=f"{model}:L{layer_idx}:H{head_idx}",
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

	return pd.DataFrame(output)
