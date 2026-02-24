import json
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.lib.npyio import NpzFile
from jaxtyping import Float
import matplotlib.pyplot as plt

from muutils.dbg import dbg
from attention_motifs.util.cached_sanitize_model_name import cached_sanitize_model_name


def prefix_dict[T_key](
	d: dict[str, T_key],
	prefix: str | list[str],
	sep: str = ".",
) -> dict[str, T_key]:
	prefix_str: str = prefix if isinstance(prefix, str) else sep.join(prefix)
	return {f"{prefix_str}{sep}{k}": v for k, v in d.items()}


def load_activations(
	model_name: str,
	base_path: Path = Path("../docs/demo"),
) -> tuple[
	dict,
	list[dict],
	list[NpzFile],
]:
	"returns (model_config, prompts, activations)"
	model_path: Path = base_path / cached_sanitize_model_name(model_name)

	# prompts
	with open(model_path / "prompts.jsonl") as f:
		prompts = [json.loads(line) for line in f]

	# activations
	activations: list[NpzFile] = [
		np.load(model_path / "prompts" / p["hash"] / "activations.npz") for p in prompts
	]

	# model config
	model_config: dict = json.loads((model_path / "model_cfg.json").read_text())

	return model_config, prompts, activations


def get_single_attn_pattern(
	sample: int,
	layer: int,
	head: int,
	activations,
) -> Float[np.ndarray, "n_ctx n_ctx"]:
	return activations[sample][f"blocks.{layer}.attn.hook_pattern"][0, head]


def plot_figs(
	n_figures: int,
	model_cfg: dict,
	prompt_dicts: list[dict],
	activations: list[NpzFile],
	figure_func: Callable,
	prompts: list[int] | None = None,
	layers: list[int] | None = None,
	heads: list[int] | None = None,
):
	n_lyrs: int = model_cfg["n_layers"]
	n_heads: int = model_cfg["n_heads"]

	if layers is None:
		layers = list(range(n_lyrs))

	if heads is None:
		heads = list(range(n_heads))

	if prompts is None:
		prompts = list(range(len(prompt_dicts)))

	for p_idx in prompts:
		for lyr in layers:
			for head in heads:
				dbg((p_idx, lyr, head))
				fig, axs = plt.subplots(1, n_figures, figsize=(n_figures * 7, 7))
				A = get_single_attn_pattern(p_idx, lyr, head, activations=activations)
				figure_func(A, axs, p_idx, lyr, head)
				prompt: dict = prompt_dicts[p_idx]
				fig.suptitle(
					f"model {model_cfg['model_name']} L{lyr}H{head}, prompt {p_idx}\n{prompt['hash'] = }, {prompt['n_tokens'] = }"
				)
				plt.show()
