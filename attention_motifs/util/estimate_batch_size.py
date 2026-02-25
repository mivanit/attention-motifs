"""Estimate maximum batch size for activation extraction given GPU VRAM.

Reads models from a pipeline config and uses the TransformerLens model table
to look up architecture parameters (n_layers, n_heads, d_model, n_params).
Estimates peak VRAM usage during ``run_with_cache`` with attention-pattern-only
caching (the pattern_lens default).

Memory model (fp32)::

    model_weights     = n_params x dtype_bytes
    cache_per_sample  = n_layers x n_heads x seq² x dtype_bytes   (accumulated)
    working_per_sample = (                                         (transient, 1 layer)
        3 x seq x d_model                                          Q, K, V
      + 2 x n_heads x seq²                                         attn scores + softmax
      + 8 x seq x d_model                                          residual + MLP
    ) x dtype_bytes
    max_batch = floor((vram - overhead - model_weights)
                      / (cache_per_sample + working_per_sample))

Usage::

    python -m attention_motifs.util.estimate_batch_size
    python -m attention_motifs.util.estimate_batch_size --vram 24
    python -m attention_motifs.util.estimate_batch_size --config tests/pipeline_cfg_test.toml
"""

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from attention_motifs.pipeline.model_table import fetch_model_table_df

BYTES_PER_GB: int = 1 << 30


@dataclass(frozen=True, kw_only=True)
class BatchEstimate:
	"""Batch size estimate for a single model."""

	model: str
	n_params: int
	n_layers: int
	n_heads: int
	d_model: int
	model_mem_gb: float
	cache_per_sample_mb: float
	working_per_sample_mb: float
	max_batch: int
	max_batch_pow2: int


def _floor_pow2(n: int) -> int:
	"""Largest power of 2 <= n. Returns 0 for n <= 0."""
	if n <= 0:
		return 0
	return 1 << (n.bit_length() - 1)


def estimate_max_batch(
	*,
	n_params: int,
	n_layers: int,
	n_heads: int,
	d_model: int,
	max_seq_len: int,
	vram_bytes: int,
	overhead_bytes: int,
	dtype_bytes: int = 4,
) -> BatchEstimate:
	"""Estimate maximum batch size for a single model."""
	seq: int = max_seq_len

	model_mem: int = n_params * dtype_bytes
	cache_per_sample: int = n_layers * n_heads * seq * seq * dtype_bytes
	working_per_sample: int = (
		3 * seq * d_model + 2 * n_heads * seq * seq + 8 * seq * d_model
	) * dtype_bytes
	per_sample: int = cache_per_sample + working_per_sample

	available: int = vram_bytes - overhead_bytes - model_mem
	max_batch: int = max(0, available // per_sample) if per_sample > 0 else 0

	return BatchEstimate(
		model="",  # filled in by caller
		n_params=n_params,
		n_layers=n_layers,
		n_heads=n_heads,
		d_model=d_model,
		model_mem_gb=model_mem / BYTES_PER_GB,
		cache_per_sample_mb=cache_per_sample / (1 << 20),
		working_per_sample_mb=working_per_sample / (1 << 20),
		max_batch=max_batch,
		max_batch_pow2=_floor_pow2(max_batch),
	)


def estimate_s4_memory(
	*,
	models: list[str],
	prompts_n_samples: int,
	pca_n_components: int,
	n_proc: int,
) -> dict:
	"""Estimate CPU RAM usage for pipeline step s4 (head distance computation).

	Looks up model architectures from the TransformerLens model table to
	compute ``total_heads = sum(n_layers * n_heads)`` across all models,
	then estimates memory for the dense ``(p, h, d)`` array and distance
	output tensor.
	"""
	from attention_motifs.util.model_name import cached_resolve_model_name

	df: pl.DataFrame = fetch_model_table_df()

	total_heads: int = 0
	per_model: dict[str, dict] = {}
	model_name: str
	for model_name in models:
		# cfg.models contains sanitized names; resolve back to the raw
		# TransformerLens default alias used in the model table CSV.
		resolved_name: str = cached_resolve_model_name(model_name)
		row_df: pl.DataFrame = df.filter(
			pl.col("name.default_alias") == resolved_name
		)
		if row_df.is_empty():
			per_model[model_name] = {"error": "not found in model table"}
			continue

		row: dict = row_df.row(0, named=True)
		n_layers_val: int | None = row.get("cfg.n_layers")
		n_heads_val: int | None = row.get("cfg.n_heads")

		if n_layers_val is None or n_heads_val is None:
			per_model[model_name] = {
				"error": "missing n_layers or n_heads in model table"
			}
			continue

		n_layers: int = int(n_layers_val)
		n_heads: int = int(n_heads_val)
		heads: int = n_layers * n_heads
		total_heads += heads
		per_model[model_name] = {
			"resolved_alias": resolved_name,
			"n_layers": n_layers,
			"n_heads": n_heads,
			"heads": heads,
		}

	p: int = prompts_n_samples
	h: int = total_heads
	d: int = pca_n_components

	dense_array_bytes: int = p * h * d * 8
	output_reduced_bytes: int = h * h * 8
	output_full_bytes: int = h * h * p * 8
	parallel_overhead_bytes: int = n_proc * h * h * 8

	# Peak for reduce=True (parallel): dense array + worker accumulators + output
	peak_reduced_bytes: int = (
		dense_array_bytes + parallel_overhead_bytes + output_reduced_bytes
	)
	# Peak for reduce=False (serial): dense array + full output tensor
	peak_full_bytes: int = dense_array_bytes + output_full_bytes

	return {
		"total_heads": h,
		"prompts": p,
		"pca_components": d,
		"n_proc": n_proc,
		"per_model": per_model,
		"dense_array_gb": round(dense_array_bytes / BYTES_PER_GB, 3),
		"output_reduced_mb": round(output_reduced_bytes / (1 << 20), 1),
		"output_full_gb": round(output_full_bytes / BYTES_PER_GB, 3),
		"parallel_overhead_gb": round(parallel_overhead_bytes / BYTES_PER_GB, 3),
		"peak_reduced_parallel_gb": round(peak_reduced_bytes / BYTES_PER_GB, 3),
		"peak_full_serial_gb": round(peak_full_bytes / BYTES_PER_GB, 3),
	}


def _detect_vram(device: str) -> tuple[str, float]:
	"""Detect GPU name and total VRAM in GB. Raises RuntimeError on failure."""
	import torch

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA not available; pass --vram manually")
	dev: torch.device = torch.device(device)
	props = torch.cuda.get_device_properties(dev)  # type inferred
	name: str = props.name
	vram_gb: float = props.total_memory / BYTES_PER_GB
	return name, vram_gb


def main(
	*,
	config_path: Path = Path("pipeline_cfg.toml"),
	models: list[str] | None = None,
	vram_gb: float | None = None,
	device: str = "cuda:0",
	max_seq_len: int | None = None,
	overhead_gb: float = 1.0,
	dtype_bytes: int = 4,
) -> dict:
	"""Estimate max batch sizes for each model.

	If *models* is provided, uses that list directly. Otherwise reads
	the model list (and ``prompts_max_chars`` as default *max_seq_len*)
	from the TOML file at *config_path*.

	Returns the report dict (same structure printed as JSON by ``cli``).
	"""
	# -- resolve models & max_seq_len from config if needed ------------------
	if models is None:
		with config_path.open("rb") as f:
			cfg: dict = tomllib.load(f)
		models = cfg.get("models", [])
		if not models:
			raise ValueError(f"no models found in {config_path}")
		if max_seq_len is None:
			max_seq_len = int(cfg.get("prompts_max_chars", 512))

	if max_seq_len is None:
		max_seq_len = 512

	# -- detect GPU ----------------------------------------------------------
	gpu_name: str
	if vram_gb is None:
		gpu_name, vram_gb = _detect_vram(device)
	else:
		gpu_name = "(manual override)"

	vram_bytes: int = int(vram_gb * BYTES_PER_GB)
	overhead_bytes: int = int(overhead_gb * BYTES_PER_GB)

	# -- fetch model table ---------------------------------------------------
	df: pl.DataFrame = fetch_model_table_df()

	# -- estimate per model --------------------------------------------------
	results: dict[str, dict] = {}
	valid_batches: list[int] = []

	model_name: str
	for model_name in models:
		row_df: pl.DataFrame = df.filter(pl.col("name.default_alias") == model_name)
		if row_df.is_empty():
			results[model_name] = {"error": "not found in model table"}
			continue

		row: dict = row_df.row(0, named=True)
		n_params_val: int | None = row.get("n_params.as_int")
		n_layers_val: int | None = row.get("cfg.n_layers")
		n_heads_val: int | None = row.get("cfg.n_heads")
		d_model_val: int | None = row.get("cfg.d_model")

		if any(
			v is None for v in (n_params_val, n_layers_val, n_heads_val, d_model_val)
		):
			results[model_name] = {"error": "missing architecture info in model table"}
			continue

		est: BatchEstimate = estimate_max_batch(
			n_params=int(n_params_val),  # type: ignore[arg-type]
			n_layers=int(n_layers_val),  # type: ignore[arg-type]
			n_heads=int(n_heads_val),  # type: ignore[arg-type]
			d_model=int(d_model_val),  # type: ignore[arg-type]
			max_seq_len=max_seq_len,
			vram_bytes=vram_bytes,
			overhead_bytes=overhead_bytes,
			dtype_bytes=dtype_bytes,
		)

		results[model_name] = {
			"n_params": est.n_params,
			"n_layers": est.n_layers,
			"n_heads": est.n_heads,
			"d_model": est.d_model,
			"model_mem_gb": round(est.model_mem_gb, 2),
			"cache_per_sample_mb": round(est.cache_per_sample_mb, 2),
			"working_per_sample_mb": round(est.working_per_sample_mb, 2),
			"max_batch": est.max_batch,
			"max_batch_pow2": est.max_batch_pow2,
		}
		valid_batches.append(est.max_batch)

	# -- overall -------------------------------------------------------------
	overall: int = min(valid_batches) if valid_batches else 0
	overall_pow2: int = _floor_pow2(overall)

	return {
		"gpu": gpu_name,
		"vram_gb": round(vram_gb, 2),
		"max_seq_len": max_seq_len,
		"overhead_gb": overhead_gb,
		"dtype_bytes": dtype_bytes,
		"models": results,
		"overall_max_batch": overall,
		"overall_max_batch_pow2": overall_pow2,
	}


def cli() -> None:
	"""Parse CLI arguments and call main."""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Estimate maximum batch size for activation extraction given GPU VRAM.",
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("pipeline_cfg.toml"),
		help="Pipeline config TOML path (default: pipeline_cfg.toml)",
	)
	parser.add_argument(
		"--vram",
		type=float,
		default=None,
		help="Override total VRAM in GB (default: auto-detect via CUDA)",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda:0",
		help="CUDA device for auto-detection (default: cuda:0)",
	)
	parser.add_argument(
		"--max-seq-len",
		type=int,
		default=None,
		help="Override max sequence length (default: prompts_max_chars from config)",
	)
	parser.add_argument(
		"--overhead",
		type=float,
		default=1.0,
		help="CUDA overhead in GB (default: 1.0)",
	)
	parser.add_argument(
		"--dtype-bytes",
		type=int,
		default=4,
		help="Bytes per parameter/element, e.g. 4 for fp32, 2 for fp16 (default: 4)",
	)

	args: argparse.Namespace = parser.parse_args()
	try:
		report: dict = main(
			config_path=args.config,
			vram_gb=args.vram,
			device=args.device,
			max_seq_len=args.max_seq_len,
			overhead_gb=args.overhead,
			dtype_bytes=args.dtype_bytes,
		)
	except ValueError as e:
		print(f"error: {e}", file=sys.stderr)
		sys.exit(2)
	print(json.dumps(report, indent=2))


if __name__ == "__main__":
	cli()
