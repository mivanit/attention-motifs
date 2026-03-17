"""Full experimental pipeline for induction head ablation studies.

Orchestrates the complete workflow:
1. Load model and candidate heads
2. Run baseline measurements
3. Perform ablations with both methods
4. Collect and organize results
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
import json
from typing import TYPE_CHECKING, Any, Iterable, cast

from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)

if TYPE_CHECKING:
	from attention_motifs.features.analysis import DistanceTensorResult

from pattern_lens.load_model import load_model

from attention_motifs.util.model_name import cached_sanitize_model_name

import polars as pl
from tqdm import tqdm

from attention_motifs.ablation.ablate import (
	AblationMethod,
	HeadAblator,
)
from attention_motifs.attnpedia import heads_from_strings
from attention_motifs.ablation.candidates import (
	CandidateHeads,
	find_candidate_induction_heads,
	get_control_heads,
)
from attention_motifs.ablation.data import (
	RepeatedSequence,
	generate_repeated_sequences,
	sequences_to_batch,
)
from attention_motifs.ablation.metrics import (
	AblationResult,
	repeated_sequence_loss,
	prefix_matching_score,
	preceding_token_score,
	icl_score,
	copying_score,
	ov_copying_score,
)

import torch
from torch import Tensor
from transformer_lens import HookedTransformer


@serializable_dataclass(kw_only=True)
class AblationConfig(SerializableDataclass):
	"""Configuration for ablation experiments.

	Attributes
	----------
	n_sequences
	    Number of repeated sequences for testing.
	seq_length
	    Base length of repeated pattern.
	n_repetitions
	    Number of times to repeat pattern.
	ablation_methods
	    List of ablation methods to test.
	n_calibration_prompts
	    Number of prompts for mean ablation calibration.
	calibration_prompts_file
	    Path to JSONL file with natural-text prompts for mean ablation
	    calibration.  If None, falls back to using test sequences.
	seed
	    Random seed for reproducibility.
	micro_batch_size
	    Sequences per micro-batch for metric computation.
	icl_prompts_file
	    Path to JSONL file with natural-text prompts for ICL evaluation.
	    If None, ICL scores are skipped.
	n_icl_prompts
	    Number of ICL prompts to sample from the file.
	max_heads_per_model
	    If set, randomly sample at most this many heads per model
	    (seeded by ``seed``). For testing only — prints a warning.
	"""

	n_sequences: int = serializable_field(default=100)
	seq_length: int = serializable_field(default=25)
	n_repetitions: int = serializable_field(default=4)
	ablation_methods: list[AblationMethod] = serializable_field(
		default_factory=lambda: [
			AblationMethod.ZERO,
			AblationMethod.MEAN,
			AblationMethod.PATTERN_PRESERVING,
		],
		serialization_fn=lambda methods: [m.value for m in methods],
		deserialize_fn=lambda methods: [AblationMethod(m) for m in methods],
	)
	n_calibration_prompts: int = serializable_field(default=50)
	calibration_prompts_file: str | None = serializable_field(
		default="data/text/pile_10k.jsonl"
	)
	seed: int = serializable_field(default=42)
	micro_batch_size: int = serializable_field(default=20)
	icl_prompts_file: str | None = serializable_field(default=None)
	n_icl_prompts: int = serializable_field(default=50)
	max_heads_per_model: int | None = serializable_field(default=None)


@serializable_dataclass
class AblationResults(SerializableDataclass):
	"""Container for experiment results.

	Attributes
	----------
	model_name
	    Name of the model tested.
	config
	    Experiment configuration.
	results
	    List of AblationResult for each head/method combination.
	baseline_loss
	    Baseline loss without any ablation.
	baseline_icl
	    Baseline ICL score without any ablation. None if not measured.
	"""

	model_name: str
	config: AblationConfig
	results: list[AblationResult] = serializable_field(
		default_factory=list,
		serialization_fn=lambda results: [r.serialize() for r in results],
		deserialize_fn=lambda results: [AblationResult.load(r) for r in results],
	)
	baseline_loss: float = serializable_field(default=0.0)
	baseline_icl: float | None = serializable_field(default=None)

	def to_dataframe(self) -> pl.DataFrame:
		"""Convert results to Polars DataFrame."""
		rows: list[dict] = [r.serialize() for r in self.results]
		return pl.DataFrame(rows)

	def save(self, path: Path | str) -> None:
		"""Save results to JSON file."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(self.serialize(), indent=2))

	@classmethod
	def read(cls, path: Path | str) -> "AblationResults":
		"""Read results from a JSON file.

		Handles backwards compatibility for old field names
		(``baseline_prefix_score`` → ``prefix_score``, etc.).
		"""
		data: dict[str, Any] = json.loads(Path(path).read_text())

		# Remap legacy field names in result dicts before load()
		for r in data.get("results", []):
			if "prefix_score" not in r and "baseline_prefix_score" in r:
				r["prefix_score"] = r.pop("baseline_prefix_score")
			if "prefix_score_legacy" not in r and "baseline_prefix_score_legacy" in r:
				r["prefix_score_legacy"] = r.pop("baseline_prefix_score_legacy")

		return cls.load(data)


def _batched_metric(
	fn: Callable[..., float],
	sequences: Sequence[RepeatedSequence],
	micro_batch_size: int,
	**kwargs: Any,
) -> float:
	"""Run a metric function in micro-batches and return weighted mean.

	Parameters
	----------
	fn
	    Metric function that accepts ``sequences=`` and returns a float.
	sequences
	    Full list of sequences.
	micro_batch_size
	    Max sequences per forward pass.
	**kwargs
	    Forwarded to *fn*.
	"""
	if micro_batch_size >= len(sequences):
		return fn(sequences=sequences, **kwargs)

	total: float = 0.0
	count: int = 0
	for i in range(0, len(sequences), micro_batch_size):
		chunk: Sequence[RepeatedSequence] = sequences[i : i + micro_batch_size]
		result: float = fn(sequences=chunk, **kwargs)
		total += result * len(chunk)
		count += len(chunk)
		torch.cuda.empty_cache()
	return total / count


def run_ablation_experiment(
	model_name: str,
	candidate_heads: list[tuple[int, int]] | list[str],
	control_heads: list[tuple[int, int]] | list[str] | None = None,
	config: AblationConfig | None = None,
	icl_prompts: list[str] | None = None,
	device: str = "cuda",
	show_progress: bool = True,
) -> AblationResults:
	"""Run full ablation experiment on a model.

	Parameters
	----------
	model_name
	    TransformerLens model name (e.g., "gpt2-small", "pythia-1b").
	candidate_heads
	    List of (layer, head) tuples or head strings to test.
	control_heads
	    Optional control heads for baseline comparison.
	config
	    Experiment configuration. Uses defaults if None.
	icl_prompts
	    Text prompts for ICL score (should be long, 500+ tokens).
	device
	    Device to run on.
	show_progress
	    Show progress bars.

	Returns
	-------
	AblationResults
	    Results container with all ablation measurements.
	"""
	if config is None:
		config = AblationConfig()
	assert config is not None

	# Convert string heads to tuples if needed
	candidate_heads_int: list[tuple[int, int]]
	if candidate_heads and isinstance(candidate_heads[0], str):
		candidate_heads_int = heads_from_strings(cast(list[str], candidate_heads))
	else:
		candidate_heads_int = cast(list[tuple[int, int]], candidate_heads)

	control_heads_int: list[tuple[int, int]] | None = None
	if control_heads:
		if isinstance(control_heads[0], str):
			control_heads_int = heads_from_strings(cast(list[str], control_heads))
		else:
			control_heads_int = cast(list[tuple[int, int]], control_heads)

	# Load model
	print(f"Loading model: {model_name}")
	model: HookedTransformer = load_model(model_name, device=device)

	# Create ablator
	ablator: HeadAblator = HeadAblator(model)

	# Generate test sequences (with BOS prepended)
	print("Generating test sequences...")
	sequences: list[RepeatedSequence] = generate_repeated_sequences(
		tokenizer=model.tokenizer,
		n_sequences=config.n_sequences,
		seq_length=config.seq_length,
		n_repetitions=config.n_repetitions,
		seed=config.seed,
		device=device,
		prepend_bos=True,
	)

	# Compute mean activations for mean ablation
	if AblationMethod.MEAN in config.ablation_methods:
		calibration_prompts: list[str] | list[Tensor] | None = None

		# Try loading natural text for calibration (preferred)
		if config.calibration_prompts_file is not None:
			cal_path: Path = Path(config.calibration_prompts_file)
			if cal_path.exists():
				from attention_motifs.ablation.data import load_icl_texts

				calibration_prompts = load_icl_texts(
					cal_path,
					n_prompts=config.n_calibration_prompts,
					seed=config.seed,
				)
				print(
					f"Computing mean activations from {len(calibration_prompts)} "
					f"natural-text prompts ({cal_path.name})..."
				)
			else:
				print(
					f"Warning: calibration file not found at {cal_path}, "
					"falling back to test sequences",
				)

		# Fall back to test sequences if no file loaded
		if calibration_prompts is None:
			print("Computing mean activations from test sequences...")
			calibration_prompts = [
				s.tokens for s in sequences[: config.n_calibration_prompts]
			]

		ablator.compute_mean_activations(
			calibration_prompts, show_progress=show_progress
		)

	# For pattern-preserving ablation: cache clean patterns once
	if AblationMethod.PATTERN_PRESERVING in config.ablation_methods:
		print("Caching clean attention patterns...")
		tokens_batch: Tensor = sequences_to_batch(sequences).to(device)
		ablator.set_clean_patterns(tokens_batch, prepend_bos=False)

	# Compute baseline metrics (no ablation)
	print("Computing baseline metrics...")
	mbs: int = config.micro_batch_size
	baseline_loss: float = _batched_metric(
		repeated_sequence_loss, sequences, mbs, model=model
	)
	baseline_icl: float | None = icl_score(model, icl_prompts) if icl_prompts else None

	exp_results: AblationResults = AblationResults(
		model_name=model_name,
		config=config,
		baseline_loss=baseline_loss,
		baseline_icl=baseline_icl,
	)

	# Combine candidate and control heads (deduplicate, candidates first)
	seen: set[tuple[int, int]] = set(candidate_heads_int)
	all_heads: list[tuple[int, int]] = list(candidate_heads_int)
	if control_heads_int:
		for h in control_heads_int:
			if h not in seen:
				all_heads.append(h)
				seen.add(h)

	# Run ablation for each head and method
	heads_to_ablate: Iterable[tuple[int, int]] = all_heads
	if show_progress:
		heads_to_ablate = tqdm(all_heads, desc="Ablating heads")

	for layer, head in heads_to_ablate:
		head_str: str = f"{model_name}:L{layer}:H{head}"

		try:
			# Compute head characterization scores (not affected by ablation)
			head_prefix: float = _batched_metric(
				prefix_matching_score,
				sequences,
				mbs,
				model=model,
				layer=layer,
				head=head,
			)
			head_prefix_legacy: float = _batched_metric(
				preceding_token_score,
				sequences,
				mbs,
				model=model,
				layer=layer,
				head=head,
			)
			head_copy: float = _batched_metric(
				copying_score,
				sequences,
				mbs,
				model=model,
				layer=layer,
				head=head,
			)
			head_ov_copy: float = _batched_metric(
				ov_copying_score,
				sequences,
				mbs,
				model=model,
				layer=layer,
				head=head,
			)

			for method in config.ablation_methods:
				# Run with ablation — only compute causal metrics
				with ablator.ablate_heads([(layer, head)], method):
					ablated_loss: float = _batched_metric(
						repeated_sequence_loss,
						sequences,
						mbs,
						model=model,
					)
					# For non-PATTERN_PRESERVING, ICL runs inside
					# the ablation context (patterns are not frozen).
					ablated_icl: float | None = (
						icl_score(model, icl_prompts)
						if icl_prompts and method != AblationMethod.PATTERN_PRESERVING
						else None
					)

				# For PATTERN_PRESERVING, ICL needs per-prompt pattern
				# caching: each prompt has a different seq length, so we
				# must re-cache clean patterns and re-enter ablate_heads
				# for each one (Olsson et al. 2022).
				if icl_prompts and method == AblationMethod.PATTERN_PRESERVING:
					heads_to_ablate = [(layer, head)]
					saved_patterns: dict | None = ablator._clean_patterns

					def _pattern_preserving_forward(
						m: HookedTransformer, tokens: Tensor
					) -> Tensor:
						ablator.set_clean_patterns(tokens)  # noqa: F821
						with ablator.ablate_heads(  # noqa: F821
							[(layer, head)], AblationMethod.PATTERN_PRESERVING
						):
							return m(tokens)

					ablated_icl = icl_score(
						model,
						icl_prompts,
						forward_fn=_pattern_preserving_forward,
					)
					# Restore repeated-sequence patterns for next head
					ablator._clean_patterns = saved_patterns

				result: AblationResult = AblationResult(
					head=head_str,
					ablation_method=method,
					baseline_repeated_loss=baseline_loss,
					ablated_repeated_loss=ablated_loss,
					loss_increase=ablated_loss - baseline_loss,
					prefix_score=head_prefix,
					prefix_score_legacy=head_prefix_legacy,
					copying_score=head_copy,
					ov_copying_score=head_ov_copy,
					baseline_icl_score=baseline_icl,
					ablated_icl_score=ablated_icl,
					icl_degradation=(ablated_icl - baseline_icl)
					if (ablated_icl is not None and baseline_icl is not None)
					else None,
				)
				exp_results.results.append(result)
		except torch.cuda.OutOfMemoryError:
			print(f"  OOM on {head_str}, skipping")
			torch.cuda.empty_cache()

	# Clean up cached patterns and free GPU memory
	ablator.clear_clean_patterns()
	del ablator
	del model
	torch.cuda.empty_cache()

	return exp_results


def run_cross_model_experiment(
	distance_result: "DistanceTensorResult",
	models: list[str],
	n_candidates_per_model: int = 10,
	n_controls_per_model: int = 5,
	config: AblationConfig | None = None,
	icl_prompts: list[str] | None = None,
	device: str = "cuda",
	output_dir: Path | str | None = None,
	show_progress: bool = True,
) -> dict[str, AblationResults]:
	"""Run ablation experiments across multiple models.

	Finds candidate induction heads for each model based on proximity
	to known GPT-2 induction heads, then runs ablation experiments.

	Parameters
	----------
	distance_result
	    Pre-computed DistanceTensorResult.
	models
	    List of model names to test.
	n_candidates_per_model
	    Number of candidate heads to test per model.
	n_controls_per_model
	    Number of control heads per model.
	config
	    Experiment configuration.
	icl_prompts
	    Prompts for ICL score.
	device
	    Device to run on.
	output_dir
	    Directory to save results. If None, results not saved.
	show_progress
	    Show progress.

	Returns
	-------
	dict[str, AblationResults]
	    Results for each model.
	"""
	output_dir_: Path | None = None
	if output_dir:
		output_dir_ = Path(output_dir)
		output_dir_.mkdir(parents=True, exist_ok=True)

	# Find candidate heads
	print("Finding candidate induction heads...")
	candidates = find_candidate_induction_heads(
		distance_result,
		k_neighbors=n_candidates_per_model * 2,
	)

	all_results: dict[str, AblationResults] = {}

	for model_name in models:
		# Resume: skip models with existing results
		if output_dir_ is not None:
			results_path: Path = (
				output_dir_ / f"{cached_sanitize_model_name(model_name)}_results.json"
			)
			if results_path.exists():
				print(f"Skipping {model_name} — results exist at {results_path}")
				all_results[model_name] = AblationResults.read(results_path)
				continue

		print(f"\n{'=' * 60}")
		print(f"Running experiment for: {model_name}")
		print(f"{'=' * 60}")

		# Get candidates for this model
		model_candidates = candidates.get_top_candidates(
			model_name, n_candidates_per_model
		)
		if not model_candidates:
			print(f"No candidates found for {model_name}, skipping...")
			continue

		candidate_heads: list[tuple[int, int]] = heads_from_strings(
			[h for h, _ in model_candidates]
		)

		# Get control heads
		control_heads: list[tuple[int, int]] = []
		if n_controls_per_model > 0:
			control_strs: list[str] = get_control_heads(
				distance_result, candidates, model_name, n_controls_per_model
			)
			control_heads = heads_from_strings(control_strs)

		# Run experiment
		results: AblationResults = run_ablation_experiment(
			model_name=model_name,
			candidate_heads=candidate_heads,
			control_heads=control_heads,
			config=config,
			icl_prompts=icl_prompts,
			device=device,
			show_progress=show_progress,
		)

		all_results[model_name] = results

		# Save intermediate results
		if output_dir:
			assert output_dir_
			results.save(
				output_dir_ / f"{cached_sanitize_model_name(model_name)}_results.json"
			)

	# Write HTML frontend if output_dir is set
	if output_dir_ is not None and all_results:
		from attention_motifs.ablation.frontend import write_ablation_frontend

		write_ablation_frontend(all_results, output_dir_)

	return all_results


def analyze_results(results: AblationResults) -> pl.DataFrame:
	"""Analyze ablation results and compute summary statistics.

	Parameters
	----------
	results
	    Experiment results.

	Returns
	-------
	pl.DataFrame
	    Summary DataFrame with statistics per head.
	"""
	df: pl.DataFrame = results.to_dataframe()

	# Group by head to compare methods
	summary: pl.DataFrame = (
		df.group_by("head")
		.agg(
			[
				pl.col("loss_increase").mean().alias("mean_loss_increase"),
				pl.col("loss_increase").max().alias("max_loss_increase"),
				pl.col("prefix_score").first().alias("prefix_score"),
				pl.col("prefix_score_legacy").first().alias("prefix_score_legacy"),
				pl.col("icl_degradation").mean().alias("mean_icl_degradation"),
				pl.col("copying_score").first().alias("copying_score"),
				pl.col("ov_copying_score").first().alias("ov_copying_score"),
			]
		)
		.sort("mean_loss_increase", descending=True)
	)

	return summary


def get_all_head_scores(results: AblationResults) -> pl.DataFrame:
	"""Return a DataFrame with all metric scores for all heads.

	Unlike ``identify_induction_heads``, this does not apply any
	thresholds — it returns all scores and lets the consumer decide
	how to filter.

	Parameters
	----------
	results
	    Experiment results.

	Returns
	-------
	pl.DataFrame
	    DataFrame with all scores, sorted by loss_increase descending.
	"""
	df: pl.DataFrame = results.to_dataframe()
	return df.sort("loss_increase", descending=True)


def evaluate_induction_scores(
	candidates: CandidateHeads,
	config: AblationConfig | None = None,
	icl_prompts: list[str] | None = None,
	device: str = "cuda",
	output_dir: Path | str | None = None,
	show_progress: bool = True,
) -> dict[str, AblationResults]:
	"""Run ablation experiments on candidate heads across all models.

	Iterates over models in ``candidates.heads_by_model``, calls
	``run_ablation_experiment()`` per model, collects results.

	Parameters
	----------
	candidates
	    Heads to evaluate, grouped by model.
	config
	    Experiment configuration. Uses defaults if None.
	icl_prompts
	    Text prompts for ICL score (should be long, 500+ tokens).
	    If None, ICL scores are skipped.
	device
	    Device to run on.
	output_dir
	    Directory to save per-model results. If None, results not saved.
	show_progress
	    Show progress bars.

	Returns
	-------
	dict[str, AblationResults]
	    Results for each model.
	"""
	if not candidates.heads_by_model:
		print("No heads to evaluate")
		return {}

	# Print summary
	print(candidates.summary())

	# Set up output directory
	output_dir_: Path | None = None
	if output_dir is not None:
		output_dir_ = Path(output_dir)
		output_dir_.mkdir(parents=True, exist_ok=True)

	# Run ablation per model
	all_results: dict[str, AblationResults] = {}
	for model_name, heads_orig in sorted(candidates.heads_by_model.items()):
		heads: list[tuple[int, int]] = list(heads_orig)
		if config is not None and config.max_heads_per_model is not None:
			warnings.warn(
				f"max_heads_per_model={config.max_heads_per_model} is set — "
				f"sampling {min(config.max_heads_per_model, len(heads))}/{len(heads)} "
				f"heads for {model_name}. Do not use for production experiments.",
				stacklevel=2,
			)
			if len(heads) > config.max_heads_per_model:
				import random

				rng: random.Random = random.Random(config.seed)
				heads = rng.sample(heads, config.max_heads_per_model)
		# Resume: skip models with existing results
		if output_dir_ is not None:
			results_path: Path = (
				output_dir_ / f"{cached_sanitize_model_name(model_name)}_results.json"
			)
			if results_path.exists():
				print(f"Skipping {model_name} — results exist at {results_path}")
				all_results[model_name] = AblationResults.read(results_path)
				continue

		print(f"\n{'=' * 60}")
		print(f"Evaluating induction scores: {model_name} ({len(heads)} heads)")
		print(f"{'=' * 60}")

		results: AblationResults = run_ablation_experiment(
			model_name=model_name,
			candidate_heads=heads,
			config=config,
			icl_prompts=icl_prompts,
			device=device,
			show_progress=show_progress,
		)

		all_results[model_name] = results

		# Save intermediate results
		if output_dir_ is not None:
			results.save(
				output_dir_ / f"{cached_sanitize_model_name(model_name)}_results.json"
			)

	# Write HTML frontend if output_dir is set
	if output_dir_ is not None and all_results:
		from attention_motifs.ablation.frontend import write_ablation_frontend

		write_ablation_frontend(all_results, output_dir_)

	return all_results


def identify_induction_heads(
	results: AblationResults,
	loss_threshold: float = 0.5,
	prefix_threshold: float = 0.1,
) -> list[str]:
	"""Identify heads that behave like induction heads based on ablation results.

	.. deprecated::
	    Use :func:`get_all_head_scores` and apply your own thresholds.

	A head is classified as an induction head if:
	1. Ablating it causes significant loss increase on repeated sequences
	2. It has a high prefix matching score

	Parameters
	----------
	results
	    Experiment results.
	loss_threshold
	    Minimum loss increase to be considered significant.
	prefix_threshold
	    Minimum baseline prefix score.

	Returns
	-------
	list[str]
	    List of head identifiers classified as induction heads.
	"""
	warnings.warn(
		"identify_induction_heads is deprecated. "
		"Use get_all_head_scores() and apply your own thresholds.",
		DeprecationWarning,
		stacklevel=2,
	)
	df: pl.DataFrame = results.to_dataframe()

	induction_heads: list[str] = (
		df.filter(
			(pl.col("loss_increase") > loss_threshold)
			& (pl.col("prefix_score") > prefix_threshold)
		)
		.select("head")
		.unique()
		.to_series()
		.to_list()
	)

	return induction_heads
