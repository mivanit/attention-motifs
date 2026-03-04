"""Full experimental pipeline for induction head ablation studies.

Orchestrates the complete workflow:
1. Load model and candidate heads
2. Run baseline measurements
3. Perform ablations with both methods
4. Collect and organize results
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Iterable, cast

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

from torch import Tensor
from transformer_lens import HookedTransformer


@dataclass
class ExperimentConfig:
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
	seed
	    Random seed for reproducibility.
	"""

	n_sequences: int = 100
	seq_length: int = 25
	n_repetitions: int = 4
	ablation_methods: list[AblationMethod] = field(
		default_factory=lambda: [AblationMethod.ZERO, AblationMethod.MEAN]
	)
	n_calibration_prompts: int = 50
	seed: int = 42


@dataclass
class ExperimentResults:
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
	    Baseline ICL score without any ablation.
	"""

	model_name: str
	config: ExperimentConfig
	results: list[AblationResult] = field(default_factory=list)
	baseline_loss: float = 0.0
	baseline_icl: float = 0.0

	def to_dataframe(self) -> pl.DataFrame:
		"""Convert results to Polars DataFrame."""
		rows: list[dict] = [r.serialize() for r in self.results]
		return pl.DataFrame(rows)

	def save(self, path: Path | str) -> None:
		"""Save results to JSON file."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)

		data: dict = {
			"model_name": self.model_name,
			"config": {
				"n_sequences": self.config.n_sequences,
				"seq_length": self.config.seq_length,
				"n_repetitions": self.config.n_repetitions,
				"ablation_methods": [m.value for m in self.config.ablation_methods],
				"n_calibration_prompts": self.config.n_calibration_prompts,
				"seed": self.config.seed,
			},
			"baseline_loss": self.baseline_loss,
			"baseline_icl": self.baseline_icl,
			"results": [r.serialize() for r in self.results],
		}
		path.write_text(json.dumps(data, indent=2))

	@classmethod
	def load(cls, path: Path | str) -> "ExperimentResults":
		"""Load results from JSON file."""
		path = Path(path)
		data: dict = json.loads(path.read_text())

		config: ExperimentConfig = ExperimentConfig(
			n_sequences=data["config"]["n_sequences"],
			seq_length=data["config"]["seq_length"],
			n_repetitions=data["config"]["n_repetitions"],
			ablation_methods=[
				AblationMethod(m) for m in data["config"]["ablation_methods"]
			],
			n_calibration_prompts=data["config"]["n_calibration_prompts"],
			seed=data["config"]["seed"],
		)

		results: list[AblationResult] = [
			AblationResult(
				head=r["head"],
				ablation_method=AblationMethod(r["ablation_method"]),
				baseline_repeated_loss=r["baseline_repeated_loss"],
				ablated_repeated_loss=r["ablated_repeated_loss"],
				loss_increase=r["loss_increase"],
				baseline_prefix_score=r["baseline_prefix_score"],
				ablated_prefix_score=r["ablated_prefix_score"],
				prefix_score_decrease=r["prefix_score_decrease"],
				baseline_prefix_score_legacy=r.get("baseline_prefix_score_legacy", 0.0),
				ablated_prefix_score_legacy=r.get("ablated_prefix_score_legacy", 0.0),
				prefix_score_decrease_legacy=r.get("prefix_score_decrease_legacy", 0.0),
				baseline_icl_score=r.get("baseline_icl_score", 0.0),
				ablated_icl_score=r.get("ablated_icl_score", 0.0),
				icl_degradation=r.get("icl_degradation", 0.0),
				copying_score=r.get("copying_score", 0.0),
				ablated_copying_score=r.get("ablated_copying_score", 0.0),
				copying_score_decrease=r.get("copying_score_decrease", 0.0),
				ov_copying_score=r.get("ov_copying_score", 0.0),
				ablated_ov_copying_score=r.get("ablated_ov_copying_score", 0.0),
				ov_copying_score_decrease=r.get("ov_copying_score_decrease", 0.0),
			)
			for r in data["results"]
		]

		return cls(
			model_name=data["model_name"],
			config=config,
			results=results,
			baseline_loss=data["baseline_loss"],
			baseline_icl=data.get("baseline_icl", 0.0),
		)


def run_ablation_experiment(
	model_name: str,
	candidate_heads: list[tuple[int, int]] | list[str],
	control_heads: list[tuple[int, int]] | list[str] | None = None,
	config: ExperimentConfig | None = None,
	icl_prompts: list[str] | None = None,
	device: str = "cuda",
	show_progress: bool = True,
) -> ExperimentResults:
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
	ExperimentResults
	    Results container with all ablation measurements.
	"""
	if config is None:
		config = ExperimentConfig()

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
		print("Computing mean activations for calibration...")
		calibration_tokens: list[Tensor] = [
			s.tokens for s in sequences[: config.n_calibration_prompts]
		]
		ablator.compute_mean_activations(
			calibration_tokens, show_progress=show_progress
		)

	# For pattern-preserving ablation: cache clean patterns once
	if AblationMethod.PATTERN_PRESERVING in config.ablation_methods:
		print("Caching clean attention patterns...")
		tokens_batch: Tensor = sequences_to_batch(sequences).to(device)
		ablator.set_clean_patterns(tokens_batch, prepend_bos=False)

	# Compute baseline metrics (no ablation)
	print("Computing baseline metrics...")
	baseline_loss: float = repeated_sequence_loss(model, sequences)
	baseline_icl: float = icl_score(model, icl_prompts) if icl_prompts else 0.0

	exp_results: ExperimentResults = ExperimentResults(
		model_name=model_name,
		config=config,
		baseline_loss=baseline_loss,
		baseline_icl=baseline_icl,
	)

	# Combine candidate and control heads
	all_heads: list[tuple[int, int]] = list(candidate_heads_int)
	if control_heads_int:
		all_heads.extend(control_heads_int)

	# Run ablation for each head and method
	heads_to_ablate: Iterable[tuple[int, int]] = all_heads
	if show_progress:
		heads_to_ablate = tqdm(all_heads, desc="Ablating heads")

	for layer, head in heads_to_ablate:
		head_str: str = f"{model_name}:L{layer}:H{head}"

		# Compute baseline scores for this head
		baseline_prefix: float = prefix_matching_score(model, layer, head, sequences)
		baseline_prefix_legacy: float = preceding_token_score(
			model, layer, head, sequences
		)
		baseline_copy: float = copying_score(model, layer, head, sequences)
		baseline_ov_copy: float = ov_copying_score(model, layer, head, sequences)

		for method in config.ablation_methods:
			# Run with ablation
			with ablator.ablate_heads([(layer, head)], method):
				ablated_loss: float = repeated_sequence_loss(model, sequences)
				ablated_prefix: float = prefix_matching_score(
					model, layer, head, sequences
				)
				ablated_prefix_legacy: float = preceding_token_score(
					model, layer, head, sequences
				)
				ablated_icl: float = (
					icl_score(model, icl_prompts) if icl_prompts else 0.0
				)
				ablated_copy: float = copying_score(model, layer, head, sequences)
				ablated_ov_copy: float = ov_copying_score(model, layer, head, sequences)

			result: AblationResult = AblationResult(
				head=head_str,
				ablation_method=method,
				baseline_repeated_loss=baseline_loss,
				ablated_repeated_loss=ablated_loss,
				loss_increase=ablated_loss - baseline_loss,
				baseline_prefix_score=baseline_prefix,
				ablated_prefix_score=ablated_prefix,
				prefix_score_decrease=baseline_prefix - ablated_prefix,
				baseline_prefix_score_legacy=baseline_prefix_legacy,
				ablated_prefix_score_legacy=ablated_prefix_legacy,
				prefix_score_decrease_legacy=(
					baseline_prefix_legacy - ablated_prefix_legacy
				),
				baseline_icl_score=baseline_icl,
				ablated_icl_score=ablated_icl,
				icl_degradation=ablated_icl - baseline_icl,
				copying_score=baseline_copy,
				ablated_copying_score=ablated_copy,
				copying_score_decrease=baseline_copy - ablated_copy,
				ov_copying_score=baseline_ov_copy,
				ablated_ov_copying_score=ablated_ov_copy,
				ov_copying_score_decrease=baseline_ov_copy - ablated_ov_copy,
			)
			exp_results.results.append(result)

	# Clean up cached patterns
	ablator.clear_clean_patterns()

	return exp_results


def run_cross_model_experiment(
	distance_result,
	models: list[str],
	n_candidates_per_model: int = 10,
	n_controls_per_model: int = 5,
	config: ExperimentConfig | None = None,
	icl_prompts: list[str] | None = None,
	device: str = "cuda",
	output_dir: Path | str | None = None,
	show_progress: bool = True,
) -> dict[str, ExperimentResults]:
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
	dict[str, ExperimentResults]
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

	all_results: dict[str, ExperimentResults] = {}

	for model_name in models:
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
		results: ExperimentResults = run_ablation_experiment(
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

	return all_results


def analyze_results(results: ExperimentResults) -> pl.DataFrame:
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
				pl.col("prefix_score_decrease").mean().alias("mean_prefix_decrease"),
				pl.col("prefix_score_decrease_legacy")
				.mean()
				.alias("mean_prefix_decrease_legacy"),
				pl.col("icl_degradation").mean().alias("mean_icl_degradation"),
				pl.col("copying_score_decrease").mean().alias("mean_copying_decrease"),
				pl.col("ov_copying_score_decrease")
				.mean()
				.alias("mean_ov_copying_decrease"),
			]
		)
		.sort("mean_loss_increase", descending=True)
	)

	return summary


def get_all_head_scores(results: ExperimentResults) -> pl.DataFrame:
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
	config: ExperimentConfig | None = None,
	device: str = "cuda",
	output_dir: Path | str | None = None,
	show_progress: bool = True,
) -> dict[str, ExperimentResults]:
	"""Run ablation experiments on candidate heads across all models.

	Iterates over models in ``candidates.heads_by_model``, calls
	``run_ablation_experiment()`` per model, collects results.

	Parameters
	----------
	candidates
	    Heads to evaluate, grouped by model.
	config
	    Experiment configuration. Uses defaults if None.
	device
	    Device to run on.
	output_dir
	    Directory to save per-model results. If None, results not saved.
	show_progress
	    Show progress bars.

	Returns
	-------
	dict[str, ExperimentResults]
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
	all_results: dict[str, ExperimentResults] = {}
	for model_name, heads in sorted(candidates.heads_by_model.items()):
		print(f"\n{'=' * 60}")
		print(f"Evaluating induction scores: {model_name} ({len(heads)} heads)")
		print(f"{'=' * 60}")

		results: ExperimentResults = run_ablation_experiment(
			model_name=model_name,
			candidate_heads=heads,
			config=config,
			device=device,
			show_progress=show_progress,
		)

		all_results[model_name] = results

		# Save intermediate results
		if output_dir_ is not None:
			results.save(
				output_dir_ / f"{cached_sanitize_model_name(model_name)}_results.json"
			)

	return all_results


def identify_induction_heads(
	results: ExperimentResults,
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
			& (pl.col("baseline_prefix_score") > prefix_threshold)
		)
		.select("head")
		.unique()
		.to_series()
		.to_list()
	)

	return induction_heads
