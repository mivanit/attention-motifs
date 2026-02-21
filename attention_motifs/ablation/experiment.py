"""Full experimental pipeline for induction head ablation studies.

Orchestrates the complete workflow:
1. Load model and candidate heads
2. Run baseline measurements
3. Perform ablations with both methods
4. Collect and organize results
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

import polars as pl
from tqdm import tqdm

from attention_motifs.ablation.ablate import (
	AblationMethod,
	HeadAblator,
)
from attention_motifs.attnpedia import heads_from_strings
from attention_motifs.ablation.candidates import (
	get_control_heads,
	find_candidate_induction_heads,
)
from attention_motifs.ablation.data import (
	RepeatedSequence,
	generate_repeated_sequences,
)
from attention_motifs.ablation.metrics import (
	AblationResult,
	repeated_sequence_loss,
	prefix_matching_score,
	icl_score,
)

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
		rows = [r.serialize() for r in self.results]
		return pl.DataFrame(rows)

	def save(self, path: Path | str) -> None:
		"""Save results to JSON file."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)

		data = {
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
		data = json.loads(path.read_text())

		config = ExperimentConfig(
			n_sequences=data["config"]["n_sequences"],
			seq_length=data["config"]["seq_length"],
			n_repetitions=data["config"]["n_repetitions"],
			ablation_methods=[
				AblationMethod(m) for m in data["config"]["ablation_methods"]
			],
			n_calibration_prompts=data["config"]["n_calibration_prompts"],
			seed=data["config"]["seed"],
		)

		results = [
			AblationResult(
				head=r["head"],
				ablation_method=AblationMethod(r["ablation_method"]),
				baseline_repeated_loss=r["baseline_repeated_loss"],
				ablated_repeated_loss=r["ablated_repeated_loss"],
				loss_increase=r["loss_increase"],
				baseline_prefix_score=r["baseline_prefix_score"],
				ablated_prefix_score=r["ablated_prefix_score"],
				prefix_score_decrease=r["prefix_score_decrease"],
				baseline_icl_score=r["baseline_icl_score"],
				ablated_icl_score=r["ablated_icl_score"],
				icl_degradation=r["icl_degradation"],
			)
			for r in data["results"]
		]

		return cls(
			model_name=data["model_name"],
			config=config,
			results=results,
			baseline_loss=data["baseline_loss"],
			baseline_icl=data["baseline_icl"],
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

	# ignores here are fine, we can assume that candidate_heads and control_heads are lists of strings or tuples, but not mixed
	# Convert string heads to tuples if needed
	if candidate_heads and isinstance(candidate_heads[0], str):
		candidate_heads = heads_from_strings(candidate_heads)  # ty: ignore[invalid-argument-type] # pyright: ignore[reportArgumentType]

	if control_heads and isinstance(control_heads[0], str):
		control_heads = heads_from_strings(control_heads)  # ty: ignore[invalid-argument-type] # pyright: ignore[reportArgumentType]

	# Load model
	print(f"Loading model: {model_name}")
	model: HookedTransformer = HookedTransformer.from_pretrained(
		model_name, device=device
	)

	# Create ablator
	ablator: HeadAblator = HeadAblator(model)

	# Generate test sequences
	print("Generating test sequences...")
	sequences: list[RepeatedSequence] = generate_repeated_sequences(
		tokenizer=model.tokenizer,
		n_sequences=config.n_sequences,
		seq_length=config.seq_length,
		n_repetitions=config.n_repetitions,
		seed=config.seed,
		device=device,
	)

	# Compute mean activations for mean ablation
	if AblationMethod.MEAN in config.ablation_methods:
		print("Computing mean activations for calibration...")
		# Use the same sequences for calibration
		calibration_tokens = [
			s.tokens for s in sequences[: config.n_calibration_prompts]
		]
		ablator.compute_mean_activations(
			calibration_tokens, show_progress=show_progress
		)

	# Compute baseline metrics (no ablation)
	print("Computing baseline metrics...")
	baseline_loss: float = repeated_sequence_loss(model, sequences)
	baseline_icl: float = icl_score(model, icl_prompts) if icl_prompts else 0.0

	results = ExperimentResults(
		model_name=model_name,
		config=config,
		baseline_loss=baseline_loss,
		baseline_icl=baseline_icl,
	)

	# Combine candidate and control heads
	all_heads = list(candidate_heads)
	if control_heads:
		all_heads.extend(control_heads)

	# Run ablation for each head and method
	head_iterator = all_heads
	if show_progress:
		head_iterator = tqdm(all_heads, desc="Ablating heads")

	for layer, head in head_iterator:
		head_str = f"{model_name}:L{layer}:H{head}"

		# Compute baseline prefix score for this head
		baseline_prefix: float = prefix_matching_score(model, layer, head, sequences)

		for method in config.ablation_methods:
			# Run with ablation
			with ablator.ablate_heads([(layer, head)], method):
				ablated_loss = repeated_sequence_loss(model, sequences)
				ablated_prefix = prefix_matching_score(model, layer, head, sequences)
				ablated_icl = icl_score(model, icl_prompts) if icl_prompts else 0.0

			result: AblationResult = AblationResult(
				head=head_str,
				ablation_method=method,
				baseline_repeated_loss=baseline_loss,
				ablated_repeated_loss=ablated_loss,
				loss_increase=ablated_loss - baseline_loss,
				baseline_prefix_score=baseline_prefix,
				ablated_prefix_score=ablated_prefix,
				prefix_score_decrease=baseline_prefix - ablated_prefix,
				baseline_icl_score=baseline_icl,
				ablated_icl_score=ablated_icl,
				icl_degradation=ablated_icl - baseline_icl,
			)
			results.results.append(result)

	return results


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
	if output_dir:
		output_dir_: Path = Path(output_dir)
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

		candidate_heads = heads_from_strings([h for h, _ in model_candidates])

		# Get control heads
		control_heads = []
		if n_controls_per_model > 0:
			control_strs = get_control_heads(
				distance_result, candidates, model_name, n_controls_per_model
			)
			control_heads = heads_from_strings(control_strs)

		# Run experiment
		results = run_ablation_experiment(
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
			results.save(output_dir_ / f"{model_name.replace('/', '_')}_results.json")

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
	df = results.to_dataframe()

	# Group by head to compare methods
	summary = (
		df.group_by("head")
		.agg(
			[
				pl.col("loss_increase").mean().alias("mean_loss_increase"),
				pl.col("loss_increase").max().alias("max_loss_increase"),
				pl.col("prefix_score_decrease").mean().alias("mean_prefix_decrease"),
				pl.col("icl_degradation").mean().alias("mean_icl_degradation"),
			]
		)
		.sort("mean_loss_increase", descending=True)
	)

	return summary


def identify_induction_heads(
	results: ExperimentResults,
	loss_threshold: float = 0.5,
	prefix_threshold: float = 0.1,
) -> list[str]:
	"""Identify heads that behave like induction heads based on ablation results.

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
	df = results.to_dataframe()

	induction_heads = (
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
