"""Evaluation metrics for induction head ablation studies.

Implements three key metrics from the literature:
1. Loss on repeated sequences - core test for induction behavior
2. Prefix matching score - how much head attends to induction positions
3. In-context learning score - early vs late loss difference
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from attention_motifs.ablation.data import (
	RepeatedSequence,
	generate_repeated_sequences,
	sequences_to_batch,
	get_induction_mask,
)
from attention_motifs.ablation.ablate import AblationMethod

from transformer_lens import HookedTransformer


@dataclass
class AblationResult:
	"""Results from an ablation experiment on a single head.

	Attributes
	----------
	head
	    Head identifier (e.g., "pythia-1b:L5:H7").
	ablation_method
	    Method used for ablation.
	baseline_repeated_loss
	    Loss on repeated sequences without ablation.
	ablated_repeated_loss
	    Loss on repeated sequences with ablation.
	loss_increase
	    Difference: ablated - baseline (positive = head was helping).
	baseline_prefix_score
	    Prefix matching score without ablation.
	ablated_prefix_score
	    Prefix matching score with ablation.
	prefix_score_decrease
	    Difference: baseline - ablated (positive = head was doing induction).
	baseline_icl_score
	    In-context learning score without ablation.
	ablated_icl_score
	    In-context learning score with ablation.
	icl_degradation
	    How much worse ICL became (more negative = worse).
	"""

	head: str
	ablation_method: AblationMethod
	baseline_repeated_loss: float
	ablated_repeated_loss: float
	loss_increase: float
	baseline_prefix_score: float
	ablated_prefix_score: float
	prefix_score_decrease: float
	baseline_icl_score: float
	ablated_icl_score: float
	icl_degradation: float

	def serialize(self) -> dict:
		"""Convert to dictionary."""
		return {
			"head": self.head,
			"ablation_method": self.ablation_method.value,
			"baseline_repeated_loss": self.baseline_repeated_loss,
			"ablated_repeated_loss": self.ablated_repeated_loss,
			"loss_increase": self.loss_increase,
			"baseline_prefix_score": self.baseline_prefix_score,
			"ablated_prefix_score": self.ablated_prefix_score,
			"prefix_score_decrease": self.prefix_score_decrease,
			"baseline_icl_score": self.baseline_icl_score,
			"ablated_icl_score": self.ablated_icl_score,
			"icl_degradation": self.icl_degradation,
		}


def repeated_sequence_loss(
	model: "HookedTransformer",
	sequences: Sequence[RepeatedSequence] | None = None,
	seq_length: int = 25,
	n_repetitions: int = 4,
	n_sequences: int = 50,
	only_induction_positions: bool = True,
	seed: int | None = 42,
) -> float:
	"""Compute mean loss on repeated token sequences.

	This is the core metric for testing induction heads. High loss increase
	when ablating a head indicates that head was helping with induction.

	Parameters
	----------
	model
	    TransformerLens model.
	sequences
	    Pre-generated sequences. If None, generates new ones.
	seq_length
	    Length of base pattern (if generating).
	n_repetitions
	    Number of repetitions (if generating).
	n_sequences
	    Number of sequences (if generating).
	only_induction_positions
	    If True, only compute loss at positions where induction should help
	    (i.e., 2nd+ repetitions). If False, compute over all positions.
	seed
	    Random seed for sequence generation.

	Returns
	-------
	float
	    Mean cross-entropy loss on the specified positions.
	"""
	if sequences is None:
		sequences = generate_repeated_sequences(
			tokenizer=model.tokenizer,
			n_sequences=n_sequences,
			seq_length=seq_length,
			n_repetitions=n_repetitions,
			seed=seed,
			device=str(model.cfg.device),
		)

	# Batch the sequences
	tokens = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	# Get logits
	with torch.no_grad():
		logits = model(tokens)  # (batch, seq_len, vocab_size)

	# Compute loss
	# Shift for next-token prediction: predict token[i+1] from logits[i]
	logits_for_loss = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
	targets = tokens[:, 1:]  # (batch, seq_len-1)

	# Compute per-position loss
	loss_per_pos = F.cross_entropy(
		logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
		targets.reshape(-1),
		reduction="none",
	).reshape(tokens.shape[0], -1)  # (batch, seq_len-1)

	if only_induction_positions:
		# Create mask for induction positions (shifted by 1 for loss alignment)
		mask = get_induction_mask(sequences, pad_to_length=tokens.shape[1])
		mask = mask[:, 1:].to(model.cfg.device)  # Shift to align with loss positions

		# Masked mean
		masked_loss = loss_per_pos * mask.float()
		mean_loss = masked_loss.sum() / mask.sum().clamp(min=1)
	else:
		mean_loss = loss_per_pos.mean()

	return mean_loss.item()


def prefix_matching_score(
	model: "HookedTransformer",
	layer: int,
	head: int,
	sequences: Sequence[RepeatedSequence] | None = None,
	seq_length: int = 25,
	n_repetitions: int = 4,
	n_sequences: int = 50,
	seed: int | None = 42,
) -> float:
	"""Compute prefix matching score for a specific attention head.

	Measures how much the head attends to the "correct" induction position:
	when seeing token A in the 2nd+ repetition, does the head attend to
	the position of A in the previous repetition?

	High score indicates induction-like attention pattern.

	Parameters
	----------
	model
	    TransformerLens model.
	layer
	    Layer index of the head to analyze.
	head
	    Head index within the layer.
	sequences
	    Pre-generated sequences.
	seq_length
	    Length of base pattern.
	n_repetitions
	    Number of repetitions.
	n_sequences
	    Number of sequences.
	seed
	    Random seed.

	Returns
	-------
	float
	    Mean attention weight on the correct prefix-matching position.
	"""
	if sequences is None:
		sequences = generate_repeated_sequences(
			tokenizer=model.tokenizer,
			n_sequences=n_sequences,
			seq_length=seq_length,
			n_repetitions=n_repetitions,
			seed=seed,
			device=str(model.cfg.device),
		)

	tokens = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	# Run model and cache attention patterns
	with torch.no_grad():
		_, cache = model.run_with_cache(
			tokens, names_filter=f"blocks.{layer}.attn.hook_pattern"
		)

	# Get attention pattern: (batch, n_heads, query_pos, key_pos)
	attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head, :, :]
	# Shape: (batch, query_pos, key_pos)

	# Compute prefix matching score
	# For each position in 2nd+ repetition, check attention to the
	# corresponding position in the previous repetition
	scores: list[float] = []

	for batch_idx, seq in enumerate(sequences):
		base_len = seq.base_length
		for rep_idx in range(1, seq.n_repetitions):
			rep_start = seq.repetition_starts[rep_idx]
			prev_start = seq.repetition_starts[rep_idx - 1]

			for offset in range(1, base_len):
				query_pos = rep_start + offset
				# The "correct" key position: same offset in previous repetition
				key_pos = (
					prev_start + offset - 1
				)  # -1 because we want the token that precedes

				if (
					query_pos < attn_pattern.shape[1]
					and key_pos < attn_pattern.shape[2]
				):
					score = attn_pattern[batch_idx, query_pos, key_pos].item()
					scores.append(score)

	return sum(scores) / len(scores) if scores else 0.0


def icl_score(
	model: "HookedTransformer",
	prompts: list[str] | list[Tensor],
	early_pos: int = 50,
	late_pos: int = 500,
) -> float:
	"""Compute in-context learning score.

	Measures the model's ability to learn from context by comparing
	loss at early vs late positions. More negative = better ICL.

	From Olsson et al. 2022: "the average loss at the 500th token in
	context minus the average loss at the 50th token"

	Parameters
	----------
	model
	    TransformerLens model.
	prompts
	    List of text prompts or tokenized sequences.
	    Should be long enough to have tokens at late_pos.
	early_pos
	    Position for early loss measurement.
	late_pos
	    Position for late loss measurement.

	Returns
	-------
	float
	    Late loss minus early loss (negative = model learns from context).
	"""
	early_losses: list[float] = []
	late_losses: list[float] = []

	with torch.no_grad():
		for prompt in prompts:
			if isinstance(prompt, str):
				tokens = model.to_tokens(prompt)
			else:
				tokens = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt

			tokens = tokens.to(model.cfg.device)

			# Skip if sequence too short
			if tokens.shape[1] <= late_pos:
				continue

			logits = model(tokens)

			# Compute loss at specific positions
			# Loss at position i predicts token i+1
			if early_pos < tokens.shape[1] - 1:
				early_logits = logits[:, early_pos, :]
				early_target = tokens[:, early_pos + 1]
				early_loss = F.cross_entropy(early_logits, early_target)
				early_losses.append(early_loss.item())

			if late_pos < tokens.shape[1] - 1:
				late_logits = logits[:, late_pos, :]
				late_target = tokens[:, late_pos + 1]
				late_loss = F.cross_entropy(late_logits, late_target)
				late_losses.append(late_loss.item())

	if not early_losses or not late_losses:
		return 0.0

	mean_early = sum(early_losses) / len(early_losses)
	mean_late = sum(late_losses) / len(late_losses)

	return mean_late - mean_early


def compute_all_metrics(
	model: "HookedTransformer",
	layer: int,
	head: int,
	repeated_sequences: Sequence[RepeatedSequence] | None = None,
	icl_prompts: list[str] | None = None,
	n_sequences: int = 50,
	seed: int | None = 42,
) -> dict[str, float]:
	"""Compute all three metrics for a single head.

	Parameters
	----------
	model
	    TransformerLens model.
	layer
	    Layer index.
	head
	    Head index.
	repeated_sequences
	    Pre-generated repeated sequences.
	icl_prompts
	    Prompts for ICL score (long texts).
	n_sequences
	    Number of sequences to generate if not provided.
	seed
	    Random seed.

	Returns
	-------
	dict
	    Dictionary with 'repeated_loss', 'prefix_score', 'icl_score'.
	"""
	# Generate sequences if needed
	if repeated_sequences is None:
		repeated_sequences = generate_repeated_sequences(
			tokenizer=model.tokenizer,
			n_sequences=n_sequences,
			seed=seed,
			device=str(model.cfg.device),
		)

	# Compute metrics
	rep_loss = repeated_sequence_loss(model, repeated_sequences)
	prefix = prefix_matching_score(model, layer, head, repeated_sequences)

	icl = 0.0
	if icl_prompts:
		icl = icl_score(model, icl_prompts)

	return {
		"repeated_loss": rep_loss,
		"prefix_score": prefix,
		"icl_score": icl,
	}


def copying_score(
	model: "HookedTransformer",
	layer: int,
	head: int,
	sequences: Sequence[RepeatedSequence] | None = None,
	n_sequences: int = 50,
	seed: int | None = 42,
) -> float:
	"""Compute copying score for a head.

	Measures how much the head's output increases the logit of the
	correct "copied" token. This is complementary to prefix_matching_score:
	prefix matching measures WHERE the head attends, copying score
	measures WHAT information it writes.

	Parameters
	----------
	model
	    TransformerLens model.
	layer
	    Layer index.
	head
	    Head index.
	sequences
	    Pre-generated sequences.
	n_sequences
	    Number of sequences if generating.
	seed
	    Random seed.

	Returns
	-------
	float
	    Mean logit increase for the correct copied token.
	"""
	if sequences is None:
		sequences = generate_repeated_sequences(
			tokenizer=model.tokenizer,
			n_sequences=n_sequences,
			seed=seed,
			device=str(model.cfg.device),
		)

	tokens = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	# Get the head's output contribution
	# This requires running with cache and extracting the head's z output
	hook_name = f"blocks.{layer}.attn.hook_z"

	with torch.no_grad():
		_, cache = model.run_with_cache(tokens, names_filter=[hook_name])

	# z: (batch, pos, n_heads, d_head)
	z = cache[hook_name][:, :, head, :]  # (batch, pos, d_head)

	# Project through output matrix to get contribution to residual stream
	# W_O: (n_heads, d_head, d_model) - we need to select the right head
	W_O = model.W_O[layer, head]  # (d_head, d_model)

	# Head's contribution to residual: z @ W_O
	head_contribution = torch.einsum("bpd,dm->bpm", z, W_O)  # (batch, pos, d_model)

	# Project to logits: contribution @ W_U
	W_U = model.W_U  # (d_model, vocab)
	logit_contribution = torch.einsum("bpm,mv->bpv", head_contribution, W_U)

	# For induction positions, check the logit of the correct next token
	scores: list[float] = []

	for batch_idx, seq in enumerate(sequences):
		base_len = seq.base_length
		for rep_idx in range(1, seq.n_repetitions):
			rep_start = seq.repetition_starts[rep_idx]

			for offset in range(base_len - 1):
				pos = rep_start + offset
				next_token = tokens[batch_idx, pos + 1].item()

				if pos < logit_contribution.shape[1]:
					logit_increase = logit_contribution[
						batch_idx, pos, next_token
					].item()
					scores.append(logit_increase)

	return sum(scores) / len(scores) if scores else 0.0
