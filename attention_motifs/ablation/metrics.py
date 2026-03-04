"""Evaluation metrics for induction head ablation studies.

Implements key metrics from the literature:
1. Loss on repeated sequences - core test for induction behavior
2. Prefix matching score - how much head attends to induction positions
3. Preceding token score - attention to token before the matching token
4. In-context learning score - early vs late loss difference
5. Copying score (induction-specific) - logit increase for correct next token
6. OV copying score (paper-style) - general OV-circuit copying tendency
"""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

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
	    Prefix matching score (offset+1, induction) without ablation.
	ablated_prefix_score
	    Prefix matching score with ablation.
	prefix_score_decrease
	    Difference: baseline - ablated (positive = head was doing induction).
	baseline_prefix_score_legacy
	    Preceding token score (offset-1) without ablation.
	ablated_prefix_score_legacy
	    Preceding token score with ablation.
	prefix_score_decrease_legacy
	    Difference: baseline - ablated for legacy prefix score.
	baseline_icl_score
	    In-context learning score without ablation.
	ablated_icl_score
	    In-context learning score with ablation.
	icl_degradation
	    How much worse ICL became (more negative = worse).
	copying_score
	    Induction-specific copying score (baseline).
	ablated_copying_score
	    Induction-specific copying score (ablated).
	copying_score_decrease
	    Difference: baseline - ablated.
	ov_copying_score
	    Paper-style OV copying score (baseline).
	ablated_ov_copying_score
	    Paper-style OV copying score (ablated).
	ov_copying_score_decrease
	    Difference: baseline - ablated.
	"""

	head: str
	ablation_method: AblationMethod
	# Repeated sequence loss
	baseline_repeated_loss: float
	ablated_repeated_loss: float
	loss_increase: float
	# Prefix matching (offset+1, induction)
	baseline_prefix_score: float
	ablated_prefix_score: float
	prefix_score_decrease: float
	# Preceding token (offset-1, legacy)
	baseline_prefix_score_legacy: float = 0.0
	ablated_prefix_score_legacy: float = 0.0
	prefix_score_decrease_legacy: float = 0.0
	# ICL
	baseline_icl_score: float = 0.0
	ablated_icl_score: float = 0.0
	icl_degradation: float = 0.0
	# Copying score (induction-specific)
	copying_score: float = 0.0
	ablated_copying_score: float = 0.0
	copying_score_decrease: float = 0.0
	# OV copying score (paper-style)
	ov_copying_score: float = 0.0
	ablated_ov_copying_score: float = 0.0
	ov_copying_score_decrease: float = 0.0

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
			"baseline_prefix_score_legacy": self.baseline_prefix_score_legacy,
			"ablated_prefix_score_legacy": self.ablated_prefix_score_legacy,
			"prefix_score_decrease_legacy": self.prefix_score_decrease_legacy,
			"baseline_icl_score": self.baseline_icl_score,
			"ablated_icl_score": self.ablated_icl_score,
			"icl_degradation": self.icl_degradation,
			"copying_score": self.copying_score,
			"ablated_copying_score": self.ablated_copying_score,
			"copying_score_decrease": self.copying_score_decrease,
			"ov_copying_score": self.ov_copying_score,
			"ablated_ov_copying_score": self.ablated_ov_copying_score,
			"ov_copying_score_decrease": self.ov_copying_score_decrease,
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
	tokens: Tensor = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	# Determine whether sequences already have BOS
	has_bos: bool = sequences[0].has_bos if sequences else False

	# Get logits
	with torch.no_grad():
		logits: Tensor = model(tokens, prepend_bos=not has_bos)

	# If model prepended BOS and our sequences didn't have it, logits are
	# shifted by 1.  But if our sequences already have BOS and we told the
	# model not to prepend, dimensions match.
	# With prepend_bos=not has_bos:
	#   - has_bos=True  -> prepend_bos=False -> logits align with tokens
	#   - has_bos=False -> prepend_bos=True  -> logits have extra BOS position
	if not has_bos:
		# Model prepended BOS, so logits[:, 0] corresponds to BOS.
		# logits[:, 1:] correspond to our tokens.
		# For next-token prediction: predict tokens[i] from logits[i] (with BOS shift)
		logits_for_loss: Tensor = logits[:, 1:-1, :]  # skip BOS, align with tokens
		targets: Tensor = tokens[:, 1:]  # predict tokens[1] from logits[1], etc.
		# Trim to match
		min_len: int = min(logits_for_loss.shape[1], targets.shape[1])
		logits_for_loss = logits_for_loss[:, :min_len, :]
		targets = targets[:, :min_len]
	else:
		# Sequences already have BOS, no extra shift
		logits_for_loss = logits[:, :-1, :]
		targets = tokens[:, 1:]

	# Compute per-position loss
	loss_per_pos: Tensor = F.cross_entropy(
		logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
		targets.reshape(-1),
		reduction="none",
	).reshape(tokens.shape[0], -1)

	if only_induction_positions:
		# Create mask for induction positions
		mask: Tensor = get_induction_mask(sequences, pad_to_length=tokens.shape[1])
		# Shift mask by 1 to align with loss positions: loss[i] predicts token[i+1]
		mask = mask[:, 1:].to(model.cfg.device)

		# Trim to match loss_per_pos
		mask = mask[:, : loss_per_pos.shape[1]]

		# Masked mean
		masked_loss: Tensor = loss_per_pos * mask.float()
		mean_loss: Tensor = masked_loss.sum() / mask.sum().clamp(min=1)
	else:
		mean_loss = loss_per_pos.mean()

	return mean_loss.item()


def _prefix_score_impl(
	model: "HookedTransformer",
	layer: int,
	head: int,
	sequences: Sequence[RepeatedSequence],
	offset_delta: int,
) -> float:
	"""Shared implementation for prefix matching / preceding token scores.

	Parameters
	----------
	model
	    TransformerLens model.
	layer
	    Layer index of the head to analyze.
	head
	    Head index within the layer.
	sequences
	    Pre-generated repeated sequences.
	offset_delta
	    Offset from the matching token position in the previous repetition.
	    ``+1`` for induction prefix matching (token AFTER matching token),
	    ``-1`` for preceding token score (token BEFORE matching token).

	Returns
	-------
	float
	    Mean attention weight on the target position.
	"""
	tokens: Tensor = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	has_bos: bool = sequences[0].has_bos if sequences else False

	# Run model and cache attention patterns
	with torch.no_grad():
		_, cache = model.run_with_cache(
			tokens,
			names_filter=f"blocks.{layer}.attn.hook_pattern",
			prepend_bos=not has_bos,
		)

	# Get attention pattern: (batch, n_heads, query_pos, key_pos)
	# When has_bos=False and prepend_bos=True, positions are shifted by 1
	# (position 0 is the model-inserted BOS).
	attn_pattern: Tensor = cache[f"blocks.{layer}.attn.hook_pattern"][:, head, :, :]
	# Shape: (batch, query_pos, key_pos)

	# Position offset: if model prepended BOS, all token positions shift by 1
	pos_shift: int = 0 if has_bos else 1

	scores: list[float] = []

	for batch_idx, seq in enumerate(sequences):
		base_len: int = seq.base_length
		for rep_idx in range(1, seq.n_repetitions):
			rep_start: int = seq.repetition_starts[rep_idx]
			prev_start: int = seq.repetition_starts[rep_idx - 1]

			# Determine valid offset range based on offset_delta
			if offset_delta > 0:
				# offset+1: offset 0 is valid (key_pos = prev_start + 1),
				# but offset = base_len - 1 would give key_pos = prev_start + base_len
				# which is the start of the next repetition — skip it.
				offset_range: range = range(0, base_len - 1)
			else:
				# offset-1: offset 0 would give key_pos = prev_start - 1 (before
				# repetition start), so start from offset 1.
				offset_range = range(1, base_len)

			for offset in offset_range:
				query_pos: int = rep_start + offset + pos_shift
				key_pos: int = prev_start + offset + offset_delta + pos_shift

				if (
					0 <= key_pos < attn_pattern.shape[2]
					and query_pos < attn_pattern.shape[1]
				):
					score: float = attn_pattern[batch_idx, query_pos, key_pos].item()
					scores.append(score)

	return sum(scores) / len(scores) if scores else 0.0


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
	"""Compute induction prefix matching score for a specific attention head.

	Measures whether the head attends to the token that induction would
	predict comes next.  On repeated sequence ``[A B C][A B C]``, at
	position B (2nd rep) the head should attend to C in the 1st rep
	(the token *after* the previous B).

	This corresponds to the K-composition induction mechanism and matches
	TransformerLens's ``get_induction_head_detection_pattern``.

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
	    Mean attention weight on the correct induction position (offset+1).
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

	return _prefix_score_impl(model, layer, head, sequences, offset_delta=+1)


def preceding_token_score(
	model: "HookedTransformer",
	layer: int,
	head: int,
	sequences: Sequence[RepeatedSequence] | None = None,
	seq_length: int = 25,
	n_repetitions: int = 4,
	n_sequences: int = 50,
	seed: int | None = 42,
) -> float:
	"""Compute preceding-token attention score for a specific head.

	Measures whether the head attends to the token that *preceded* the
	current token in earlier repetitions.  On repeated ``[A B C][A B C]``,
	at position B (2nd rep) the score checks attention to A in the 1st rep
	(the token *before* the previous B).

	This is the metric described in the paper's Methods section as "tokens
	that preceded the same token in earlier repeats" (offset-1).  It may
	capture Q-composition or other non-standard induction patterns.

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
	    Mean attention weight on the preceding-token position (offset-1).
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

	return _prefix_score_impl(model, layer, head, sequences, offset_delta=-1)


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
				tokens: Tensor = model.to_tokens(prompt)
			else:
				tokens = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt

			tokens = tokens.to(model.cfg.device)

			# Skip if sequence too short
			if tokens.shape[1] <= late_pos:
				continue

			logits: Tensor = model(tokens)

			# Compute loss at specific positions
			# Loss at position i predicts token i+1
			if early_pos < tokens.shape[1] - 1:
				early_logits: Tensor = logits[:, early_pos, :]
				early_target: Tensor = tokens[:, early_pos + 1]
				early_loss: Tensor = F.cross_entropy(early_logits, early_target)
				early_losses.append(early_loss.item())

			if late_pos < tokens.shape[1] - 1:
				late_logits: Tensor = logits[:, late_pos, :]
				late_target: Tensor = tokens[:, late_pos + 1]
				late_loss: Tensor = F.cross_entropy(late_logits, late_target)
				late_losses.append(late_loss.item())

	if not early_losses or not late_losses:
		return 0.0

	mean_early: float = sum(early_losses) / len(early_losses)
	mean_late: float = sum(late_losses) / len(late_losses)

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
	"""Compute all metrics for a single head.

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
	    Dictionary with all metric values.
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
	rep_loss: float = repeated_sequence_loss(model, repeated_sequences)
	prefix: float = prefix_matching_score(model, layer, head, repeated_sequences)
	preceding: float = preceding_token_score(model, layer, head, repeated_sequences)
	copy: float = copying_score(model, layer, head, repeated_sequences)
	ov_copy: float = ov_copying_score(model, layer, head, repeated_sequences)

	icl: float = 0.0
	if icl_prompts:
		icl = icl_score(model, icl_prompts)

	return {
		"repeated_loss": rep_loss,
		"prefix_score": prefix,
		"preceding_token_score": preceding,
		"copying_score": copy,
		"ov_copying_score": ov_copy,
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
	"""Compute induction-specific copying score for a head.

	Measures how much the head's output increases the logit of the
	correct "copied" token at induction positions on repeated sequences.
	This is complementary to ``prefix_matching_score``: prefix matching
	measures WHERE the head attends, copying score measures WHAT
	information it writes.

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

	tokens: Tensor = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	has_bos: bool = sequences[0].has_bos if sequences else False

	# Get the head's output contribution
	hook_name: str = f"blocks.{layer}.attn.hook_z"

	with torch.no_grad():
		_, cache = model.run_with_cache(
			tokens, names_filter=[hook_name], prepend_bos=not has_bos
		)

	# z: (batch, pos, n_heads, d_head)
	z: Tensor = cache[hook_name][:, :, head, :]  # (batch, pos, d_head)

	# Project through output matrix to get contribution to residual stream
	W_O: Tensor = model.W_O[layer, head]  # (d_head, d_model)

	# Head's contribution to residual: z @ W_O
	head_contribution: Tensor = torch.einsum(
		"bpd,dm->bpm", z, W_O
	)  # (batch, pos, d_model)

	# Project to logits: contribution @ W_U
	W_U: Tensor = model.W_U  # (d_model, vocab)
	logit_contribution: Tensor = torch.einsum("bpm,mv->bpv", head_contribution, W_U)

	pos_shift: int = 0 if has_bos else 1

	# For induction positions, check the logit of the correct next token
	scores: list[float] = []

	for batch_idx, seq in enumerate(sequences):
		base_len: int = seq.base_length
		for rep_idx in range(1, seq.n_repetitions):
			rep_start: int = seq.repetition_starts[rep_idx]

			for offset in range(base_len - 1):
				pos: int = rep_start + offset
				next_token_pos: int = pos + 1
				# Get next token from the original (un-shifted) sequence
				if next_token_pos < tokens.shape[1]:
					next_token: int = tokens[batch_idx, next_token_pos].item()
					# logit_contribution position is shifted if model added BOS
					logit_pos: int = pos + pos_shift
					if logit_pos < logit_contribution.shape[1]:
						logit_increase: float = logit_contribution[
							batch_idx, logit_pos, int(next_token)
						].item()
						scores.append(logit_increase)

	return sum(scores) / len(scores) if scores else 0.0


def ov_copying_score(
	model: "HookedTransformer",
	layer: int,
	head: int,
	sequences: Sequence[RepeatedSequence] | None = None,
	n_sequences: int = 50,
	seq_length: int = 25,
	seed: int | None = 42,
) -> float:
	"""Compute paper-style OV copying score.

	Measures the general tendency of a head to copy the token it attends
	to into its output, regardless of sequence structure.

	From Olsson et al. 2022: compute the head's logit contribution,
	subtract mean, apply ReLU, then measure the fraction of positive
	logit mass allocated to the attended-to token.  Scaled to [-1, 1].

	Parameters
	----------
	model
	    TransformerLens model.
	layer
	    Layer index.
	head
	    Head index.
	sequences
	    Pre-generated sequences (uses the token content only).
	n_sequences
	    Number of sequences if generating.
	seq_length
	    Base sequence length if generating.
	seed
	    Random seed.

	Returns
	-------
	float
	    OV copying score in [-1, 1].  Higher = stronger copying behavior.
	"""
	if sequences is None:
		sequences = generate_repeated_sequences(
			tokenizer=model.tokenizer,
			n_sequences=n_sequences,
			seq_length=seq_length,
			seed=seed,
			device=str(model.cfg.device),
		)

	tokens: Tensor = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	has_bos: bool = sequences[0].has_bos if sequences else False

	hook_z_name: str = f"blocks.{layer}.attn.hook_z"
	hook_pattern_name: str = f"blocks.{layer}.attn.hook_pattern"

	with torch.no_grad():
		_, cache = model.run_with_cache(
			tokens,
			names_filter=[hook_z_name, hook_pattern_name],
			prepend_bos=not has_bos,
		)

	# z: (batch, pos, d_head)
	z: Float[Tensor, "batch pos d_head"] = cache[hook_z_name][:, :, head, :]
	# attn: (batch, dest_pos, src_pos)
	attn: Float[Tensor, "batch dest src"] = cache[hook_pattern_name][:, head, :, :]

	# Head contribution to logits
	W_O: Tensor = model.W_O[layer, head]  # (d_head, d_model)
	W_U: Tensor = model.W_U  # (d_model, vocab)
	head_logits: Float[Tensor, "batch pos vocab"] = torch.einsum(
		"bpd,dm,mv->bpv", z, W_O, W_U
	)

	# Subtract per-position mean, apply ReLU (paper: "subtracting the mean
	# of the logits and passing through a ReLU")
	mean_logits: Float[Tensor, "batch pos 1"] = head_logits.mean(dim=-1, keepdim=True)
	positive_logits: Float[Tensor, "batch pos vocab"] = F.relu(
		head_logits - mean_logits
	)

	# For each (batch, dest_pos): compute attention-weighted fraction of
	# logit mass that goes to the attended-to token.
	#
	# attended_logit = sum_j attn[b, q, j] * positive_logits[b, q, tokens[b, j]]
	# total_logit    = sum_v positive_logits[b, q, v]
	# raw_ratio      = attended_logit / total_logit   (in [0, 0.5] for a copying head)
	# score          = 2 * raw_ratio - 1              (scaled to [-1, 1])

	batch_size: int = tokens.shape[0]
	seq_len: int = attn.shape[1]  # dest positions (may include model BOS)

	# Determine the token tensor that aligns with attn positions
	if not has_bos:
		# Model added BOS at position 0; attn has seq_len = tokens.shape[1] + 1
		# Build aligned token tensor with BOS prepended
		bos_id: int = getattr(model.tokenizer, "bos_token_id", None) or 0
		bos_col: Tensor = torch.full(
			(batch_size, 1), bos_id, dtype=tokens.dtype, device=tokens.device
		)
		aligned_tokens: Tensor = torch.cat([bos_col, tokens], dim=1)
		# Trim to match attn dim
		aligned_tokens = aligned_tokens[:, :seq_len]
	else:
		aligned_tokens = tokens[:, :seq_len]

	# Vectorized computation:
	# For each src position j, gather the logit for token at j
	# src_tokens shape: (batch, src_len) -> expand to (batch, dest, src)
	src_len: int = attn.shape[2]
	src_tokens: Tensor = aligned_tokens[:, :src_len]  # (batch, src_len)

	# Gather logits for attended-to tokens: positive_logits[b, q, src_tokens[b, j]]
	# Expand src_tokens to (batch, dest, src) for gathering
	src_tokens_expanded: Tensor = src_tokens.unsqueeze(1).expand(
		batch_size, seq_len, src_len
	)

	# positive_logits: (batch, dest, vocab) -> gather along vocab dim
	# We need positive_logits[b, q, src_tokens[b, j]] for each (b, q, j)
	# Reshape for gathering: (batch * dest, vocab) x (batch * dest, src) doesn't work
	# Instead: for each dest position, gather src token logits
	# positive_logits[:, :, :].gather(2, ...) — need to index vocab dim

	# Approach: expand positive_logits to (batch, dest, src) by gathering vocab dim
	# at src_tokens indices
	# For each (b, q, j): logit = positive_logits[b, q, src_tokens_expanded[b, q, j]]
	attended_logits: Tensor = torch.zeros(
		batch_size, seq_len, src_len, device=tokens.device
	)
	for q in range(seq_len):
		# positive_logits[:, q, :] shape: (batch, vocab)
		# src_tokens shape: (batch, src_len)
		attended_logits[:, q, :] = torch.gather(
			positive_logits[:, q, :], dim=1, index=src_tokens.long()
		)

	# Attention-weighted sum of attended logits per (batch, dest)
	# attn: (batch, dest, src), attended_logits: (batch, dest, src)
	weighted_attended: Tensor = (attn * attended_logits).sum(dim=-1)  # (batch, dest)

	# Total positive logit mass per (batch, dest)
	total_positive: Tensor = positive_logits.sum(dim=-1)  # (batch, dest)

	# Raw ratio (avoid division by zero)
	valid_mask: Tensor = total_positive > 1e-10
	raw_ratio: Tensor = torch.zeros_like(weighted_attended)
	raw_ratio[valid_mask] = weighted_attended[valid_mask] / total_positive[valid_mask]

	# Scale to [-1, 1]: score = 2 * ratio - 1
	scaled: Tensor = 2.0 * raw_ratio - 1.0

	# Average over valid positions (skip position 0 which is BOS)
	start_pos: int = 1
	valid_scores: Tensor = scaled[:, start_pos:]
	valid_counts: Tensor = valid_mask[:, start_pos:].float()

	if valid_counts.sum() > 0:
		return (valid_scores * valid_counts).sum().item() / valid_counts.sum().item()
	return 0.0
