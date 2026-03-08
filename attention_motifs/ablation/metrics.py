"""Evaluation metrics for induction head ablation studies.

Implements key metrics from the literature:
1. Loss on repeated sequences - core test for induction behavior
2. Prefix matching score - how much head attends to induction positions
3. Preceding token score - attention to token before the matching token
4. In-context learning score - early vs late loss difference
5. Copying score (induction-specific) - logit increase for correct next token
6. OV copying score (paper-style) - general OV-circuit copying tendency
"""

from typing import Sequence

from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)

import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Bool, Float, Int

from attention_motifs.ablation.data import (
	RepeatedSequence,
	generate_repeated_sequences,
	sequences_to_batch,
	get_induction_mask,
)
from attention_motifs.ablation.ablate import AblationMethod

from transformer_lens import HookedTransformer


@serializable_dataclass
class AblationResult(SerializableDataclass):
	"""Results from an ablation experiment on a single head.

	Metrics are split into two categories:

	**Ablation impact** (causal metrics that change when head is ablated):
	- Loss on repeated sequences (baseline vs ablated)

	**Head characterization** (properties of the head, computed once without
	ablation; these are NOT affected by ablation because prefix scores read
	attention patterns (computed before hook_z) and copying scores become
	tautologically zero when z is ablated):
	- Prefix matching score (offset+1, induction)
	- Prefix matching score (offset-1, legacy)
	- Copying score (induction-specific)
	- OV copying score (paper-style)

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
	prefix_score
	    Prefix matching score (offset+1, induction). Head characterization.
	prefix_score_legacy
	    Preceding token score (offset-1). Head characterization.
	copying_score
	    Induction-specific copying score. Head characterization.
	ov_copying_score
	    Paper-style OV copying score. Head characterization.
	baseline_icl_score
	    In-context learning score without ablation. None if not measured.
	ablated_icl_score
	    In-context learning score with ablation. None if not measured.
	icl_degradation
	    How much worse ICL became (more positive = worse). None if not measured.
	"""

	head: str
	ablation_method: AblationMethod = serializable_field(
		serialization_fn=lambda x: x.value,
		deserialize_fn=lambda x: AblationMethod(x),
	)
	# Ablation impact (causal)
	baseline_repeated_loss: float
	ablated_repeated_loss: float
	loss_increase: float
	# Head characterization (not affected by ablation)
	prefix_score: float
	prefix_score_legacy: float = 0.0
	copying_score: float = 0.0
	ov_copying_score: float = 0.0
	# ICL — None means not measured (no prompts provided)
	baseline_icl_score: float | None = serializable_field(default=None)
	ablated_icl_score: float | None = serializable_field(default=None)
	icl_degradation: float | None = serializable_field(default=None)


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
	tokens: Int[Tensor, "batch seq"] = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	# Determine whether sequences already have BOS
	has_bos: bool = sequences[0].has_bos if sequences else False

	# Get logits
	with torch.no_grad():
		logits: Float[Tensor, "batch seq vocab"] = model(
			tokens, prepend_bos=not has_bos
		)

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
		logits_for_loss: Float[Tensor, "batch pos vocab"] = logits[
			:, 1:-1, :
		]  # skip BOS, align with tokens
		targets: Int[Tensor, "batch pos"] = tokens[
			:, 1:
		]  # predict tokens[1] from logits[1], etc.
		# Trim to match
		min_len: int = min(logits_for_loss.shape[1], targets.shape[1])
		logits_for_loss = logits_for_loss[:, :min_len, :]
		targets = targets[:, :min_len]
	else:
		# Sequences already have BOS, no extra shift
		logits_for_loss = logits[:, :-1, :]
		targets = tokens[:, 1:]

	# Compute per-position loss
	loss_per_pos: Float[Tensor, "batch pos"] = F.cross_entropy(
		logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
		targets.reshape(-1),
		reduction="none",
	).reshape(tokens.shape[0], -1)

	if only_induction_positions:
		# Create mask for induction positions
		mask: Bool[Tensor, "batch seq"] = get_induction_mask(
			sequences, pad_to_length=tokens.shape[1]
		)
		# Shift mask by 1 to align with loss positions: loss[i] predicts token[i+1]
		mask = mask[:, 1:].to(model.cfg.device)

		# Trim to match loss_per_pos
		mask = mask[:, : loss_per_pos.shape[1]]

		# Masked mean
		masked_loss: Float[Tensor, "batch pos"] = loss_per_pos * mask.float()
		mean_loss: Float[Tensor, ""] = masked_loss.sum() / mask.sum().clamp(min=1)
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
	tokens: Int[Tensor, "batch seq"] = sequences_to_batch(sequences)
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
	attn_pattern: Float[Tensor, "batch query key"] = cache[
		f"blocks.{layer}.attn.hook_pattern"
	][:, head, :, :]
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

	.. note:: **Divergence from Olsson et al. 2022.**

	   The paper's "prefix matching" evaluator (§ Methods, Head activation
	   evaluators) measures attention to "the tokens that preceded the same
	   token in earlier repeats" — i.e. offset **-1** from the earlier
	   occurrence.  The paper's informal definition is consistent: "does the
	   head attend to earlier tokens that are *followed by* a token that
	   matches the present token?"

	   This function uses offset **+1** instead: for query token A in rep 2,
	   it measures attention to the token *after* A in rep 1 (= B, the
	   token the induction head should copy).  This matches the actual
	   K-composition mechanism — a previous-token head writes "A preceded
	   me" into position B's residual stream, and the induction head's key
	   at B matches the query at A — and is what TransformerLens's
	   ``get_induction_head_detection_pattern`` computes (via
	   ``torch.roll(duplicate_pattern, shifts=1, dims=1)``).

	   The paper's literal offset-1 metric is available as
	   :func:`preceding_token_score`; both are tracked in
	   :class:`AblationResult` (``prefix_score`` for offset+1,
	   ``prefix_score_legacy`` for offset-1).

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

	This is the literal metric from the Olsson et al. 2022 "prefix matching"
	evaluator (§ Methods, Head activation evaluators): "the average of all
	attention pattern entries attending from a given token back to the
	tokens that preceded the same token in earlier repeats" — offset **-1**.
	See :func:`prefix_matching_score` for why that function uses offset+1
	instead and how the two relate.

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
	    Tensor inputs should already include BOS if the model expects it
	    (``default_prepend_bos=True``), since TransformerLens only
	    auto-prepends BOS for string inputs.  If a tensor is missing BOS,
	    it will be prepended automatically.
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
				# to_tokens() prepends BOS when default_prepend_bos=True
				tokens: Int[Tensor, "1 seq"] = model.to_tokens(prompt)
			else:
				tokens = prompt.unsqueeze(0) if prompt.dim() == 1 else prompt
				# Tensor inputs bypass TransformerLens BOS prepending
				# (prepend_bos only affects string inputs in HookedTransformer.forward).
				# Prepend BOS here if model expects it and tensor doesn't have it,
				# so tensor and string paths produce equivalent model contexts.
				if model.cfg.default_prepend_bos:
					bos_id: int = getattr(model.tokenizer, "bos_token_id", None) or 0
					if tokens.shape[1] == 0 or tokens[0, 0].item() != bos_id:
						bos_tensor: Int[Tensor, "batch 1"] = torch.full(
							(tokens.shape[0], 1),
							bos_id,
							dtype=tokens.dtype,
							device=tokens.device,
						)
						tokens = torch.cat([bos_tensor, tokens], dim=1)

			tokens = tokens.to(model.cfg.device)

			# Skip if sequence too short
			if tokens.shape[1] <= late_pos:
				continue

			logits: Float[Tensor, "1 seq vocab"] = model(tokens)

			# Compute loss at specific positions
			# Loss at position i predicts token i+1
			if early_pos < tokens.shape[1] - 1:
				early_logits: Float[Tensor, "1 vocab"] = logits[:, early_pos, :]
				early_target: Int[Tensor, " 1"] = tokens[:, early_pos + 1]
				early_loss: Float[Tensor, ""] = F.cross_entropy(
					early_logits, early_target
				)
				early_losses.append(early_loss.item())

			if late_pos < tokens.shape[1] - 1:
				late_logits: Float[Tensor, "1 vocab"] = logits[:, late_pos, :]
				late_target: Int[Tensor, " 1"] = tokens[:, late_pos + 1]
				late_loss: Float[Tensor, ""] = F.cross_entropy(late_logits, late_target)
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

	tokens: Int[Tensor, "batch seq"] = sequences_to_batch(sequences)
	tokens = tokens.to(model.cfg.device)

	has_bos: bool = sequences[0].has_bos if sequences else False

	# Get the head's output contribution
	hook_name: str = f"blocks.{layer}.attn.hook_z"

	with torch.no_grad():
		_, cache = model.run_with_cache(
			tokens, names_filter=[hook_name], prepend_bos=not has_bos
		)

	# z: (batch, pos, n_heads, d_head)
	z: Float[Tensor, "batch pos d_head"] = cache[hook_name][:, :, head, :]
	del cache

	# Project through output matrix to get contribution to residual stream
	W_O: Float[Tensor, "d_head d_model"] = model.W_O[layer, head]

	# Head's contribution to residual: z @ W_O
	head_contribution: Float[Tensor, "batch pos d_model"] = torch.einsum(
		"bpd,dm->bpm", z, W_O
	)
	del z

	# Project to logits: contribution @ W_U
	W_U: Float[Tensor, "d_model vocab"] = model.W_U
	logit_contribution: Float[Tensor, "batch pos vocab"] = torch.einsum(
		"bpm,mv->bpv", head_contribution, W_U
	)
	del head_contribution

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
					next_token: int = int(tokens[batch_idx, next_token_pos].item())
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

	From Olsson et al. 2022 (§ Methods, Head activation evaluators,
	"Copying"): compute the head's logit contribution, subtract mean,
	apply ReLU, then "compute the ratio of the amount it raises the logits
	of the token being attended to, to that of all tokens in this sample.
	This value ranges from 0 (only raises other tokens) to 0.5 (only
	raises the present token), so we scale it into the range of -1 to 1."

	.. note::

	   The paper's copying evaluator specifies a single *non-repeated*
	   sequence of 25 random tokens, while in practice this function
	   receives the same repeated sequences used by the other metrics.
	   The repetition structure affects the head's attention pattern
	   (induction heads will attend strongly to offset+1 positions),
	   making this score somewhat induction-specific rather than a pure
	   OV-circuit measure.  See :func:`copying_score` for an explicitly
	   induction-specific alternative.

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

	tokens: Int[Tensor, "batch seq"] = sequences_to_batch(sequences)
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
		del cache

		# Head contribution to logits
		W_O: Float[Tensor, "d_head d_model"] = model.W_O[layer, head]
		W_U: Float[Tensor, "d_model vocab"] = model.W_U
		head_logits: Float[Tensor, "batch pos vocab"] = torch.einsum(
			"bpd,dm,mv->bpv", z, W_O, W_U
		)
		del z

		# Subtract per-position mean, apply ReLU (paper: "subtracting the mean
		# of the logits and passing through a ReLU")
		mean_logits: Float[Tensor, "batch pos 1"] = head_logits.mean(
			dim=-1, keepdim=True
		)
		positive_logits: Float[Tensor, "batch pos vocab"] = F.relu(
			head_logits - mean_logits
		)
		del head_logits, mean_logits

		# For each (batch, dest_pos): compute attention-weighted fraction of
		# logit mass that goes to the attended-to token.
		#
		# attended_logit = sum_j attn[b, q, j] * positive_logits[b, q, tokens[b, j]]
		# total_logit    = sum_{t in sample} positive_logits[b, q, t]
		# raw_ratio      = attended_logit / total_logit   (in [0, 0.5] for a copying head)
		# score          = 2 * raw_ratio - 1              (scaled to [-1, 1])
		#
		# Paper: "ratio ... to that of all tokens in this sample" — the
		# denominator sums over the unique token types present in the
		# sequence, not the full vocabulary.

		batch_size: int = tokens.shape[0]
		seq_len: int = attn.shape[1]  # dest positions (may include model BOS)

		# Determine the token tensor that aligns with attn positions
		if not has_bos:
			# Model added BOS at position 0; attn has seq_len = tokens.shape[1] + 1
			# Build aligned token tensor with BOS prepended
			bos_id: int = getattr(model.tokenizer, "bos_token_id", None) or 0
			bos_col: Int[Tensor, "batch 1"] = torch.full(
				(batch_size, 1), bos_id, dtype=tokens.dtype, device=tokens.device
			)
			aligned_tokens: Int[Tensor, "batch seq"] = torch.cat(
				[bos_col, tokens], dim=1
			)
			# Trim to match attn dim
			aligned_tokens = aligned_tokens[:, :seq_len]
		else:
			aligned_tokens = tokens[:, :seq_len]

		# Vectorized computation:
		# For each src position j, gather the logit for token at j
		src_len: int = attn.shape[2]
		src_tokens: Int[Tensor, "batch src"] = aligned_tokens[:, :src_len]

		# Gather logits for attended-to tokens: positive_logits[b, q, src_tokens[b, j]]
		attended_logits: Float[Tensor, "batch dest src"] = torch.zeros(
			batch_size, seq_len, src_len, device=tokens.device
		)
		for q in range(seq_len):
			attended_logits[:, q, :] = torch.gather(
				positive_logits[:, q, :], dim=1, index=src_tokens.long()
			)

		# Attention-weighted sum of attended logits per (batch, dest)
		weighted_attended: Float[Tensor, "batch dest"] = (attn * attended_logits).sum(
			dim=-1
		)
		del attended_logits

		# Total positive logit mass over sample tokens per (batch, dest).
		# Paper: "to that of all tokens in this sample" — sum only over the
		# unique token types present in each sequence, not the full vocabulary.
		total_positive: Float[Tensor, "batch dest"] = torch.zeros(
			batch_size, seq_len, device=tokens.device
		)
		for b in range(batch_size):
			unique_b: Int[Tensor, " n_unique"] = aligned_tokens[b].unique()
			total_positive[b] = positive_logits[b, :, unique_b].sum(dim=-1)
		del positive_logits

		# Raw ratio (avoid division by zero)
		valid_mask: Bool[Tensor, "batch dest"] = total_positive > 1e-10
		raw_ratio: Float[Tensor, "batch dest"] = torch.zeros_like(weighted_attended)
		raw_ratio[valid_mask] = (
			weighted_attended[valid_mask] / total_positive[valid_mask]
		)

		# Scale to [-1, 1]: score = 2 * ratio - 1
		scaled: Float[Tensor, "batch dest"] = 2.0 * raw_ratio - 1.0

		# Average over valid positions (skip position 0 which is BOS)
		start_pos: int = 1
		valid_scores: Float[Tensor, "batch pos"] = scaled[:, start_pos:]
		valid_counts: Float[Tensor, "batch pos"] = valid_mask[:, start_pos:].float()

		if valid_counts.sum() > 0:
			return (
				valid_scores * valid_counts
			).sum().item() / valid_counts.sum().item()
		return 0.0
