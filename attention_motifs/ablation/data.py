"""Synthetic data generation for induction head ablation studies.

Generates sequences designed to elicit and test induction behavior:
- Repeated random token sequences ([A B C][A B C]...)
- Natural text with repetitions
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from jaxtyping import Int


@dataclass
class RepeatedSequence:
	"""A sequence with repeated patterns for induction testing.

	Attributes
	----------
	tokens
	    Full token sequence including all repetitions.
	base_length
	    Length of the base pattern before repetition.
	n_repetitions
	    Number of times the pattern is repeated.
	repetition_starts
	    Token indices where each repetition begins.
	"""

	tokens: Int[Tensor, "seq_len"]
	base_length: int
	n_repetitions: int
	repetition_starts: list[int]

	@property
	def seq_len(self) -> int:
		return self.tokens.shape[0]

	def get_induction_positions(self) -> list[int]:
		"""Get positions where induction should help prediction.

		These are positions in the 2nd+ repetition where the model
		could use the previous occurrence to predict the next token.

		Returns
		-------
		list[int]
		    Token positions (after position 0 of each repetition).
		"""
		positions: list[int] = []
		for rep_idx, start in enumerate(self.repetition_starts):
			if rep_idx == 0:
				continue  # Skip first repetition
			# All positions after the start of this repetition
			for pos in range(start + 1, min(start + self.base_length, self.seq_len)):
				positions.append(pos)
		return positions


def generate_repeated_sequences(
	tokenizer,
	n_sequences: int = 100,
	seq_length: int = 25,
	n_repetitions: int = 4,
	exclude_special: bool = True,
	exclude_rare: bool = True,
	rare_threshold: int = 1000,
	seed: int | None = None,
	device: str = "cpu",
) -> list[RepeatedSequence]:
	"""Generate random token sequences repeated multiple times.

	Creates sequences of the form [A1 A2 ... An][A1 A2 ... An]...
	where A1...An are random tokens. This is the standard test for
	induction heads: they should learn to predict Ai+1 after seeing Ai
	in the repeated portion.

	Parameters
	----------
	tokenizer
	    HuggingFace or TransformerLens tokenizer.
	n_sequences
	    Number of sequences to generate.
	seq_length
	    Length of the base pattern (before repetition).
	n_repetitions
	    Number of times to repeat the pattern.
	exclude_special
	    Exclude special tokens (BOS, EOS, PAD, etc.).
	exclude_rare
	    Exclude rare tokens (high token IDs).
	rare_threshold
	    Token IDs above vocab_size - rare_threshold are considered rare.
	seed
	    Random seed for reproducibility.
	device
	    Device to place tensors on.

	Returns
	-------
	list[RepeatedSequence]
	    List of generated sequences with metadata.
	"""
	if seed is not None:
		torch.manual_seed(seed)

	# Get vocab size
	vocab_size = (
		tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
	)

	# Build set of valid token IDs
	valid_tokens: list[int] = list(range(vocab_size))

	if exclude_special:
		# Common special token attributes
		special_ids: set[int] = set()
		for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
			token_id = getattr(tokenizer, attr, None)
			if token_id is not None:
				special_ids.add(token_id)

		# Also check for special_tokens_map
		if hasattr(tokenizer, "all_special_ids"):
			special_ids.update(tokenizer.all_special_ids)

		valid_tokens = [t for t in valid_tokens if t not in special_ids]

	if exclude_rare:
		# Exclude very high token IDs (often rare/special)
		max_token = vocab_size - rare_threshold
		valid_tokens = [t for t in valid_tokens if t < max_token]

	valid_tokens_tensor = torch.tensor(valid_tokens, device=device)
	n_valid = len(valid_tokens_tensor)

	sequences: list[RepeatedSequence] = []

	for _ in range(n_sequences):
		# Sample random indices into valid_tokens
		indices = torch.randint(0, n_valid, (seq_length,), device=device)
		base_tokens = valid_tokens_tensor[indices]

		# Repeat the sequence
		full_tokens = base_tokens.repeat(n_repetitions)

		# Track repetition starts
		repetition_starts = [i * seq_length for i in range(n_repetitions)]

		sequences.append(
			RepeatedSequence(
				tokens=full_tokens,
				base_length=seq_length,
				n_repetitions=n_repetitions,
				repetition_starts=repetition_starts,
			)
		)

	return sequences


def generate_abab_sequences(
	tokenizer,
	n_sequences: int = 100,
	pattern_length: int = 2,
	n_repetitions: int = 10,
	seed: int | None = None,
	device: str = "cpu",
) -> list[RepeatedSequence]:
	"""Generate simple A-B-A-B pattern sequences.

	Creates minimal induction patterns: [A B][A B][A B]...
	This is the simplest test case for induction heads.

	Parameters
	----------
	tokenizer
	    Tokenizer for getting valid tokens.
	n_sequences
	    Number of sequences to generate.
	pattern_length
	    Length of each A-B unit (typically 2).
	n_repetitions
	    Number of A-B repetitions.
	seed
	    Random seed.
	device
	    Device for tensors.

	Returns
	-------
	list[RepeatedSequence]
	    List of A-B-A-B sequences.
	"""
	return generate_repeated_sequences(
		tokenizer=tokenizer,
		n_sequences=n_sequences,
		seq_length=pattern_length,
		n_repetitions=n_repetitions,
		seed=seed,
		device=device,
	)


def sequences_to_batch(
	sequences: Sequence[RepeatedSequence],
	pad_to_length: int | None = None,
	pad_token_id: int = 0,
) -> Int[Tensor, "batch seq_len"]:
	"""Convert list of sequences to a batched tensor.

	Parameters
	----------
	sequences
	    List of RepeatedSequence objects.
	pad_to_length
	    If provided, pad all sequences to this length.
	    If None, pads to max sequence length in batch.
	pad_token_id
	    Token ID to use for padding.

	Returns
	-------
	Tensor
	    Batched token tensor of shape (batch, seq_len).
	"""
	if not sequences:
		raise ValueError("Empty sequence list")

	max_len = max(s.seq_len for s in sequences)
	target_len = pad_to_length if pad_to_length is not None else max_len

	batch: list[Tensor] = []
	for seq in sequences:
		tokens = seq.tokens
		if tokens.shape[0] < target_len:
			padding = torch.full(
				(target_len - tokens.shape[0],),
				pad_token_id,
				dtype=tokens.dtype,
				device=tokens.device,
			)
			tokens = torch.cat([tokens, padding])
		elif tokens.shape[0] > target_len:
			tokens = tokens[:target_len]
		batch.append(tokens)

	return torch.stack(batch)


def get_induction_mask(
	sequences: Sequence[RepeatedSequence],
	pad_to_length: int | None = None,
) -> Int[Tensor, "batch seq_len"]:
	"""Create a mask indicating positions where induction should help.

	Returns a boolean mask that is True at positions where the model
	could use induction to predict the next token (i.e., positions in
	the 2nd+ repetition of the pattern).

	Parameters
	----------
	sequences
	    List of RepeatedSequence objects.
	pad_to_length
	    If provided, create mask of this length.

	Returns
	-------
	Tensor
	    Boolean mask of shape (batch, seq_len).
	"""
	if not sequences:
		raise ValueError("Empty sequence list")

	max_len = max(s.seq_len for s in sequences)
	target_len = pad_to_length if pad_to_length is not None else max_len

	masks: list[Tensor] = []
	for seq in sequences:
		mask = torch.zeros(target_len, dtype=torch.bool, device=seq.tokens.device)
		for pos in seq.get_induction_positions():
			if pos < target_len:
				mask[pos] = True
		masks.append(mask)

	return torch.stack(masks)


def generate_long_context_prompts(
	tokenizer,
	source_texts: list[str],
	target_length: int = 512,
	device: str = "cpu",
) -> list[Int[Tensor, "seq_len"]]:
	"""Generate long prompts for ICL score evaluation.

	Takes source texts and tokenizes them to fixed length for
	measuring loss at early vs late positions.

	Parameters
	----------
	tokenizer
	    Tokenizer for encoding text.
	source_texts
	    List of text strings.
	target_length
	    Target sequence length (truncate or pad).
	device
	    Device for tensors.

	Returns
	-------
	list[Tensor]
	    List of tokenized sequences.
	"""
	sequences: list[Tensor] = []

	for text in source_texts:
		tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)

		if tokens.shape[0] >= target_length:
			tokens = tokens[:target_length]
		else:
			# Repeat text until we reach target length
			while tokens.shape[0] < target_length:
				more_tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
				tokens = torch.cat([tokens, more_tokens])
			tokens = tokens[:target_length]

		sequences.append(tokens.to(device))

	return sequences
