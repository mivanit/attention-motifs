"""Tests for the ablation study infrastructure."""

import json
from pathlib import Path

import pytest
import torch

from attention_motifs.ablation.data import (
	RepeatedSequence,
	generate_repeated_sequences,
	sequences_to_batch,
	get_induction_mask,
)
from attention_motifs.ablation.ablate import (
	AblationMethod,
	parse_head_string,
)
from attention_motifs.attnpedia import heads_from_strings
from attention_motifs.ablation.candidates import (
	get_known_induction_heads,
	DistanceCandidates,
)
from attention_motifs.ablation.experiment import AblationConfig, AblationResults
from attention_motifs.ablation.metrics import AblationResult


# ============================================================
# Tests for data.py - Sequence Generation
# ============================================================


class TestRepeatedSequence:
	"""Tests for RepeatedSequence dataclass."""

	def test_basic_creation(self):
		tokens: torch.Tensor = torch.tensor([1, 2, 3, 1, 2, 3])
		seq: RepeatedSequence = RepeatedSequence(
			tokens=tokens,
			base_length=3,
			n_repetitions=2,
			repetition_starts=[0, 3],
		)
		assert seq.seq_len == 6
		assert seq.base_length == 3
		assert seq.n_repetitions == 2

	def test_induction_positions(self):
		# [A B C][A B C] - positions 4, 5 are in 2nd repetition (after start)
		tokens: torch.Tensor = torch.tensor([1, 2, 3, 1, 2, 3])
		seq: RepeatedSequence = RepeatedSequence(
			tokens=tokens,
			base_length=3,
			n_repetitions=2,
			repetition_starts=[0, 3],
		)
		positions: list[int] = seq.get_induction_positions()
		# Position 3 is start of 2nd rep, so positions 4 and 5 are induction positions
		assert 4 in positions
		assert 5 in positions
		# Position 3 is the START of the 2nd repetition, not counted
		assert 3 not in positions
		# First repetition has no induction positions
		assert 0 not in positions
		assert 1 not in positions
		assert 2 not in positions

	def test_has_bos_field(self):
		"""Test that has_bos field works correctly."""
		tokens: torch.Tensor = torch.tensor([0, 1, 2, 3, 1, 2, 3])
		seq: RepeatedSequence = RepeatedSequence(
			tokens=tokens,
			base_length=3,
			n_repetitions=2,
			repetition_starts=[1, 4],
			has_bos=True,
		)
		assert seq.has_bos is True
		assert seq.seq_len == 7

	def test_induction_positions_with_bos(self):
		"""Test induction positions are correct when BOS is prepended."""
		# [BOS][A B C][A B C]
		tokens: torch.Tensor = torch.tensor([0, 1, 2, 3, 1, 2, 3])
		seq: RepeatedSequence = RepeatedSequence(
			tokens=tokens,
			base_length=3,
			n_repetitions=2,
			repetition_starts=[1, 4],
			has_bos=True,
		)
		positions: list[int] = seq.get_induction_positions()
		# Position 4 is start of 2nd rep, so 5 and 6 are induction positions
		assert 5 in positions
		assert 6 in positions
		assert 4 not in positions
		# BOS and first repetition have no induction positions
		assert 0 not in positions
		assert 1 not in positions


class TestGenerateRepeatedSequences:
	"""Tests for sequence generation functions."""

	def _make_mock_tokenizer(self):
		"""Create a mock tokenizer with proper attributes."""

		class MockTokenizer:
			vocab_size = 5000
			bos_token_id = 0
			eos_token_id = 1
			pad_token_id = 2
			unk_token_id = 3
			all_special_ids = [0, 1, 2, 3]

		return MockTokenizer()

	def test_basic_generation(self):
		tokenizer = self._make_mock_tokenizer()
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=tokenizer,
			n_sequences=10,
			seq_length=5,
			n_repetitions=3,
			seed=42,
			prepend_bos=False,
		)
		assert len(sequences) == 10
		for seq in sequences:
			assert seq.base_length == 5
			assert seq.n_repetitions == 3
			assert seq.seq_len == 15  # 5 * 3

	def test_repetition_structure(self):
		tokenizer = self._make_mock_tokenizer()
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=tokenizer,
			n_sequences=5,
			seq_length=4,
			n_repetitions=3,
			seed=42,
			prepend_bos=False,
		)
		for seq in sequences:
			tokens: torch.Tensor = seq.tokens
			# Check that the sequence is actually repeated
			assert torch.all(tokens[0:4] == tokens[4:8])
			assert torch.all(tokens[0:4] == tokens[8:12])

	def test_seed_reproducibility(self):
		tokenizer = self._make_mock_tokenizer()
		seq1: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer, n_sequences=5, seed=123
		)
		seq2: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer, n_sequences=5, seed=123
		)
		for s1, s2 in zip(seq1, seq2):
			assert torch.all(s1.tokens == s2.tokens)

	def test_prepend_bos(self):
		"""Test that BOS token is correctly prepended."""
		tokenizer = self._make_mock_tokenizer()
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=tokenizer,
			n_sequences=5,
			seq_length=4,
			n_repetitions=2,
			seed=42,
			prepend_bos=True,
		)
		for seq in sequences:
			assert seq.has_bos is True
			# BOS (1) + 4*2 tokens = 9
			assert seq.seq_len == 9
			# First token should be BOS
			assert seq.tokens[0].item() == 0  # bos_token_id
			# repetition_starts should be shifted by 1
			assert seq.repetition_starts[0] == 1
			assert seq.repetition_starts[1] == 5
			# Repetitions should still match
			assert torch.all(seq.tokens[1:5] == seq.tokens[5:9])

	def test_no_bos(self):
		"""Test generation without BOS prepend."""
		tokenizer = self._make_mock_tokenizer()
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=tokenizer,
			n_sequences=5,
			seq_length=4,
			n_repetitions=2,
			seed=42,
			prepend_bos=False,
		)
		for seq in sequences:
			assert seq.has_bos is False
			assert seq.seq_len == 8  # 4 * 2
			assert seq.repetition_starts[0] == 0
			assert seq.repetition_starts[1] == 4

	def test_exclude_common_tokens(self):
		"""Test that common tokens (low IDs) are excluded."""
		tokenizer = self._make_mock_tokenizer()
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=tokenizer,
			n_sequences=10,
			seq_length=10,
			n_repetitions=2,
			seed=42,
			exclude_common=True,
			common_threshold=100,
			prepend_bos=False,
		)
		for seq in sequences:
			# No token should be below common_threshold (except BOS if prepended)
			for token_id in seq.tokens.tolist():
				assert token_id >= 100, (
					f"Token {token_id} is below common_threshold=100"
				)

	def test_exclude_common_disabled(self):
		"""Test that common tokens can be included."""
		tokenizer = self._make_mock_tokenizer()
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=tokenizer,
			n_sequences=50,
			seq_length=20,
			n_repetitions=2,
			seed=42,
			exclude_common=False,
			prepend_bos=False,
		)
		# With 50 sequences of length 20 and common tokens allowed,
		# at least some tokens should be below 100 (probabilistic, but very likely)
		all_tokens: list[int] = []
		for seq in sequences:
			all_tokens.extend(seq.tokens.tolist())
		has_low_token: bool = any(t < 100 for t in all_tokens)
		# Special tokens (0-3) are still excluded, but 4-99 should appear
		assert has_low_token, "Expected some tokens below 100 when exclude_common=False"

	def test_custom_bos_token_id(self):
		"""Test custom BOS token ID."""
		tokenizer = self._make_mock_tokenizer()
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=tokenizer,
			n_sequences=3,
			seq_length=4,
			n_repetitions=2,
			seed=42,
			prepend_bos=True,
			bos_token_id=50256,
		)
		for seq in sequences:
			assert seq.tokens[0].item() == 50256


class TestSequencesBatch:
	"""Tests for batching sequences."""

	def test_sequences_to_batch(self):
		sequences: list[RepeatedSequence] = [
			RepeatedSequence(
				tokens=torch.tensor([1, 2, 3]),
				base_length=3,
				n_repetitions=1,
				repetition_starts=[0],
			),
			RepeatedSequence(
				tokens=torch.tensor([4, 5, 6]),
				base_length=3,
				n_repetitions=1,
				repetition_starts=[0],
			),
		]
		batch: torch.Tensor = sequences_to_batch(sequences)
		assert batch.shape == (2, 3)
		assert torch.all(batch[0] == torch.tensor([1, 2, 3]))
		assert torch.all(batch[1] == torch.tensor([4, 5, 6]))

	def test_batch_with_padding(self):
		sequences: list[RepeatedSequence] = [
			RepeatedSequence(
				tokens=torch.tensor([1, 2]),
				base_length=2,
				n_repetitions=1,
				repetition_starts=[0],
			),
			RepeatedSequence(
				tokens=torch.tensor([3, 4, 5]),
				base_length=3,
				n_repetitions=1,
				repetition_starts=[0],
			),
		]
		batch: torch.Tensor = sequences_to_batch(sequences, pad_token_id=0)
		assert batch.shape == (2, 3)
		assert batch[0, 2] == 0  # padded

	def test_induction_mask(self):
		sequences: list[RepeatedSequence] = [
			RepeatedSequence(
				tokens=torch.tensor([1, 2, 1, 2]),
				base_length=2,
				n_repetitions=2,
				repetition_starts=[0, 2],
			),
		]
		mask: torch.Tensor = get_induction_mask(sequences)
		assert mask.shape == (1, 4)
		# Position 3 should be an induction position (2nd rep, after start)
		assert mask[0, 3]
		# Position 2 is the start of 2nd rep, not an induction position
		assert not mask[0, 2]


# ============================================================
# Tests for ablate.py - Head Parsing and AblationMethod
# ============================================================


class TestHeadParsing:
	"""Tests for head string parsing utilities."""

	def test_parse_head_string_short_format(self):
		layer, head = parse_head_string("L5:H5")
		assert layer == 5
		assert head == 5

	def test_parse_head_string_full_format(self):
		layer, head = parse_head_string("gpt2-small:L6:H9")
		assert layer == 6
		assert head == 9

	def test_parse_head_string_different_numbers(self):
		layer, head = parse_head_string("L0:H11")
		assert layer == 0
		assert head == 11

	def test_heads_from_strings(self):
		heads: list[tuple[int, int]] = heads_from_strings(
			["L5:H5", "L6:H9", "gpt2-small:L7:H10"]
		)
		assert heads == [(5, 5), (6, 9), (7, 10)]

	def test_invalid_format(self):
		with pytest.raises(ValueError):
			parse_head_string("invalid")


class TestAblationMethod:
	"""Tests for AblationMethod enum."""

	def test_enum_values(self):
		assert AblationMethod.ZERO.value == "zero"
		assert AblationMethod.MEAN.value == "mean"

	def test_pattern_preserving_enum(self):
		"""Test that PATTERN_PRESERVING enum value exists."""
		assert AblationMethod.PATTERN_PRESERVING.value == "pattern_preserving"

	def test_all_methods(self):
		"""Test that all expected ablation methods are defined."""
		methods: set[str] = {m.value for m in AblationMethod}
		assert methods == {"zero", "mean", "pattern_preserving"}


# ============================================================
# Tests for candidates.py - Candidate Identification
# ============================================================


class TestKnownInductionHeads:
	"""Tests for getting known induction heads."""

	def test_get_known_heads(self):
		heads: list[str] = get_known_induction_heads()
		assert len(heads) > 0
		# All heads should be from gpt2-small
		for head in heads:
			assert head.startswith("gpt2-small:")

	def test_specific_heads_present(self):
		heads: list[str] = get_known_induction_heads()
		# These are well-known induction heads from IOI paper
		assert "gpt2-small:L5:H5" in heads
		assert "gpt2-small:L6:H9" in heads


class TestDistanceCandidates:
	"""Tests for DistanceCandidates dataclass."""

	def test_basic_creation(self):
		candidates: DistanceCandidates = DistanceCandidates(
			reference_heads=["gpt2-small:L5:H5"],
			candidates_by_model={
				"pythia-1b": [("pythia-1b:L5:H7", 0.8), ("pythia-1b:L6:H3", 0.5)],
			},
			k_neighbors=10,
		)
		assert len(candidates.reference_heads) == 1
		assert "pythia-1b" in candidates.candidates_by_model

	def test_get_top_candidates(self):
		candidates: DistanceCandidates = DistanceCandidates(
			reference_heads=["gpt2-small:L5:H5"],
			candidates_by_model={
				"pythia-1b": [
					("pythia-1b:L5:H7", 0.8),
					("pythia-1b:L6:H3", 0.5),
					("pythia-1b:L4:H1", 0.3),
				],
			},
			k_neighbors=10,
		)
		top2: list[tuple[str, float]] = candidates.get_top_candidates("pythia-1b", n=2)
		assert len(top2) == 2
		assert top2[0][0] == "pythia-1b:L5:H7"
		assert top2[1][0] == "pythia-1b:L6:H3"

	def test_get_all_candidates(self):
		candidates: DistanceCandidates = DistanceCandidates(
			reference_heads=["gpt2-small:L5:H5"],
			candidates_by_model={
				"pythia-1b": [("pythia-1b:L5:H7", 0.8)],
				"gemma-2b": [("gemma-2b:L3:H2", 0.6)],
			},
			k_neighbors=10,
		)
		all_cands: list[tuple[str, float]] = candidates.get_all_candidates()
		assert len(all_cands) == 2
		# Should be sorted by score descending
		assert all_cands[0][1] >= all_cands[1][1]

	def test_to_dataframe(self):
		candidates: DistanceCandidates = DistanceCandidates(
			reference_heads=["gpt2-small:L5:H5"],
			candidates_by_model={
				"pythia-1b": [("pythia-1b:L5:H7", 0.8)],
			},
			k_neighbors=10,
		)
		df = candidates.to_dataframe()
		assert "head" in df.columns
		assert "model" in df.columns
		assert "layer" in df.columns
		assert "score" in df.columns
		assert len(df) == 1


class TestDistanceCandidatesRoundtrip:
	"""Round-trip serialization tests for DistanceCandidates."""

	def _make_candidates(self) -> DistanceCandidates:
		return DistanceCandidates(
			reference_heads=["gpt2-small:L5:H5", "gpt2-small:L6:H9"],
			candidates_by_model={
				"pythia-1b": [("pythia-1b:L5:H7", 0.8), ("pythia-1b:L6:H3", 0.5)],
				"gemma-2b": [("gemma-2b:L3:H2", 0.6)],
			},
			all_neighbors={
				"gpt2-small:L5:H5": [("pythia-1b:L5:H7", 0.8)],
			},
			k_neighbors=20,
			model_n_layers={"pythia-1b": 16, "gemma-2b": 18},
		)

	def test_save_read_roundtrip(self, tmp_path: Path) -> None:
		"""save → read preserves all fields.

		Note: JSON round-trips tuples as lists, so we compare with
		list-converted originals.
		"""
		original: DistanceCandidates = self._make_candidates()
		path: Path = tmp_path / "candidates.json"
		original.save(path)

		assert path.exists()
		restored: DistanceCandidates = DistanceCandidates.read(path)

		assert restored.reference_heads == original.reference_heads
		assert restored.k_neighbors == original.k_neighbors
		assert restored.model_n_layers == original.model_n_layers

		# JSON converts tuples to lists, so compare element-by-element
		for model in original.candidates_by_model:
			assert model in restored.candidates_by_model
			for (rh, rs), (oh, os) in zip(
				restored.candidates_by_model[model],
				original.candidates_by_model[model],
			):
				assert rh == oh
				assert rs == os

		for ref_head in original.all_neighbors:
			assert ref_head in restored.all_neighbors
			for (rh, rs), (oh, os) in zip(
				restored.all_neighbors[ref_head],
				original.all_neighbors[ref_head],
			):
				assert rh == oh
				assert rs == os


# ============================================================
# Tests for metrics.py - AblationResult
# ============================================================


class TestAblationResult:
	"""Tests for AblationResult dataclass."""

	def test_creation(self):
		result: AblationResult = AblationResult(
			head="pythia-1b:L5:H7",
			ablation_method=AblationMethod.ZERO,
			baseline_repeated_loss=3.5,
			ablated_repeated_loss=5.0,
			loss_increase=1.5,
			prefix_score=0.15,
			baseline_icl_score=-0.5,
			ablated_icl_score=-0.2,
			icl_degradation=0.3,
		)
		assert result.head == "pythia-1b:L5:H7"
		assert result.loss_increase == 1.5

	def test_optional_fields_defaults(self):
		"""Test that optional fields have correct defaults."""
		result: AblationResult = AblationResult(
			head="test:L0:H0",
			ablation_method=AblationMethod.ZERO,
			baseline_repeated_loss=3.0,
			ablated_repeated_loss=4.0,
			loss_increase=1.0,
			prefix_score=0.1,
		)
		# Characterization fields default to 0.0
		assert result.prefix_score_legacy == 0.0
		assert result.copying_score == 0.0
		assert result.ov_copying_score == 0.0
		# ICL fields default to None (not measured)
		assert result.baseline_icl_score is None
		assert result.ablated_icl_score is None
		assert result.icl_degradation is None

	def test_to_dict(self):
		result: AblationResult = AblationResult(
			head="test:L0:H0",
			ablation_method=AblationMethod.MEAN,
			baseline_repeated_loss=3.0,
			ablated_repeated_loss=4.0,
			loss_increase=1.0,
			prefix_score=0.1,
			baseline_icl_score=-0.3,
			ablated_icl_score=-0.1,
			icl_degradation=0.2,
		)
		d: dict = result.serialize()
		assert d["head"] == "test:L0:H0"
		assert d["ablation_method"] == "mean"
		assert d["loss_increase"] == 1.0

	def test_serialize_includes_all_fields(self):
		"""Test that serialize() includes all metric fields."""
		result: AblationResult = AblationResult(
			head="test:L0:H0",
			ablation_method=AblationMethod.ZERO,
			baseline_repeated_loss=3.0,
			ablated_repeated_loss=4.0,
			loss_increase=1.0,
			prefix_score=0.1,
			prefix_score_legacy=0.05,
			copying_score=0.7,
			ov_copying_score=0.6,
		)
		d: dict = result.serialize()
		assert d["prefix_score"] == 0.1
		assert d["prefix_score_legacy"] == 0.05
		assert d["copying_score"] == 0.7
		assert d["ov_copying_score"] == 0.6
		assert d["baseline_icl_score"] is None
		assert d["ablated_icl_score"] is None
		assert d["icl_degradation"] is None

	def test_pattern_preserving_ablation_method(self):
		"""Test AblationResult with pattern-preserving method."""
		result: AblationResult = AblationResult(
			head="test:L0:H0",
			ablation_method=AblationMethod.PATTERN_PRESERVING,
			baseline_repeated_loss=3.0,
			ablated_repeated_loss=4.0,
			loss_increase=1.0,
			prefix_score=0.1,
		)
		d: dict = result.serialize()
		assert d["ablation_method"] == "pattern_preserving"


# ============================================================
# Round-trip serialization tests
# ============================================================


def _make_ablation_result(*, with_icl: bool = True) -> AblationResult:
	"""Create a fully-populated AblationResult for testing."""
	kwargs: dict = dict(
		head="test-model:L3:H7",
		ablation_method=AblationMethod.ZERO,
		baseline_repeated_loss=3.5,
		ablated_repeated_loss=5.0,
		loss_increase=1.5,
		prefix_score=0.15,
		prefix_score_legacy=0.12,
		copying_score=0.7,
		ov_copying_score=0.6,
	)
	if with_icl:
		kwargs.update(
			baseline_icl_score=-0.5,
			ablated_icl_score=-0.2,
			icl_degradation=0.3,
		)
	return AblationResult(**kwargs)


def _make_ablation_config() -> AblationConfig:
	"""Create an AblationConfig with non-default values."""
	return AblationConfig(
		n_sequences=50,
		seq_length=30,
		n_repetitions=3,
		ablation_methods=[AblationMethod.ZERO, AblationMethod.PATTERN_PRESERVING],
		n_calibration_prompts=100,
		calibration_prompts_file="data/text/test.jsonl",
		seed=123,
		micro_batch_size=10,
		icl_prompts_file="data/text/icl.jsonl",
		n_icl_prompts=25,
	)


def _make_ablation_results() -> AblationResults:
	"""Create an AblationResults with nested objects for testing."""
	return AblationResults(
		model_name="test-model",
		config=_make_ablation_config(),
		results=[
			_make_ablation_result(with_icl=True),
			AblationResult(
				head="test-model:L1:H0",
				ablation_method=AblationMethod.PATTERN_PRESERVING,
				baseline_repeated_loss=3.0,
				ablated_repeated_loss=3.8,
				loss_increase=0.8,
				prefix_score=0.05,
				prefix_score_legacy=0.0,
				copying_score=0.0,
				ov_copying_score=0.0,
			),
		],
		baseline_loss=3.5,
		baseline_icl=-0.3,
	)


class TestAblationResultRoundtrip:
	"""Round-trip serialization tests for AblationResult."""

	def test_serialize_load_roundtrip(self) -> None:
		"""serialize → load preserves all fields."""
		original: AblationResult = _make_ablation_result(with_icl=True)
		data: dict = original.serialize()
		restored: AblationResult = AblationResult.load(data)

		assert restored.head == original.head
		assert restored.ablation_method == original.ablation_method
		assert restored.baseline_repeated_loss == original.baseline_repeated_loss
		assert restored.ablated_repeated_loss == original.ablated_repeated_loss
		assert restored.loss_increase == original.loss_increase
		assert restored.prefix_score == original.prefix_score
		assert restored.prefix_score_legacy == original.prefix_score_legacy
		assert restored.copying_score == original.copying_score
		assert restored.ov_copying_score == original.ov_copying_score
		assert restored.baseline_icl_score == original.baseline_icl_score
		assert restored.ablated_icl_score == original.ablated_icl_score
		assert restored.icl_degradation == original.icl_degradation

	def test_serialize_load_roundtrip_defaults(self) -> None:
		"""Round-trip preserves default values (None, 0.0)."""
		original: AblationResult = _make_ablation_result(with_icl=False)
		data: dict = original.serialize()
		restored: AblationResult = AblationResult.load(data)

		assert restored.baseline_icl_score is None
		assert restored.ablated_icl_score is None
		assert restored.icl_degradation is None
		assert restored.prefix_score_legacy == original.prefix_score_legacy
		assert restored.copying_score == original.copying_score


class TestAblationConfigRoundtrip:
	"""Round-trip serialization tests for AblationConfig."""

	def test_serialize_load_roundtrip(self) -> None:
		"""serialize → load preserves all fields, including enum list."""
		original: AblationConfig = _make_ablation_config()
		data: dict = original.serialize()
		restored: AblationConfig = AblationConfig.load(data)

		assert restored.n_sequences == original.n_sequences
		assert restored.seq_length == original.seq_length
		assert restored.n_repetitions == original.n_repetitions
		assert restored.ablation_methods == original.ablation_methods
		assert restored.n_calibration_prompts == original.n_calibration_prompts
		assert restored.calibration_prompts_file == original.calibration_prompts_file
		assert restored.seed == original.seed
		assert restored.micro_batch_size == original.micro_batch_size
		assert restored.icl_prompts_file == original.icl_prompts_file
		assert restored.n_icl_prompts == original.n_icl_prompts

	def test_serialize_load_roundtrip_defaults(self) -> None:
		"""Default config round-trips correctly."""
		original: AblationConfig = AblationConfig()
		data: dict = original.serialize()
		restored: AblationConfig = AblationConfig.load(data)

		assert restored.ablation_methods == original.ablation_methods
		assert restored.icl_prompts_file is None
		assert restored.seed == 42


class TestAblationResultsRoundtrip:
	"""Round-trip serialization tests for AblationResults."""

	def test_serialize_load_roundtrip(self) -> None:
		"""serialize → load preserves all fields including nested objects."""
		original: AblationResults = _make_ablation_results()
		data: dict = original.serialize()
		restored: AblationResults = AblationResults.load(data)

		assert restored.model_name == original.model_name
		assert restored.baseline_loss == original.baseline_loss
		assert restored.baseline_icl == original.baseline_icl

		# Nested config
		assert restored.config.n_sequences == original.config.n_sequences
		assert restored.config.ablation_methods == original.config.ablation_methods

		# Nested results list
		assert len(restored.results) == len(original.results)
		assert restored.results[0].head == original.results[0].head
		assert restored.results[0].ablation_method == original.results[0].ablation_method
		assert restored.results[0].baseline_icl_score == original.results[0].baseline_icl_score
		assert restored.results[1].head == original.results[1].head
		assert restored.results[1].baseline_icl_score is None

	def test_save_read_roundtrip(self, tmp_path: Path) -> None:
		"""save → read preserves all fields."""
		original: AblationResults = _make_ablation_results()
		path: Path = tmp_path / "ablation_results.json"
		original.save(path)

		assert path.exists()
		restored: AblationResults = AblationResults.read(path)

		assert restored.model_name == original.model_name
		assert restored.baseline_loss == original.baseline_loss
		assert restored.baseline_icl == original.baseline_icl
		assert len(restored.results) == len(original.results)
		assert restored.results[0].head == original.results[0].head
		assert restored.results[0].loss_increase == original.results[0].loss_increase
		assert restored.config.seed == original.config.seed

	def test_read_legacy_field_names(self, tmp_path: Path) -> None:
		"""read() remaps old field names (baseline_prefix_score → prefix_score)."""
		# Build raw JSON with legacy field names
		legacy_data: dict = {
			"model_name": "legacy-model",
			"config": AblationConfig().serialize(),
			"results": [
				{
					"head": "legacy-model:L0:H0",
					"ablation_method": "zero",
					"baseline_repeated_loss": 3.0,
					"ablated_repeated_loss": 4.0,
					"loss_increase": 1.0,
					"baseline_prefix_score": 0.15,
					"baseline_prefix_score_legacy": 0.12,
					"copying_score": 0.0,
					"ov_copying_score": 0.0,
					"baseline_icl_score": None,
					"ablated_icl_score": None,
					"icl_degradation": None,
				}
			],
			"baseline_loss": 3.0,
			"baseline_icl": None,
		}
		path: Path = tmp_path / "legacy_results.json"
		path.write_text(json.dumps(legacy_data))

		restored: AblationResults = AblationResults.read(path)
		assert restored.results[0].prefix_score == 0.15
		assert restored.results[0].prefix_score_legacy == 0.12

	def test_serialize_load_with_none_icl(self) -> None:
		"""Round-trip preserves None baseline_icl and None ICL result fields."""
		original: AblationResults = AblationResults(
			model_name="no-icl-model",
			config=AblationConfig(),
			results=[_make_ablation_result(with_icl=False)],
			baseline_loss=2.0,
			baseline_icl=None,
		)
		data: dict = original.serialize()
		restored: AblationResults = AblationResults.load(data)

		assert restored.baseline_icl is None
		assert restored.results[0].baseline_icl_score is None
		assert restored.results[0].ablated_icl_score is None
		assert restored.results[0].icl_degradation is None


# ============================================================
# Integration tests (require transformer-lens)
# ============================================================


@pytest.mark.skipif(
	not torch.cuda.is_available(),
	reason="GPU not available for integration tests",
)
class TestIntegration:
	"""Integration tests that require transformer-lens and GPU."""

	@pytest.fixture(scope="class")
	def model(self):
		"""Load a small model for testing."""
		try:
			from pattern_lens.load_model import load_model

			# Use smallest available model
			return load_model(
				"pythia-14m", device="cuda" if torch.cuda.is_available() else "cpu"
			)
		except ImportError:
			pytest.skip("transformer-lens not installed")

	def test_generate_sequences_with_real_tokenizer(self, model):
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=model.tokenizer,
			n_sequences=5,
			seq_length=10,
			n_repetitions=2,
			seed=42,
			prepend_bos=False,
		)
		assert len(sequences) == 5
		for seq in sequences:
			assert seq.seq_len == 20

	def test_generate_sequences_with_bos(self, model):
		"""Test BOS prepending with a real tokenizer."""
		sequences: list[RepeatedSequence] = generate_repeated_sequences(
			tokenizer=model.tokenizer,
			n_sequences=5,
			seq_length=10,
			n_repetitions=2,
			seed=42,
			prepend_bos=True,
		)
		assert len(sequences) == 5
		for seq in sequences:
			assert seq.has_bos is True
			# BOS (1) + 10*2 = 21
			assert seq.seq_len == 21
			assert seq.repetition_starts[0] == 1

	def test_ablator_creation(self, model):
		from attention_motifs.ablation.ablate import HeadAblator

		ablator: HeadAblator = HeadAblator(model)
		assert ablator.n_layers == model.cfg.n_layers
		assert ablator.n_heads == model.cfg.n_heads

	def test_cache_clean_patterns(self, model):
		"""Test clean pattern caching for pattern-preserving ablation."""
		from attention_motifs.ablation.ablate import HeadAblator

		ablator: HeadAblator = HeadAblator(model)
		tokens: torch.Tensor = model.to_tokens("Hello world test")
		patterns: dict = ablator.cache_clean_patterns(tokens)

		# Should have one pattern per layer
		assert len(patterns) == model.cfg.n_layers
		# Each pattern should have correct shape
		for name, pattern in patterns.items():
			assert pattern.dim() == 4  # batch, n_heads, dest, src

	def test_pattern_preserving_ablation_context(self, model):
		"""Test pattern-preserving ablation context manager."""
		from attention_motifs.ablation.ablate import HeadAblator

		ablator: HeadAblator = HeadAblator(model)
		tokens: torch.Tensor = model.to_tokens("Hello world test")

		# Should raise if clean patterns not cached
		with pytest.raises(ValueError, match="set_clean_patterns"):
			with ablator.ablate_heads(
				[(0, 0)], method=AblationMethod.PATTERN_PRESERVING
			):
				pass

		# Should work after caching
		ablator.set_clean_patterns(tokens)
		with (
			ablator.ablate_heads([(0, 0)], method=AblationMethod.PATTERN_PRESERVING),
			torch.no_grad(),
		):
			output: torch.Tensor = model(tokens, prepend_bos=False)
			assert output is not None


# ============================================================
# Tests for frontend.py - Ablation Frontend
# ============================================================


class TestAblationFrontend:
	"""Tests for ablation frontend HTML generation."""

	def _make_results(self) -> dict[str, AblationResults]:
		"""Create mock AblationResults for testing."""
		results: AblationResults = AblationResults(
			model_name="test-model",
			config=AblationConfig(
				n_sequences=10,
				seq_length=5,
				n_repetitions=2,
			),
			baseline_loss=3.5,
			baseline_icl=-0.3,
			results=[
				AblationResult(
					head="test-model:L0:H0",
					ablation_method=AblationMethod.ZERO,
					baseline_repeated_loss=3.5,
					ablated_repeated_loss=4.2,
					loss_increase=0.7,
					prefix_score=0.15,
				),
			],
		)
		return {"test-model": results}

	def test_write_ablation_frontend(self, tmp_path: Path) -> None:
		"""Test that write_ablation_frontend writes HTML and data JSON."""
		from attention_motifs.ablation.frontend import write_ablation_frontend

		all_results: dict[str, AblationResults] = self._make_results()
		output_path: Path = write_ablation_frontend(all_results, tmp_path)

		assert output_path.exists()
		assert output_path.name == "index.html"

		data_path: Path = tmp_path / "ablation_results.json"
		assert data_path.exists()

		import json

		data: dict = json.loads(data_path.read_text())
		assert "test-model" in data["models"]
		assert data["models"]["test-model"]["results"][0]["head"] == "test-model:L0:H0"

	def test_serialize_results_structure(self) -> None:
		"""Test that _serialize_results produces correct structure."""
		from attention_motifs.ablation.frontend import _serialize_results

		all_results: dict[str, AblationResults] = self._make_results()
		data: dict = _serialize_results(all_results)

		assert "models" in data
		assert "test-model" in data["models"]
		model_data: dict = data["models"]["test-model"]
		assert model_data["baseline_loss"] == 3.5
		assert model_data["config"]["n_sequences"] == 10
		assert len(model_data["results"]) == 1
		assert model_data["results"][0]["head"] == "test-model:L0:H0"
