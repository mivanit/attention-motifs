"""Tests for the ablation study infrastructure."""

import pytest
import torch
import numpy as np

from attention_motifs.ablation.data import (
    RepeatedSequence,
    generate_repeated_sequences,
    sequences_to_batch,
    get_induction_mask,
    generate_abab_sequences,
)
from attention_motifs.ablation.ablate import (
    AblationMethod,
    parse_head_string,
    heads_from_strings,
)
from attention_motifs.ablation.candidates import (
    get_known_induction_heads,
    CandidateHeads,
    INDUCTION_TYPES,
)
from attention_motifs.ablation.metrics import AblationResult


# ============================================================
# Tests for data.py - Sequence Generation
# ============================================================


class TestRepeatedSequence:
    """Tests for RepeatedSequence dataclass."""

    def test_basic_creation(self):
        tokens = torch.tensor([1, 2, 3, 1, 2, 3])
        seq = RepeatedSequence(
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
        tokens = torch.tensor([1, 2, 3, 1, 2, 3])
        seq = RepeatedSequence(
            tokens=tokens,
            base_length=3,
            n_repetitions=2,
            repetition_starts=[0, 3],
        )
        positions = seq.get_induction_positions()
        # Position 3 is start of 2nd rep, so positions 4 and 5 are induction positions
        assert 4 in positions
        assert 5 in positions
        # Position 3 is the START of the 2nd repetition, not counted
        assert 3 not in positions
        # First repetition has no induction positions
        assert 0 not in positions
        assert 1 not in positions
        assert 2 not in positions


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
        sequences = generate_repeated_sequences(
            tokenizer=tokenizer,
            n_sequences=10,
            seq_length=5,
            n_repetitions=3,
            seed=42,
        )
        assert len(sequences) == 10
        for seq in sequences:
            assert seq.base_length == 5
            assert seq.n_repetitions == 3
            assert seq.seq_len == 15  # 5 * 3

    def test_repetition_structure(self):
        tokenizer = self._make_mock_tokenizer()
        sequences = generate_repeated_sequences(
            tokenizer=tokenizer,
            n_sequences=5,
            seq_length=4,
            n_repetitions=3,
            seed=42,
        )
        for seq in sequences:
            tokens = seq.tokens
            # Check that the sequence is actually repeated
            assert torch.all(tokens[0:4] == tokens[4:8])
            assert torch.all(tokens[0:4] == tokens[8:12])

    def test_seed_reproducibility(self):
        tokenizer = self._make_mock_tokenizer()
        seq1 = generate_repeated_sequences(tokenizer, n_sequences=5, seed=123)
        seq2 = generate_repeated_sequences(tokenizer, n_sequences=5, seed=123)
        for s1, s2 in zip(seq1, seq2):
            assert torch.all(s1.tokens == s2.tokens)


class TestSequencesBatch:
    """Tests for batching sequences."""

    def test_sequences_to_batch(self):
        sequences = [
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
        batch = sequences_to_batch(sequences)
        assert batch.shape == (2, 3)
        assert torch.all(batch[0] == torch.tensor([1, 2, 3]))
        assert torch.all(batch[1] == torch.tensor([4, 5, 6]))

    def test_batch_with_padding(self):
        sequences = [
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
        batch = sequences_to_batch(sequences, pad_token_id=0)
        assert batch.shape == (2, 3)
        assert batch[0, 2] == 0  # padded

    def test_induction_mask(self):
        sequences = [
            RepeatedSequence(
                tokens=torch.tensor([1, 2, 1, 2]),
                base_length=2,
                n_repetitions=2,
                repetition_starts=[0, 2],
            ),
        ]
        mask = get_induction_mask(sequences)
        assert mask.shape == (1, 4)
        # Position 3 should be an induction position (2nd rep, after start)
        assert mask[0, 3] == True
        # Position 2 is the start of 2nd rep, not an induction position
        assert mask[0, 2] == False


# ============================================================
# Tests for ablate.py - Head Parsing
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
        heads = heads_from_strings(["L5:H5", "L6:H9", "gpt2-small:L7:H10"])
        assert heads == [(5, 5), (6, 9), (7, 10)]

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_head_string("invalid")


class TestAblationMethod:
    """Tests for AblationMethod enum."""

    def test_enum_values(self):
        assert AblationMethod.ZERO.value == "zero"
        assert AblationMethod.MEAN.value == "mean"


# ============================================================
# Tests for candidates.py - Candidate Identification
# ============================================================


class TestKnownInductionHeads:
    """Tests for getting known induction heads."""

    def test_get_known_heads(self):
        heads = get_known_induction_heads()
        assert len(heads) > 0
        # All heads should be from gpt2-small
        for head in heads:
            assert head.startswith("gpt2-small:")

    def test_specific_heads_present(self):
        heads = get_known_induction_heads()
        # These are well-known induction heads from IOI paper
        assert "gpt2-small:L5:H5" in heads
        assert "gpt2-small:L6:H9" in heads


class TestCandidateHeads:
    """Tests for CandidateHeads dataclass."""

    def test_basic_creation(self):
        candidates = CandidateHeads(
            reference_heads=["gpt2-small:L5:H5"],
            candidates_by_model={
                "pythia-1b": [("pythia-1b:L5:H7", 0.8), ("pythia-1b:L6:H3", 0.5)],
            },
            k_neighbors=10,
        )
        assert len(candidates.reference_heads) == 1
        assert "pythia-1b" in candidates.candidates_by_model

    def test_get_top_candidates(self):
        candidates = CandidateHeads(
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
        top2 = candidates.get_top_candidates("pythia-1b", n=2)
        assert len(top2) == 2
        assert top2[0][0] == "pythia-1b:L5:H7"
        assert top2[1][0] == "pythia-1b:L6:H3"

    def test_get_all_candidates(self):
        candidates = CandidateHeads(
            reference_heads=["gpt2-small:L5:H5"],
            candidates_by_model={
                "pythia-1b": [("pythia-1b:L5:H7", 0.8)],
                "gemma-2b": [("gemma-2b:L3:H2", 0.6)],
            },
            k_neighbors=10,
        )
        all_cands = candidates.get_all_candidates()
        assert len(all_cands) == 2
        # Should be sorted by score descending
        assert all_cands[0][1] >= all_cands[1][1]

    def test_to_dataframe(self):
        candidates = CandidateHeads(
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


# ============================================================
# Tests for metrics.py - AblationResult
# ============================================================


class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_creation(self):
        result = AblationResult(
            head="pythia-1b:L5:H7",
            ablation_method=AblationMethod.ZERO,
            baseline_repeated_loss=3.5,
            ablated_repeated_loss=5.0,
            loss_increase=1.5,
            baseline_prefix_score=0.15,
            ablated_prefix_score=0.05,
            prefix_score_decrease=0.1,
            baseline_icl_score=-0.5,
            ablated_icl_score=-0.2,
            icl_degradation=0.3,
        )
        assert result.head == "pythia-1b:L5:H7"
        assert result.loss_increase == 1.5

    def test_to_dict(self):
        result = AblationResult(
            head="test:L0:H0",
            ablation_method=AblationMethod.MEAN,
            baseline_repeated_loss=3.0,
            ablated_repeated_loss=4.0,
            loss_increase=1.0,
            baseline_prefix_score=0.1,
            ablated_prefix_score=0.05,
            prefix_score_decrease=0.05,
            baseline_icl_score=-0.3,
            ablated_icl_score=-0.1,
            icl_degradation=0.2,
        )
        d = result.to_dict()
        assert d["head"] == "test:L0:H0"
        assert d["ablation_method"] == "mean"
        assert d["loss_increase"] == 1.0


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
            from transformer_lens import HookedTransformer

            # Use smallest available model
            return HookedTransformer.from_pretrained(
                "pythia-14m", device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except ImportError:
            pytest.skip("transformer-lens not installed")

    def test_generate_sequences_with_real_tokenizer(self, model):
        sequences = generate_repeated_sequences(
            tokenizer=model.tokenizer,
            n_sequences=5,
            seq_length=10,
            n_repetitions=2,
            seed=42,
        )
        assert len(sequences) == 5
        for seq in sequences:
            assert seq.seq_len == 20

    def test_ablator_creation(self, model):
        from attention_motifs.ablation.ablate import HeadAblator

        ablator = HeadAblator(model)
        assert ablator.n_layers == model.cfg.n_layers
        assert ablator.n_heads == model.cfg.n_heads
