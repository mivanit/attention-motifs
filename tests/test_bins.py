"""Tests for util.bins — linear and logarithmic bin edge computation."""

from __future__ import annotations

import pytest
import torch

from attention_motifs.util.bins import Bins


class TestLinearBins:
	def test_edge_count(self) -> None:
		bins: Bins = Bins(n_bins=10, start=0.0, stop=1.0, scale="lin")
		assert bins.edges.shape[0] == 11  # n_bins + 1

	def test_edge_values(self) -> None:
		bins: Bins = Bins(n_bins=4, start=0.0, stop=1.0, scale="lin")
		expected: torch.Tensor = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
		torch.testing.assert_close(bins.edges, expected)

	def test_centers_count(self) -> None:
		bins: Bins = Bins(n_bins=10, start=0.0, stop=1.0, scale="lin")
		assert bins.centers.shape[0] == 10

	def test_centers_between_edges(self) -> None:
		bins: Bins = Bins(n_bins=8, start=0.0, stop=1.0, scale="lin")
		for i in range(bins.n_bins):
			assert bins.centers[i] > bins.edges[i]
			assert bins.centers[i] < bins.edges[i + 1]


class TestLogBins:
	def test_start_zero(self) -> None:
		"""start=0 → first edge is 0, rest log-spaced, total n_bins+1 edges."""
		bins: Bins = Bins(n_bins=8, start=0.0, stop=1.0, scale="log")
		assert bins.edges[0].item() == 0.0
		assert bins.edges.shape[0] == 9  # 1 (zero) + 8 (logspace)
		# all positive after first
		assert (bins.edges[1:] > 0).all()

	def test_start_positive(self) -> None:
		"""start > _log_min → pure logspace, n_bins+1 edges."""
		bins: Bins = Bins(n_bins=8, start=0.1, stop=1.0, scale="log")
		assert bins.edges.shape[0] == 9
		assert bins.edges[0].item() > 0

	def test_small_start_prepends_zero(self) -> None:
		"""start < _log_min with _zero_in_small_start_log → first edge is 0."""
		bins: Bins = Bins(
			n_bins=8,
			start=1e-6,
			stop=1.0,
			scale="log",
			_zero_in_small_start_log=True,
		)
		assert bins.edges[0].item() == 0.0
		assert bins.edges.shape[0] == 9  # 1 (zero) + 8

	def test_negative_start_raises(self) -> None:
		bins: Bins = Bins(n_bins=8, start=-1.0, stop=1.0, scale="log")
		with pytest.raises(ValueError, match="start must be positive"):
			_ = bins.edges


class TestInvalidScale:
	def test_invalid_scale_raises(self) -> None:
		bins: Bins = Bins(n_bins=8, start=0.0, stop=1.0, scale="xyz")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
		with pytest.raises(ValueError, match="Invalid scale"):
			_ = bins.edges


class TestChangedNBinsCopy:
	def test_preserves_other_params(self) -> None:
		original: Bins = Bins(n_bins=8, start=0.1, stop=2.0, scale="log")
		copy: Bins = original.changed_n_bins_copy(16)
		assert copy.n_bins == 16
		assert copy.start == original.start
		assert copy.stop == original.stop
		assert copy.scale == original.scale
