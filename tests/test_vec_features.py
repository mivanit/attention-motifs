"""Tests for features.vec_features — statistical feature extraction from 1D arrays."""

from __future__ import annotations

import numpy as np
import pytest

from attention_motifs.features.vec_features import (
	VEC_FEATURES_NAMES,
	vec_features,
	vec_features_arr,
	vec_features_fast,
)


class TestVecFeatures:
	"""Tests for the pure-Python vec_features function."""

	def test_known_values(self) -> None:
		"""Simple known array → verify basic stats."""
		arr: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		feats: dict[str, float] = vec_features(arr)

		assert feats["mean"] == pytest.approx(3.0)
		assert feats["median"] == pytest.approx(3.0)
		assert feats["variance"] == pytest.approx(2.5)  # ddof=1
		assert feats["rms"] == pytest.approx(np.sqrt(np.mean(arr**2)))
		# perfect linear trend → r2 ≈ 1
		assert feats["linreg.r2"] == pytest.approx(1.0, abs=1e-6)
		assert feats["linreg.slope"] == pytest.approx(1.0, abs=1e-6)

	def test_constant_array(self) -> None:
		"""All-same values → variance=0, autocorr=0."""
		arr: np.ndarray = np.full(10, 5.0)
		feats: dict[str, float] = vec_features(arr)

		assert feats["mean"] == pytest.approx(5.0)
		assert feats["variance"] == pytest.approx(0.0, abs=1e-10)
		# scipy.stats.skew returns NaN for constant arrays (zero std)
		assert np.isnan(feats["skewness"]) or feats["skewness"] == pytest.approx(
			0.0, abs=1e-10
		)
		# autocorr should be 0 since std is 0
		assert feats["autocorr_lag1"] == pytest.approx(0.0, abs=1e-10)

	def test_nan_handling(self) -> None:
		"""Array with NaN → replaced with 0, no crash."""
		arr: np.ndarray = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
		feats: dict[str, float] = vec_features(arr)
		# should not crash, NaN replaced with 0
		assert not np.isnan(feats["mean"])

	def test_negative_values(self) -> None:
		"""Negative array → correct skew and mean."""
		arr: np.ndarray = np.array([-5.0, -3.0, -1.0, 0.0, 10.0])
		feats: dict[str, float] = vec_features(arr)
		assert feats["mean"] == pytest.approx(0.2)
		# right-skewed (long tail toward positive)
		assert feats["skewness"] > 0

	def test_input_validation(self) -> None:
		"""2D array → ValueError."""
		arr: np.ndarray = np.array([[1, 2], [3, 4]], dtype=float)
		with pytest.raises(ValueError, match="1-dimensional"):
			vec_features(arr)

	def test_reduced_vs_full(self) -> None:
		"""reduced=False includes extra features."""
		arr: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		reduced: dict[str, float] = vec_features(arr, reduced=True)
		full: dict[str, float] = vec_features(arr, reduced=False)
		assert "energy" not in reduced
		assert "energy" in full
		assert "kurtosis" not in reduced
		assert "kurtosis" in full


class TestVecFeaturesArr:
	"""Tests for the Numba-compiled vec_features_arr function."""

	def test_output_length(self) -> None:
		arr: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		result: np.ndarray = vec_features_arr(arr)
		assert result.shape == (len(VEC_FEATURES_NAMES),)

	def test_known_values(self) -> None:
		arr: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		result: np.ndarray = vec_features_arr(arr)
		feats: dict[str, float] = dict(zip(VEC_FEATURES_NAMES, result))

		assert feats["mean"] == pytest.approx(3.0)
		assert feats["median"] == pytest.approx(3.0)
		assert feats["variance"] == pytest.approx(2.5, rel=1e-4)
		assert feats["linreg.r2"] == pytest.approx(1.0, abs=1e-4)

	def test_input_validation(self) -> None:
		arr: np.ndarray = np.array([[1, 2], [3, 4]], dtype=float)
		with pytest.raises(ValueError, match="1-D"):
			vec_features_arr(arr)

	def test_single_element(self) -> None:
		"""Size-1 array → no crash, variance=0."""
		arr: np.ndarray = np.array([42.0])
		result: np.ndarray = vec_features_arr(arr)
		feats: dict[str, float] = dict(zip(VEC_FEATURES_NAMES, result))
		assert feats["mean"] == pytest.approx(42.0)
		assert feats["variance"] == pytest.approx(0.0, abs=1e-10)


class TestVecFeaturesFast:
	def test_keys_match_names(self) -> None:
		arr: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		feats: dict[str, float] = vec_features_fast(arr)
		assert set(feats.keys()) == set(VEC_FEATURES_NAMES)

	def test_values_match_arr(self) -> None:
		arr: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
		feats_dict: dict[str, float] = vec_features_fast(arr)
		feats_arr: np.ndarray = vec_features_arr(arr)
		for i, name in enumerate(VEC_FEATURES_NAMES):
			assert feats_dict[name] == pytest.approx(feats_arr[i], abs=1e-10)
