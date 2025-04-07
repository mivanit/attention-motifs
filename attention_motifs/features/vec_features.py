import numpy as np
import scipy.stats as stats
import scipy.signal as signal

# muutils
from muutils.dbg import dbg_tensor

# attention-motifs


def vec_features(
	arr: np.ndarray,
	compute_distribution: bool = True,
	compute_timeseries: bool = True,
) -> dict[str, float]:
	if len(arr.shape) != 1:
		dbg_tensor(arr)
		raise ValueError(f"Input arr must be 1-dimensional, got {arr.shape}")

	n: int = arr.size

	dist_features: dict[str, float] = dict()
	timeseries_features: dict[str, float] = dict()

	if compute_distribution:
		# dbg_tensor(arr)
		bins: int = 10 if n >= 10 else n
		hist, _ = np.histogram(arr, bins=bins)
		probs: np.ndarray = (
			hist.astype(float) / hist.sum() if hist.sum() > 0 else hist.astype(float)
		)
		dist_features = dict(
			mean=np.mean(arr),
			median=np.median(arr),
			variance=np.var(arr, ddof=1),
			std=np.std(arr, ddof=1),
			skewness=stats.skew(arr),
			kurtosis=stats.kurtosis(arr),
			entropy=stats.entropy(probs, base=2),
			L1_norm=np.sum(np.abs(arr)) / n,
			L2_norm=np.linalg.norm(arr, ord=2) / n,
			rms=np.sqrt(np.mean(arr**2)),
			energy=np.sum(arr**2),
		)

	if compute_timeseries:
		# Lag-1 Autocorrelation (Pearson correlation between arr[:-1] and arr[1:])
		autocorr_lag1: float = (
			np.corrcoef(arr[:-1], arr[1:])[0, 1]
			if np.std(arr[:-1]) > 0 and np.std(arr[1:]) > 0
			else 0.0
		)

		# PSD using Welch's method (total power)
		freqs, psd_vals = signal.welch(arr, nperseg=n)
		psd_total_power: float = np.sum(psd_vals)

		# Linear regression using scipy.stats.linregress
		t: np.ndarray = np.arange(n)
		linreg_result = stats.linregress(t, arr)
		line_fit: dict[str, float] = dict(
			slope=linreg_result.slope,
			intercept=linreg_result.intercept,
			r2=linreg_result.rvalue**2,
		)

		timeseries_features: dict[str, float] = dict(
			zero_crossing_rate=np.sum(np.diff(np.signbit(arr))) / (n - 1),
			autocorr_lag1=autocorr_lag1,
			psd_total_power=psd_total_power,
			**{f"linreg.{k}": v for k, v in line_fit.items()},
		)

	return {**dist_features, **timeseries_features}
