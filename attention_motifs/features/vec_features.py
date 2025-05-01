from typing import Final
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
from jaxtyping import Float
from numba import njit

from muutils.dbg import dbg_tensor

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

	# return {**dist_features, **timeseries_features}
	# merge dicts in a way that keeps numba happy
	features: dict[str, float] = dict()
	features.update(dist_features)
	features.update(timeseries_features)
	return features


VEC_FEATURES_NAMES: Final[list[str]] = [
    # distribution statistics
    "mean",
    "median",
    "variance",          # unbiased (ddof = 1)
    "std",
    "skewness",          # Pearson, bias-corrected
    "kurtosis",          # excess (Fisher, i.e. −3)
    "entropy",           # Shannon, base-2
    "L1_norm",
    "L2_norm",
    "rms",
    "energy",
    # time-series statistics
    "zero_crossing_rate",
    "autocorr_lag1",
    # "psd_total_power",
    "linreg.slope",
    "linreg.intercept",
    "linreg.r2",
]

N_VEC_FEATURES: Final[int] = len(VEC_FEATURES_NAMES)

@njit(cache=True, fastmath=True)
def vec_features_arr(
    x: Float[np.ndarray, "*n"],
) -> Float[np.ndarray, f"{N_VEC_FEATURES}"]:
    """Return the full 17-element feature vector for a 1-D array.

    The column order is fixed by ``VEC_FEATURES_NAMES`` so that downstream
    code can reconstruct a ``dict`` via

    ```python
    feats = vec_features_arr(arr)
    feat_dict = dict(zip(VEC_FEATURES_NAMES, feats))
    ```

    # Parameters
     - `x : Float[np.ndarray, "*n"]`  
       1-D input vector (any numeric dtype; cast to ``float64``).

    # Returns
     - `Float[np.ndarray, f"{N_VEC_FEATURES}"]`  
       Feature array in the exact order defined above.

    # Raises
     - `ValueError` : if the input is not 1-D.
    """
    if x.ndim != 1:
        raise ValueError("vec_features_arr expects a 1-D array")

    n: int = x.size
    x64 = x.astype(np.float64)

    # -----------------------------------------------------------------
    # Distribution features
    # -----------------------------------------------------------------
    mean = x64.mean()
    median = np.median(x64)

    diff = x64 - mean
    var_unbiased = diff.dot(diff) / (n - 1) if n > 1 else 0.0
    std_unbiased = np.sqrt(var_unbiased)

    skew = (diff ** 3).sum() / n / (std_unbiased ** 3 + 1e-12)
    kurt = (diff ** 4).sum() / n / (std_unbiased ** 4 + 1e-12) - 3.0

    # histogram for entropy ------------------------------------------------
    bins = 10 if n >= 10 else n
    # manual histogram (NumPy's is not Numba-safe for dynamic bins)
    min_val = x64.min()
    max_val = x64.max()
    width = max_val - min_val + 1e-12  # avoid zero width
    hist = np.zeros(bins, dtype=np.int64)
    for v in x64:
        idx = int((v - min_val) / width * bins)
        if idx == bins:  # right edge
            idx -= 1
        hist[idx] += 1
    total = hist.sum()
    entropy = 0.0
    if total > 0:
        for h in hist:
            if h > 0:
                p = h / total
                entropy -= p * (np.log(p) / np.log(2.0))  # log2

    L1 = np.abs(x64).sum() / n
    L2 = np.sqrt((x64 ** 2).sum()) / n
    rms = np.sqrt((x64 ** 2).mean())
    energy = (x64 ** 2).sum()

    # -----------------------------------------------------------------
    # Time-series features
    # -----------------------------------------------------------------
    # zero-crossing rate
    zc = 0
    for i in range(1, n):
        if (x64[i - 1] >= 0 and x64[i] < 0) or (x64[i - 1] < 0 and x64[i] >= 0):
            zc += 1
    zcr = zc / (n - 1) if n > 1 else 0.0

    # autocorrelation lag-1
    if n > 1:
        x0 = x64[:-1]
        x1 = x64[1:]
        m0 = x0.mean()
        m1 = x1.mean()
        cov = ((x0 - m0) * (x1 - m1)).sum() / (n - 1)
        std0 = np.sqrt(((x0 - m0) ** 2).sum() / (n - 1) + 1e-12)
        std1 = np.sqrt(((x1 - m1) ** 2).sum() / (n - 1) + 1e-12)
        autocorr = cov / (std0 * std1)
    else:
        autocorr = 0.0

    # PSD power via FFT (simplified Welch: single segment)
    # fft_vals = np.fft.rfft(x64)
    # psd_total = (np.abs(fft_vals) ** 2).sum() / n

    # linear regression on t = 0..n-1
    t = np.arange(n, dtype=np.float64)
    mean_t = (n - 1) / 2.0
    var_t = ((t - mean_t) ** 2).sum() / (n - 1) if n > 1 else 0.0
    cov_tx = ((t - mean_t) * diff).sum() / (n - 1) if n > 1 else 0.0

    slope = cov_tx / (var_t + 1e-12)
    intercept = mean - slope * mean_t
    r = cov_tx / (np.sqrt(var_t) * std_unbiased + 1e-12)
    r2 = r * r

    # -----------------------------------------------------------------
    # Pack result
    # -----------------------------------------------------------------
    return np.array(
        [
            mean,
            median,
            var_unbiased,
            std_unbiased,
            skew,
            kurt,
            entropy,
            L1,
            L2,
            rms,
            energy,
            zcr,
            autocorr,
            # psd_total,
            slope,
            intercept,
            r2,
        ],
        dtype=np.float64,
    )


def vec_features_fast(
	x: Float[np.ndarray, "*n"],
) -> Float[np.ndarray, f"{N_VEC_FEATURES}"]:
	return {k: v for k, v in zip(VEC_FEATURES_NAMES, vec_features_arr(x))}