import numpy as np
from jaxtyping import Float
from scipy.optimize import linprog
import scipy.stats as stats

import muutils.dbg

muutils.dbg.DBG_TENSOR_ARRAY_SUMMARY_DEFAULTS["sparkline_logy"] = True


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	ss_res: float = np.sum((y_true - y_pred) ** 2)
	ss_tot: float = np.sum((y_true - np.mean(y_true)) ** 2)
	return 1 - ss_res / ss_tot if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)


def scaled_beta(x: np.ndarray, alpha: float, beta: float, scale: float) -> np.ndarray:
	"""Scaled beta distribution."""
	return stats.beta.pdf(x, alpha, beta) * scale


def compute_envelope_params(
	x: list[float],
	y: list[float],
	envelope_type: str = "lower",
) -> tuple[float, float, float]:
	"""Compute the parameters (slope, intercept) for an envelope or best-fit line and its R^2 measure.

	This function computes a line L(x) = m*x + b along with an R^2 measure of fit.
	It supports three modes based on the envelope_type parameter:
	  - "lower": Computes the highest line that lies entirely below the points (i.e. m*x + b <= y for all x)
	             by maximizing m*x_mid + b, where x_mid = (min(x) + max(x)) / 2.
	  - "upper": Computes the lowest line that lies entirely above the points (i.e. m*x + b >= y for all x)
	             by minimizing m*x_mid + b.
	  - "bestfit": Computes the ordinary least-squares line of best fit for the data.

	# Parameters:
	 - `x : list[float]`
	     The x-values (assumed to be in increasing order).
	 - `y : list[float]`
	     The y-values corresponding to x.
	 - `envelope_type : str`
	     One of "lower", "upper", or "bestfit" indicating which line to compute.

	# Returns:
	 - `Tuple[float, float, float]`
	     A tuple (m, b, r2) where the line is given by L(x) = m*x + b and r2 is the coefficient of determination.

	# Raises:
	 - `ValueError` : if the input lists are not of equal length, contain fewer than 3 points,
	   or if envelope_type is not one of "lower", "upper", or "bestfit".
	"""
	if len(x) != len(y) or len(x) < 3:
		raise ValueError(
			"Input lists must have the same length and contain at least 3 points."
		)

	x_arr: np.ndarray = np.array(x, dtype=float)
	y_arr: np.ndarray = np.array(y, dtype=float)

	if envelope_type == "bestfit":
		# Ordinary least squares best-fit line.
		m: float
		b: float
		m, b = np.polyfit(x_arr, y_arr, 1)
		y_pred: np.ndarray = m * x_arr + b
		r2: float = compute_r2(y_arr, y_pred)
		return m, b, r2

	# For envelope constraints we compute the midpoint of x.
	x_min: float = np.min(x_arr)
	x_max: float = np.max(x_arr)
	x_mid: float = (x_min + x_max) / 2.0

	if envelope_type == "lower":
		# For lower envelope: maximize m*x_mid + b subject to m*x_i + b <= y_i.
		# This is equivalent to minimizing -m*x_mid - b.
		c: np.ndarray = np.array([-x_mid, -1.0])
		A_ub: np.ndarray = np.column_stack((x_arr, np.ones_like(x_arr)))
		b_ub: np.ndarray = y_arr.copy()
	elif envelope_type == "upper":
		# For upper envelope: minimize m*x_mid + b subject to m*x_i + b >= y_i.
		# Rewrite constraint: -m*x_i - b <= -y_i.
		c: np.ndarray = np.array([x_mid, 1.0])
		A_ub: np.ndarray = -np.column_stack((x_arr, np.ones_like(x_arr)))
		b_ub: np.ndarray = -y_arr.copy()
	else:
		raise ValueError("envelope_type must be 'lower', 'upper', or 'bestfit'.")

	res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, method="highs")
	if not res.success:
		raise ValueError("Linear programming failed: " + res.message)

	m: float = res.x[0]
	b: float = res.x[1]
	y_pred: np.ndarray = m * x_arr + b
	r2: float = compute_r2(y_arr, y_pred)
	return m, b, r2


def linear_plot(
	x: Float[np.ndarray, " len(x)"], m: float, b: float
) -> Float[np.ndarray, " len(x)"]:
	"""Compute the linear function L(x) = m*x + b."""
	return m * x + b


def skew_lt(
	L: Float[np.ndarray, "n n"],
) -> Float[np.ndarray, "n n"]:
	"""Shift rows of a lower-triangular matrix so that its diagonal becomes the rightmost column.

	# Parameters:
	 - `L : Float[np.ndarray, "n n"]`
	   A square lower-triangular matrix of shape `(n, n)`.

	# Returns:
	 - `Float[np.ndarray, "n n"]`
	   A matrix of shape `(n, n)` with rows shifted so the original diagonal is in the rightmost column.
	"""
	n: int = L.shape[0]
	S: Float[np.ndarray, "n n"] = np.zeros_like(L)
	i, j = np.tril_indices(n)  # row indices i, col indices j of the lower triangle
	S[i, j + (n - i - 1)] = L[i, j]  # shift columns so diagonal ends up at column n-1
	return S
