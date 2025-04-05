from typing import Callable

import numpy as np
from jaxtyping import Float, Int
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from muutils.dbg import dbg, dbg_tensor

from attention_motifs.matrix_powers import matrix_powers


def cross_entropy(
	p: Float[np.ndarray, " d"],
	q: Float[np.ndarray, " d"],
) -> float:
	"$H(p,q)=-\sum _{x\in {\mathcal {X}}}p(x)\,\log q(x)$"
	return -np.sum(p * np.log(q))

def l2_norm(
	p: Float[np.ndarray, " d"],
	q: Float[np.ndarray, " d"],
) -> float:
	return np.linalg.norm(p - q, ord=2)

def kl_divergence(
	p: Float[np.ndarray, " d"],
	q: Float[np.ndarray, " d"],
) -> float:
	return np.sum(p * np.log(p / q))

def sigmoid(
	x: Float[np.ndarray, " d"],  # input
	x0: float,  # x-value of the sigmoid's midpoint
	k: float,  # steepness of the sigmoid
	L: float = 1.0,  # curve's maximum value
	b: float = 0.0,  # curve's minimum value
) -> Float[np.ndarray, " d"]:
	"""Sigmoid function

	- `x: Float[np.ndarray, " d"]`
	  input
	- `x0: float`
	  x-value of the sigmoid's midpoint
	- `k: float`
	  steepness of the sigmoid
	- `L: float`
	  curve's maximum value
	  (default: `1.0`)
	- `b: float`
	  curve's minimum value
	  (default: `0.0`)

	"""
	return L / (1 + np.exp(-k * (x - x0))) + b


def transition_tensor(
	A: Float[np.ndarray, "n_ctx n_ctx"],
	exact: int = 10,
	approx_l10: int = 3,
	approx_pts: int = 20,
	res_norm: Callable[
		[Float[np.ndarray, " d"], Float[np.ndarray, " d"]], float
	] = l2_norm,
) -> tuple[
	Int[np.ndarray, " n_idxs"],  # idxs
	Float[np.ndarray, "n_idxs n_ctx n_ctx"],  # resampled transition tensor
	Float[np.ndarray, "n_idxs n_ctx"],  # resampled residuals
]:
	"""
	Compute the 3D transition tensor, residuals, and then resample

	- compute: `X` of shape `(K, n_ctx, n_ctx)` such that `X[t, i, j]` = Probability of being in state j at step k if we start in state i at step 0. `X[t, i, j] = A^t_[i, j]`
	- compute: `residuals` of shape `(K, n_ctx)` such that `residuals[t, i] = res_norm(X[t, i, :], X[t-1, i, :])`
	- resample both with the first `exact` points and then `approx_pts` points between `exact` and `approx_l10**10` on a log scale.

	# Parameters
	- `A: np.ndarray`
	    An n x n transition matrix (row-stochastic for a standard Markov chain).
	- `exact: int`
	    The number of exact points to include in the resampling.
	- `approx_l10: int`
	    The log base 10 of the maximum number of points to include in the resampling.
	- `approx_pts: int`
	    The number of points to include in the resampling between `exact` and `approx_l10**10`.

	# Returns
	- `idxs: Int[np.ndarray, "n_idxs"]`
	- `X: Float[np.ndarray, "n_idxs n_ctx n_ctx"]`
	    A 3D NumPy array of shape `(K, n, n)` such that:
	    - X[k, :, k] is A^(idxs[k]) for k >= 1.
	    - X[0, :, :] is the identity matrix (k=0).
	- `residuals: Float[np.ndarray, "n_idxs n_ctx"]`
	    A 2D NumPy array of shape `(K, n)` such that:
	    - residuals[k, i] is the residual between X[k, i, :] and X[k-1, i, :].
		- residuals[0, i] is NaN.
	"""
	max_K: int = int(np.power(10, approx_l10)) + 1
	assert max_K > exact, f"approx_l10 must be greater than exact {exact = } {max_K = } {approx_l10 = }"

	n_ctx: int = A.shape[0]
	assert A.shape[0] == A.shape[1], f"Matrix must be square, but got {A.shape = }"

	# indices we want
	resampled_idxs: Int[np.ndarray, "approx_pts"] = np.logspace(
		np.log10(exact), approx_l10, approx_pts, base=10, dtype=int
	)
	idxs: Int[np.ndarray, "n_idxs"] = np.concatenate([np.arange(exact), resampled_idxs])
	n_idxs: int = len(idxs)


	# Compute powers of A iteratively
	needed_powers: list[int] = sorted(
		p
		for p in set(idxs).union(set(idxs - 1))
		if p >= 0
	)
	A_powers_arr: Float[np.ndarray, "len(needed_powers) n_ctx n_ctx"] = matrix_powers(
		A, powers=needed_powers
	)
	A_powers: dict[int, Float[np.ndarray, "n_ctx n_ctx"]] = {
		p: A_powers_arr[i] for i, p in enumerate(needed_powers)
	}
	A_powers[-1] = np.eye(n_ctx, dtype=A.dtype)
	tt_resampled: Float[np.ndarray, "n_idxs n_ctx n_ctx"] = np.array(
		[A_powers[p] for p in idxs],
		dtype=A.dtype,
	)

	# compute residuals
	res_resampled: Float[np.ndarray, "n_idxs n_ctx"] = np.full(
		(n_idxs, n_ctx), np.nan, dtype=A.dtype
	)
	for i_idx in range(n_idxs):
		for i_ctx in range(n_ctx):
			res_resampled[i_idx, i_ctx] = res_norm(
				tt_resampled[i_idx, i_ctx, :], tt_resampled[i_idx - 1, i_ctx, :]
			)

	dbg_tensor(res_resampled)

	# )
	# 	[
	# 		[
	# 			res_norm(A_powers[p][i, :], A_powers[p - 1][i, :])
	# 			for i in range(n_ctx)
	# 		]
	# 		for p in idxs
	# 	],
	# 	dtype=A.dtype,
	# )

	return idxs, tt_resampled, res_resampled


def tt_fig(
	A: np.ndarray,
	axs: np.ndarray,
	p_idx: int,
	lyr: int,
	head: int,
):
	#
	axs[0].set_title("raw attention")
	axs[0].matshow(A)

	# compute transition_tensor
	idxs, tt, res = transition_tensor(A, exact=20, approx_l10=3.0, approx_pts=20)

	#
	axs[1].set_title("transition tensor")
	axs[1].matshow(np.log10(1 - tt[:, :, 0].T + 1e-8), aspect=tt.shape[0] / tt.shape[1])
	axs[1].set_xticks(range(len(idxs)))
	axs[1].set_xticklabels(idxs)
	axs[1].tick_params(axis="x", rotation=90)

	#
	axs[2].set_title("transition curves")
	axs[2].set_yscale("log")
	cmap = plt.get_cmap("viridis")
	for i in range(tt.shape[0]):
		axs[2].plot(tt[i, 0, :], label=f"i={i}", color=cmap(i / tt.shape[0]))
		x: np.ndarray = np.arange(tt.shape[2])
		y: np.ndarray = tt[i, 0, :]
		# Initial parameter guess: amplitude, midpoint, steepness, baseline
		p0: list[float] = [max(y) - min(y), np.median(x), 1.0, min(y)]
		try:
			popt: np.ndarray
			popt, _ = curve_fit(sigmoid, x, y, p0=p0)
			y_fit: np.ndarray = sigmoid(x, *popt)
			axs[2].plot(x, y_fit, ":", label=f"i={i} fit", color=cmap(i / tt.shape[0]))
		except Exception:
			# Fallback: plot raw data if the fit fails
			axs[2].plot(x, y, label=f"i={i}", color=cmap(i / tt.shape[0]))

	#
	axs[3].set_title("residuals tensor")
	# aspect shoudl be such that the image is square, although the matrix is not
	axs[3].matshow(res.T, aspect=(res.shape[0] / res.shape[1]))
	axs[3].set_xticks(range(len(idxs)))
	axs[3].set_xticklabels(idxs)
	axs[3].tick_params(axis="x", rotation=90)

	#
	axs[4].set_title("residuals curves")
	dbg(res.shape)
	for i in range(res.shape[0]):
		axs[4].plot(res[i, :], label=f"i={i}", color=cmap(i / res.shape[0]))
		axs[4].set_xticks(range(len(idxs)))
		axs[4].set_xticklabels(idxs)
		axs[4].tick_params(axis="x", rotation=90)
		axs[4].set_yscale("log")
