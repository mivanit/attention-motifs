import numpy as np
from jaxtyping import Float

from attention_motifs.util.util import prefix_dict
from attention_motifs.util.bins import Bins
from attention_motifs.features.vec_features import vec_features
from attention_motifs.math.cos_sim import cosine_similarity_matrix
from attention_motifs.math.math import skew_lt


def gram_features(G: Float[np.ndarray, "n_ctx n_ctx"]) -> dict[str, float]:
	# dbg_tensor(G)

	# --- existing histogram features ---
	bins: Bins = Bins(n_bins=32, start=0.0, stop=1.0)
	G_flat: Float[np.ndarray, " n"] = G.flatten()
	# Guard against NaN in input
	if np.any(np.isnan(G_flat)):
		G_flat = np.nan_to_num(G_flat, nan=0.5)
	x_hist: Float[np.ndarray, " n_bins"]
	x_hist, _ = np.histogram(G_flat, bins.edges, density=True)
	# Guard against NaN from empty histogram (when all values outside bin range)
	if np.any(np.isnan(x_hist)):
		x_hist = np.nan_to_num(x_hist, nan=0.0)

	# --- row-sum and col-sum features (position order is meaningful) ---
	row_sums: Float[np.ndarray, " n_ctx"] = G.sum(axis=1)
	if np.any(np.isnan(row_sums)):
		row_sums = np.nan_to_num(row_sums, nan=0.0)

	col_sums: Float[np.ndarray, " n_ctx"] = G.sum(axis=0)
	if np.any(np.isnan(col_sums)):
		col_sums = np.nan_to_num(col_sums, nan=0.0)

	return {
		**prefix_dict(vec_features(x_hist), prefix="hist"),
		**prefix_dict(vec_features(row_sums, reduced=False), prefix="rowsum"),
		**prefix_dict(vec_features(col_sums, reduced=False), prefix="colsum"),
		**prefix_dict(
			vec_features(G_flat, reduced=False, dist_only=True), prefix="flat"
		),
	}
	# TODO: mass as a function of distance from diagonal


def compute_scalar_features(
	A: Float[np.ndarray, "n_ctx n_ctx"],
) -> dict[str, float]:
	# dbg_tensor(A)
	A_log: Float[np.ndarray, "n_ctx n_ctx"] = np.nan_to_num(
		np.log(A + 1e-9), nan=-10, neginf=-20, posinf=0
	)
	# dbg_tensor(A_log)

	A_skew: Float[np.ndarray, "n_ctx n_ctx"] = skew_lt(A)
	# dbg_tensor(A_skew)
	A_log_skew: Float[np.ndarray, "n_ctx n_ctx"] = skew_lt(A_log)

	n_ctx: int = A.shape[0]
	idx: Float[np.ndarray, " n_ctx"] = np.arange(n_ctx, dtype=np.float64)

	# per-row entropy: -sum(A * log(A), axis=1)
	row_entropy: Float[np.ndarray, " n_ctx"] = -np.nansum(A * np.log(A + 1e-9), axis=1)

	# per-row weighted attention distance: sum(A[i,j] * |i-j|, axis=1)
	dist_matrix: Float[np.ndarray, "n_ctx n_ctx"] = np.abs(idx[:, None] - idx[None, :])
	attn_distance: Float[np.ndarray, " n_ctx"] = np.sum(A * dist_matrix, axis=1)

	# per-row max attention value
	row_max: Float[np.ndarray, " n_ctx"] = np.max(A, axis=1)

	# column sums: total attention received per position
	col_sum: Float[np.ndarray, " n_ctx"] = np.sum(A, axis=0)

	# band energy: fraction of attention mass within k diagonals
	band_k: int = max(1, n_ctx // 4)
	band_mask: Float[np.ndarray, "n_ctx n_ctx"] = (dist_matrix <= band_k).astype(
		np.float64
	)
	band_energy: float = float(np.sum(A * band_mask) / np.sum(A))

	# subdiagonal: attention to immediately preceding token
	# NOTE: length n_ctx-1; vec_features produces NaN for n_ctx<=2 (not encountered in practice)
	prev_tok: Float[np.ndarray, " n_ctx_minus1"] = np.diag(A, k=-1)

	return dict(
		# diagonal: standard features, fit diff to beta dist
		**prefix_dict(vec_features(A.diagonal(), reduced=False), prefix="diag"),
		# attention to position 0: BOS token for models with default_prepend_bos=True
		# (GPT-2, Pythia, TinyStories, Gemma), first content token otherwise (e.g. Llama)
		**prefix_dict(vec_features(A[:, 0], reduced=False), prefix="first_tok"),
		# attention to final position
		**prefix_dict(vec_features(A[:, -1], reduced=False), prefix="last_tok"),
		# attention to immediately preceding token (subdiagonal)
		**prefix_dict(vec_features(prev_tok, reduced=False), prefix="prev_tok"),
		# per-row entropy of attention distribution
		**prefix_dict(vec_features(row_entropy, reduced=False), prefix="row_entropy"),
		# weighted average attention distance per row
		**prefix_dict(
			vec_features(attn_distance, reduced=False), prefix="attn_distance"
		),
		# max attention value per row (peakedness)
		**prefix_dict(vec_features(row_max, reduced=False), prefix="row_max"),
		# total attention received per position
		**prefix_dict(vec_features(col_sum, reduced=False), prefix="col_sum"),
		# fraction of attention within k-diagonal band
		band_energy=band_energy,
		# transition tensor: standard features, standard features on diff, linear envelope on transition time
		# 	TODO: standard features on decay rate
		# markov transition not that important?
		# **prefix_dict(
		# 	tt_features(A),
		# 	prefix="markov_transition",
		# ),
		# # {log, raw} gram matrix of {rows, cols, rows of skewed}: beta fit hist
		# # 	TODO: fit fft in `gram_features`, but this is expensive
		**prefix_dict(
			gram_features(A @ A.T),
			prefix=["gram", "row"],
		),
		**prefix_dict(
			gram_features(A.T @ A),
			prefix=["gram", "col"],
		),
		**prefix_dict(
			gram_features(A_skew @ A_skew.T),
			prefix=["gram", "skew", "row"],
		),
		**prefix_dict(
			gram_features(A_skew.T @ A_skew),
			prefix=["gram", "skew", "col"],
		),
		**prefix_dict(
			gram_features(cosine_similarity_matrix(A_log)),
			prefix=["log", "gram", "row"],
		),
		**prefix_dict(
			gram_features(cosine_similarity_matrix(A_log, col=True)),
			prefix=["log", "gram", "col"],
		),
		**prefix_dict(
			gram_features(cosine_similarity_matrix(A_log_skew)),
			prefix=["log", "gram", "skew", "row"],
		),
		**prefix_dict(
			gram_features(cosine_similarity_matrix(A_log_skew, col=True)),
			prefix=["log", "gram", "skew", "col"],
		),
	)
