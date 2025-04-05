import numpy as np
import matplotlib.pyplot as plt

from muutils.spinner import SpinnerContext
from muutils.tensor_info import array_summary

from pattern_lens.consts import DIVIDER_S1, DIVIDER_S2, SPINNER_KWARGS
from pattern_lens.figure_util import (
	AttentionMatrix,
)
from pattern_lens.attn_figure_funcs import (
	register_attn_figure_multifunc,
)
from pattern_lens.figure_util import matplotlib_multifigure_saver
from pattern_lens.figures import figures_main

from attention_motifs.math import compute_envelope_params, linear_plot
from attention_motifs.transition_tensor import transition_tensor


# gram matrices and FFTs


"""
	gram = pattern @ pattern.T
	fft = fft2(gram)
	fft_shifted = np.fft.fftshift(fft)
	data: dict = {
		"Pattern": pattern,
		"Gram": gram,
		"FFT (abs)": np.abs(fft),
		"FFT (log abs)": np.log(np.abs(fft)),
		"FFT (angle)": np.angle(fft),
		"FFT (real)": np.real(fft),
		"FFT (log abs real)": np.log(np.abs(np.real(fft))),
		"FFT (imag)": np.imag(fft),
		"Shifted FFT (abs)": np.abs(fft_shifted),
		"Shifted FFT (log abs)": np.log(np.abs(fft_shifted)),
		"Shifted FFT (angle)": np.angle(fft_shifted),
		"Shifted FFT (real)": np.real(fft_shifted),
		"Shifted FFT (log abs real)": np.log(np.abs(np.real(fft_shifted))),
		"Shifted FFT (imag)": np.imag(fft_shifted),
"""


# @register_attn_figure_func
# @matplotlib_multifigure_saver(["fft", "fft_abs"])
# def fft(attn_matrix: AttentionMatrix, axes: dict[str, plt.Axes]) -> None:
# 	"abs of 2D fft of raw attention matrix"
# 	fft = fft2(attn_matrix)
# 	axes["fft"].matshow(fft.real, cmap="viridis")
# 	axes["fft_abs"].matshow(np.abs(fft), cmap="viridis")


@register_attn_figure_multifunc(["diag_wgts", "first_tok_wgts"])
@matplotlib_multifigure_saver(["diag_wgts", "first_tok_wgts"])
def basic(attn_matrix: AttentionMatrix, axes: dict[str, plt.Axes]) -> None:
	"abs of 2D fft of raw attention matrix"
	axes["diag_wgts"].plot(np.diag(attn_matrix), "o")
	axes["first_tok_wgts"].plot(attn_matrix[0], "o")


@register_attn_figure_multifunc(["tensor", "time", "time_diffs"])
@matplotlib_multifigure_saver(["tensor", "time", "time_diffs"])
def markov_transition(attn_matrix: AttentionMatrix, axes: dict[str, plt.Axes]) -> None:
	# manual config here
	p_threshold: float = 0.95
	idxs, tt, res = transition_tensor(
		attn_matrix, exact=100, approx_l10=7.0, approx_pts=100
	)
	n_ctx: int = attn_matrix.shape[0]

	#
	axes["tensor"].set_title("transition tensor")
	axes["tensor"].matshow(
		np.log1p(1 - tt[:, :, 0].T), aspect=(tt.shape[0] / tt.shape[1])
	)
	axes["tensor"].set_xticks(range(len(idxs)))
	axes["tensor"].set_xticklabels(idxs)
	axes["tensor"].tick_params(axis="x", rotation=90)
	axes["tensor"].set_xlabel("markov iteration")
	axes["tensor"].set_ylabel("token idx")

	#
	axes["time"].set_title(f"time to transition probability > {p_threshold}")
	indices_raw = np.apply_along_axis(
		lambda row: np.searchsorted(row, p_threshold, side="right"),
		axis=0,
		arr=tt[:, :, 0],
	)
	axes["tensor"].plot(indices_raw, np.arange(indices_raw.shape[0]), "r.")
	idxs_with_inf = np.concatenate((idxs, [1e10]))
	indices_adjusted = np.array(idxs_with_inf[indices_raw], dtype=float)
	# if last element, set to inf
	# indices_adjusted[indices_raw == len(idxs)] = 1e10
	indices_adjusted_l10 = np.log10(indices_adjusted[1:])
	axes["time"].plot(indices_adjusted_l10, "ro")
	axes["time"].set_xlabel("token idx")
	axes["time"].set_ylabel("log10(iters to transition)")
	idxs_x = np.arange(len(indices_adjusted_l10))
	for envtype in ("lower", "upper", "bestfit"):
		env_lower = compute_envelope_params(
			x=idxs_x,
			y=indices_adjusted_l10,
			envelope_type=envtype,
		)
		axes["time"].plot(
			idxs_x,
			linear_plot(idxs_x, env_lower[0], env_lower[1]),
			label=f"{envtype}, $R^2={env_lower[2]:.3f}$",
		)
	axes["time"].legend()

	#
	indices_adjusted_diff = np.diff(indices_adjusted)
	axes["time_diffs"].set_title(
		f"dist of transition times diff\n${array_summary(indices_adjusted_diff, fmt='latex', dtype=False)}$"
	)
	axes["time_diffs"].hist(indices_adjusted_diff, bins=10)
	axes["time_diffs"].set_xlabel("diff")
	axes["time_diffs"].set_ylabel("count")


# @register_attn_figure_func
# @save_matrix_wrapper(fmt="png", diverging_colormap=True)
# def gram(attn_matrix: AttentionMatrix) -> Matrix2D:
# 	"Gram matrix A A^T"
# 	return attn_matrix @ attn_matrix.T


# @register_attn_figure_func
# @save_matrix_wrapper(fmt="png", diverging_colormap=True, normalize=True)
# def gram_col(attn_matrix: AttentionMatrix) -> Matrix2D:
# 	"Column-wise Gram matrix A^T A"
# 	return attn_matrix.T @ attn_matrix


# @register_attn_figure_func
# @save_matrix_wrapper(fmt="png", diverging_colormap=True, normalize=True)
# def gram_fft(attn_matrix: AttentionMatrix) -> Matrix2D:
# 	"2D fft of Gram matrix A A^T"
# 	gram = attn_matrix @ attn_matrix.T
# 	return np.abs(fft2(gram))


# @register_attn_figure_func
# @save_matrix_wrapper(fmt="png", diverging_colormap=True, normalize=True)
# def gram_col_fft(attn_matrix: AttentionMatrix) -> Matrix2D:
# 	"2D fft of column-wise Gram matrix A^T A"
# 	col_gram = attn_matrix.T @ attn_matrix
# 	return np.abs(fft2(col_gram))


# @register_attn_figure_func
# @matplotlib_figure_saver(fmt="svgz")
# def gram_hist(attn_matrix: AttentionMatrix, ax: plt.Axes) -> None:
# 	gram = attn_matrix @ attn_matrix.T
# 	flat_gram = gram.flatten()

# 	ax.hist(flat_gram, bins=50, density=True, alpha=0.7)
# 	# x = np.linspace(0, 1, 100)
# 	# ax.plot(x, beta.pdf(x, a, b), "r-", lw=2, label="Beta fit")
# 	# ax.set_title(f"Histogram of Gram Matrix Values (Beta: a={a:.2f}, b={b:.2f})")
# 	ax.set_title("Histogram of Gram Matrix Values")
# 	# ax.legend()


# @register_attn_figure_func
# @matplotlib_figure_saver(fmt="svgz")
# def degree_dist(attn_matrix: AttentionMatrix, ax: plt.Axes) -> None:
# 	"sum each column, plot histogram"
# 	degrees_ax0 = np.sum(attn_matrix, axis=0)
# 	ax.hist(degrees_ax0, bins=50, density=True, alpha=0.7, label="Ax0")
# 	degrees_ax1 = np.sum(attn_matrix, axis=1)
# 	ax.hist(degrees_ax1, bins=50, density=True, alpha=0.7, label="Ax1")
# 	ax.legend()
# 	ax.set_title("Histogram of Node Degrees")


if __name__ == "__main__":
	import argparse

	print(DIVIDER_S1)
	with SpinnerContext(message="parsing args", **SPINNER_KWARGS):
		arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()
		# input and output
		arg_parser.add_argument(
			"--model",
			"-m",
			type=str,
			required=True,
			help="The model name(s) to use. comma separated with no whitespace if multiple",
		)
		arg_parser.add_argument(
			"--save-path",
			"-s",
			type=str,
			required=False,
			help="The path to save the attention patterns",
		)
		# number of samples
		arg_parser.add_argument(
			"--n-samples",
			"-n",
			type=int,
			required=False,
			help="The max number of samples to process, do all in the file if None",
			default=None,
		)
		# force overwrite of existing figures
		arg_parser.add_argument(
			"--force",
			"-f",
			type=bool,
			required=False,
			help="Force overwrite of existing figures",
			default=False,
		)

		# parallel processing
		arg_parser.add_argument(
			"--parallel",
			"-p",
			type=int,
			required=False,
			help="Use parallel processing",
			default=1,
		)

		args: argparse.Namespace = arg_parser.parse_args()

	print(f"args parsed: {args}")

	models: list[str]
	if "," in args.model:
		models = args.model.split(",")
	else:
		models = [args.model]

	n_models: int = len(models)
	for idx, model in enumerate(models):
		print(DIVIDER_S2)
		print(f"processing model {idx + 1} / {n_models}: {model}")
		print(DIVIDER_S2)
		figures_main(
			model_name=model,
			save_path=args.save_path,
			n_samples=args.n_samples,
			force=args.force,
			parallel=bool(args.parallel),
		)

	print(DIVIDER_S1)
