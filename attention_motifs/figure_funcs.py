import functools

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2

from muutils.spinner import SpinnerContext

from pattern_lens.consts import DIVIDER_S1, DIVIDER_S2, SPINNER_KWARGS
from pattern_lens.figure_util import (
	matplotlib_figure_saver,
	save_matrix_wrapper,
	AttentionMatrix,
	Matrix2D,
)
from pattern_lens.attn_figure_funcs import register_attn_figure_func
from pattern_lens.figures import figures_main


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

@register_attn_figure_func
@save_matrix_wrapper(fmt="png", normalize=True)
def fft(attn_matrix: AttentionMatrix) -> Matrix2D:
	"abs of 2D fft of raw attention matrix"
	return fft2(attn_matrix)

@register_attn_figure_func
@save_matrix_wrapper(fmt="png", normalize=True)
def fft_abs(attn_matrix: AttentionMatrix) -> Matrix2D:
	"abs of 2D fft of raw attention matrix"
	return np.abs(fft2(attn_matrix))


@register_attn_figure_func
@save_matrix_wrapper(fmt="png", diverging_colormap=True)
def gram(attn_matrix: AttentionMatrix) -> Matrix2D:
	"Gram matrix A A^T"
	return attn_matrix @ attn_matrix.T


@register_attn_figure_func
@save_matrix_wrapper(fmt="png", diverging_colormap=True, normalize=True)
def gram_col(attn_matrix: AttentionMatrix) -> Matrix2D:
	"Column-wise Gram matrix A^T A"
	return attn_matrix.T @ attn_matrix


@register_attn_figure_func
@save_matrix_wrapper(fmt="png", diverging_colormap=True, normalize=True)
def gram_fft(attn_matrix: AttentionMatrix) -> Matrix2D:
	"2D fft of Gram matrix A A^T"
	gram = attn_matrix @ attn_matrix.T
	return np.abs(fft2(gram))


@register_attn_figure_func
@save_matrix_wrapper(fmt="png", diverging_colormap=True, normalize=True)
def gram_col_fft(attn_matrix: AttentionMatrix) -> Matrix2D:
	"2D fft of column-wise Gram matrix A^T A"
	col_gram = attn_matrix.T @ attn_matrix
	return np.abs(fft2(col_gram))


@register_attn_figure_func
@matplotlib_figure_saver(fmt="svgz")
def gram_hist(attn_matrix: AttentionMatrix, ax: plt.Axes) -> None:
	gram = attn_matrix @ attn_matrix.T
	flat_gram = gram.flatten()

	ax.hist(flat_gram, bins=50, density=True, alpha=0.7)
	# x = np.linspace(0, 1, 100)
	# ax.plot(x, beta.pdf(x, a, b), "r-", lw=2, label="Beta fit")
	# ax.set_title(f"Histogram of Gram Matrix Values (Beta: a={a:.2f}, b={b:.2f})")
	ax.set_title("Histogram of Gram Matrix Values")
	# ax.legend()


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
		print(f"processing model {idx+1} / {n_models}: {model}")
		print(DIVIDER_S2)
		figures_main(
			model_name=model,
			save_path=args.save_path,
			n_samples=args.n_samples,
			force=args.force,
			parallel=bool(args.parallel),
		)

	print(DIVIDER_S1)
