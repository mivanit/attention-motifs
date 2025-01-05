from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.fft import fft2

from pattern_lens.figure_util import matplotlib_figure_saver, save_matrix_wrapper, AttentionMatrix, Matrix2D
from pattern_lens.attn_figure_funcs import register_attn_figure_func
from pattern_lens.figures import figures_main


# gram matrices and FFTs

@register_attn_figure_func
@save_matrix_wrapper(fmt="png")
def fft(attn_matrix: AttentionMatrix) -> Matrix2D:
    "2D fft of raw attention matrix"
    fft_result = np.abs(fft2(attn_matrix))
    return np.fft.fftshift(fft_result)

@register_attn_figure_func
@save_matrix_wrapper(fmt="png")
def gram(attn_matrix: AttentionMatrix) -> Matrix2D:
    "Gram matrix A A^T"
    return attn_matrix @ attn_matrix.T

@register_attn_figure_func
@save_matrix_wrapper(fmt="png")
def gram_col(attn_matrix: AttentionMatrix) -> Matrix2D:
    "Column-wise Gram matrix A^T A"
    return attn_matrix.T @ attn_matrix


@register_attn_figure_func
@save_matrix_wrapper(fmt="png")
def gram_fft(attn_matrix: AttentionMatrix) -> Matrix2D:
    "2D fft of Gram matrix A A^T"
    gram = attn_matrix @ attn_matrix.T
    return np.abs(fft2(gram))

@register_attn_figure_func
@save_matrix_wrapper(fmt="png")
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
    ax.legend()










