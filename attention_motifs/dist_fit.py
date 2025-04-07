from typing import Protocol, Sequence

import numpy as np
from jaxtyping import Float
from scipy.stats import beta, gamma
from scipy.optimize import curve_fit


class FitFunction(Protocol):
	def __call__(self, x: Float, *params: float) -> Float:
		pass


FIT_FUNCTIONS: dict[str, tuple[Sequence[float], FitFunction]] = dict(
	beta=(
		(1.0, 1.0),
		lambda x, *params: beta.pdf(x, params[0], params[1]),
	),
	gen_beta=(
		(1.0, 1.0, 1.0),
		lambda x, *params: beta.pdf(x, params[0], params[1]) * params[2],
	),
	linear=(
		(1.0, 1.0),
		lambda x, *params: params[0] + params[1] * x,
	),
	cubic=(
		(1.0, 1.0, 1.0, 1.0),
		lambda x, *params: params[0]
		+ params[1] * x
		+ params[2] * (x**2)
		+ params[3] * (x**3),
	),
	weibull=(
		(1.0, 1.0, 1.0),
		lambda x, *params: params[2]
		* (params[0] / params[1])
		* (x / params[1]) ** (params[0] - 1)
		* np.exp(-((x / params[1]) ** params[0])),
	),
	power_law=(
		(1.0, 1.0),
		lambda x, *params: params[0] * np.power(x, params[1]),
	),
	exp_decay=(
		(1.0, 1.0, 1.0),
		lambda x, *params: params[0] * np.exp(-params[1] * x) + params[2],
	),
	gaussian=(
		(1.0, 1.0, 1.0),
		lambda x, *params: params[0]
		* np.exp(-0.5 * ((x - params[1]) / params[2]) ** 2),
	),
	inverse_gaussian=(
		(1.0, 1.0),
		lambda x, *params: np.sqrt(params[1] / (2 * np.pi * x**3))
		* np.exp(-params[1] * (x - params[0]) ** 2 / (2 * params[0] ** 2 * x)),
	),
	lognormal=(
		(1.0, 1.0),
		lambda x, *params: np.exp(-0.5 * ((np.log(x) - params[0]) / params[1]) ** 2)
		/ (x * params[1] * np.sqrt(2 * np.pi)),
	),
	gamma=(
		(1.0, 1.0),
		lambda x, *params: gamma.pdf(x, params[0], scale=params[1]),
	),
)


def hist_beta_fit(x: Float[np.ndarray, " n_ctx"]):
	popt, pcov = curve_fit(
		FIT_FUNCTIONS["beta"][1],
		x,
		FIT_FUNCTIONS["beta"][1](x, *FIT_FUNCTIONS["beta"][0]),
	)
