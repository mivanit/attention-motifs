import numpy as np
from jaxtyping import Float
import torch
import scipy.stats as stats

# attention-motifs
from attention_motifs.math.transition_tensor import (
	transition_tensor_torch,
)


def tt_features(
	A: Float[np.ndarray, "n_ctx n_ctx"],
	p_threshold: float = 0.0,
	device: str = "cuda",
) -> dict[str, float]:
	A_torch = torch.tensor(A, device=device)
	idxs_tr, tt_tr, _ = transition_tensor_torch(
		A_torch, exact=32, approx_l10=5, approx_pts=32
	)
	idxs = idxs_tr.cpu().numpy()
	tt = tt_tr.cpu().numpy()
	# idxs, tt, _ = transition_tensor(A, exact=32, approx_l10=5.0, approx_pts=32)

	output: dict[str, float] = dict()

	indices_raw = np.apply_along_axis(
		lambda row: np.searchsorted(row, p_threshold, side="right"),
		axis=0,
		arr=tt[:, :, 0],
	)
	idxs_with_inf = np.concatenate((idxs, [1e10]))
	indices_adjusted = np.array(idxs_with_inf[indices_raw], dtype=float)
	# if last element, set to inf
	# indices_adjusted[indices_raw == len(idxs)] = 1e10
	# indices_adjusted_l10 = np.log10(indices_adjusted[1:])
	# ax_tt_time.plot(indices_adjusted_l10, "ro")
	# removing because 0 variance
	# output.update(
	# 	prefix_dict(
	# 		vec_features(indices_adjusted_l10),
	# 		prefix="time",
	# 	)
	# )

	# removing because 0 variance
	# idxs_x = np.arange(len(indices_adjusted_l10))
	# for envtype in ("lower", "upper"):
	# 	m, b, r2 = compute_envelope_params(
	# 		x=idxs_x,
	# 		y=indices_adjusted_l10,
	# 		envelope_type=envtype,
	# 	)
	# 	output.update(
	# 		prefix_dict(
	# 			dict(
	# 				slope=m,
	# 				intercept=b,
	# 				r2=r2,
	# 			),
	# 			prefix=["env", envtype],
	# 		)
	# 	)

	indices_adjusted_diff = np.diff(indices_adjusted)
	output.update(
		dict(
			skewness=float(stats.skew(indices_adjusted_diff)),
			kurtosis=float(stats.kurtosis(indices_adjusted_diff)),
		)
		# all except skew and cov are highly correlated with other things
		# prefix_dict(
		# 	vec_features(indices_adjusted_diff),
		# 	prefix="diff",
		# )
	)

	return output
