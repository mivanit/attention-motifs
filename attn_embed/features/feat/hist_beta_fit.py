import numpy as np
from jaxtyping import Float

# muutils

# attention-motifs
from attention_motifs.util.bins import Bins
from attention_motifs.features.vec_features import vec_features
from attention_motifs.util import prefix_dict


def hist_beta_fit(
	x: Float[np.ndarray, " k"],
	bins: Bins,
) -> dict[str, float]:
	# dbg_tensor(x)
	x_hist, _ = np.histogram(x, bins.edges, density=True)
	# dbg_tensor(x_hist)

	# alpha_est, beta_est, loc, scale = stats.beta.fit(x, method="MM")

	output: dict[str, float] = dict(
		# alpha=alpha_est,
		# beta=beta_est,
		# loc=loc,
		# scale=scale,
		**prefix_dict(
			vec_features(x_hist),
			prefix=["hist", "raw"],
		),
		# **prefix_dict(
		# 	vec_features(
		# 		x_hist - scaled_beta(bins.centers, alpha_est, beta_est, scale)
		# 	),
		# 	prefix=["hist", "diff"],
		# ),
	)

	return output
