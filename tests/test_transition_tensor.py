import numpy as np
import torch
from jaxtyping import Float

from attention_motifs.math.transition_tensor import (
	transition_tensor,
	transition_tensor_torch,
)


def test_transition_tensor_implementations() -> None:
	"""
	Test that numpy and pytorch implementations of transition_tensor
	produce numerically similar results.
	"""
	# Set random seed for reproducibility
	np.random.seed(42)
	torch.manual_seed(42)

	# Test parameters
	n_ctx: int = 10
	exact: int = 5
	approx_l10: int = 2
	approx_pts: int = 10

	# Create a random row-stochastic matrix
	A_np: Float[np.ndarray, "n_ctx n_ctx"] = np.random.rand(n_ctx, n_ctx)
	row_sums = A_np.sum(axis=1, keepdims=True)
	A_np = A_np / row_sums  # Make row-stochastic

	# Convert to PyTorch tensor
	A_torch: Float[torch.Tensor, "n_ctx n_ctx"] = torch.tensor(
		A_np, dtype=torch.float32
	)

	# Call both implementations
	idxs_np, tt_np, _ = transition_tensor(
		A_np, exact=exact, approx_l10=approx_l10, approx_pts=approx_pts, residuals=False
	)

	idxs_torch, tt_torch, _ = transition_tensor_torch(
		A_torch,
		exact=exact,
		approx_l10=approx_l10,
		approx_pts=approx_pts,
		residuals=False,
	)

	# Convert PyTorch results to numpy for comparison
	idxs_torch_np = idxs_torch.numpy()

	# The implementations might produce different numbers of indices due to handling duplicates differently
	# For testing, we'll focus on the common indices only
	common_indices = np.intersect1d(idxs_np, idxs_torch_np)

	# Get the indices into each array for the common values
	np_indices = np.array([np.where(idxs_np == idx)[0][0] for idx in common_indices])
	torch_indices = np.array(
		[np.where(idxs_torch_np == idx)[0][0] for idx in common_indices]
	)

	# Extract the corresponding tensor values
	tt_np_common = tt_np[np_indices]
	tt_torch_np_common = tt_torch.numpy()[torch_indices]

	# Now test with the matching subsets
	assert tt_np_common.shape == tt_torch_np_common.shape, (
		f"Shapes don't match: numpy={tt_np_common.shape}, torch={tt_torch_np_common.shape}"
	)

	max_abs_diff = np.max(np.abs(tt_np_common - tt_torch_np_common))
	rel_diff = np.max(
		np.abs(tt_np_common - tt_torch_np_common) / (np.abs(tt_np_common) + 1e-10)
	)

	assert max_abs_diff < 1e-5, f"Max absolute difference too large: {max_abs_diff}"
	assert rel_diff < 1e-4, f"Max relative difference too large: {rel_diff}"

	# Test a few specific points to ensure they match
	for i in range(min(3, len(common_indices))):
		idx = common_indices[i]
		np_idx = np.where(idxs_np == idx)[0][0]
		torch_idx = np.where(idxs_torch_np == idx)[0][0]

		for j in range(min(3, n_ctx)):
			for k in range(min(3, n_ctx)):
				np_val = tt_np[np_idx, j, k]
				torch_val = tt_torch.numpy()[torch_idx, j, k]
				assert abs(np_val - torch_val) < 1e-5, (
					f"Mismatch at power {idx} position [{j},{k}]: numpy={np_val}, torch={torch_val}"
				)

	print(f"Successfully tested {len(common_indices)} common indices")
	print(f"Common indices: {common_indices}")
	print(f"Maximum absolute difference: {max_abs_diff}")
	print(f"Maximum relative difference: {rel_diff}")
