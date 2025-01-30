import pytest
import torch
from torch import Tensor
from jaxtyping import Float, Int

from attention_motifs.ae import contrastive_loss  # replace with actual import path


@pytest.mark.parametrize(
	"batch_size,latent_dim,num_classes",
	[
		(4, 8, 2),  # small batch, small latent, few classes
		(8, 16, 4),  # medium batch, typical latent size, more classes
	],
)
def test_contrastive_loss_random_data(
	batch_size: int, latent_dim: int, num_classes: int
) -> None:
	"""
	Tests whether the loss runs without error on random data
	and returns a finite scalar.
	"""
	h: Float[Tensor, "batch latent_dim"] = torch.randn(batch_size, latent_dim)
	classes: Int[Tensor, " batch"] = torch.randint(0, num_classes, (batch_size,))
	loss_val = contrastive_loss(h, classes, temperature=0.07)

	assert isinstance(loss_val, Tensor), "Loss must be a torch.Tensor"
	assert loss_val.dim() == 0, "Loss must be a scalar (0-dim tensor)"
	assert torch.isfinite(loss_val), "Loss returned NaN or Inf"


def test_contrastive_loss_same_class() -> None:
	"""
	Checks behavior when all samples belong to the same class.
	The loss should still be computable and typically yield a negative log-likelihood
	that is finite.
	"""
	batch_size: int = 4
	latent_dim: int = 8
	h: Float[Tensor, "batch latent_dim"] = torch.randn(batch_size, latent_dim)
	# All samples in the same class
	classes: Int[Tensor, " batch"] = torch.zeros(batch_size, dtype=torch.int32)

	loss_val = contrastive_loss(h, classes, temperature=0.07)
	assert torch.isfinite(loss_val), (
		"Loss returned NaN or Inf when all samples share a class"
	)


def test_contrastive_loss_distinct_classes() -> None:
	"""
	Checks behavior when each sample is in its own class (no positives).
	By default, we warn or handle the case with zero contribution for each sample.
	"""
	batch_size: int = 4
	latent_dim: int = 8
	h: Float[Tensor, "batch latent_dim"] = torch.randn(batch_size, latent_dim)
	# Each sample in a unique class
	classes: Int[Tensor, " batch"] = torch.arange(batch_size, dtype=torch.int32)

	# We expect a warning about "No positive pairs found in batch"
	with pytest.warns(UserWarning, match="No positive pairs found in batch"):
		loss_val = contrastive_loss(h, classes, temperature=0.07)
	# The function might return 0, or some finite value.
	assert torch.isfinite(loss_val), (
		"Loss returned NaN or Inf with all distinct classes"
	)


def test_contrastive_loss_partial_positives() -> None:
	"""
	Mixed scenario where some samples share classes and some do not.
	Verifies that the loss is finite.
	"""
	# For instance, classes = [0,0,1,2]
	# => two positives in the first pair, and singletons otherwise
	h: Float[Tensor, "batch latent_dim"] = torch.tensor(
		[[1.0, 0.0], [0.99, 0.01], [-1.0, 2.0], [2.0, -3.0]]
	)
	classes: Int[Tensor, " batch"] = torch.tensor([0, 0, 1, 2])

	loss_val = contrastive_loss(h, classes, temperature=0.07)
	assert torch.isfinite(loss_val), "Loss returned NaN or Inf in a mixed scenario"
	# Optionally check that the loss is > 0 or something else:
	assert loss_val > 0, "Loss should be positive if there are meaningful negatives"


def test_contrastive_loss_invariance_to_scale() -> None:
	"""
	Tests that scaling the input embeddings by a constant
	does not change the final loss, since embeddings are normalized internally.
	"""
	batch_size: int = 4
	latent_dim: int = 8
	h: Float[Tensor, "batch latent_dim"] = torch.randn(batch_size, latent_dim)
	classes: Int[Tensor, " batch"] = torch.tensor([0, 0, 1, 1])

	loss_normal = contrastive_loss(h, classes, temperature=0.07)
	loss_scaled = contrastive_loss(10.0 * h, classes, temperature=0.07)

	# They should be very close because h is normalized before computing similarities
	assert torch.allclose(loss_normal, loss_scaled, atol=1e-6), (
		"Loss should be invariant to global scaling of embeddings."
	)
