from pathlib import Path
import warnings
import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool


def convert_tril_rowstoch(
	x: Float[Tensor, "batch c H W"],
) -> Float[Tensor, "batch c H W"]:
	"converts a square matrix to a row-stochastic lower triangular one"
	x = x + torch.triu(torch.ones_like(x) * float("-inf"), diagonal=1)
	x = F.softmax(x, dim=-1)
	return x


def contrastive_loss(
	h: Float[Tensor, "batch latent_dim"],
	classes: Int[Tensor, " batch"],
	temperature: float = 0.07,
) -> Float[Tensor, ""]:
	"""Compute a supervised contrastive loss.

	Pushes samples of the same class together and pushes
	samples from different classes apart.

	# Parameters:
	 - `h : Float[Tensor, "batch latent_dim"]`
	    latent embeddings
	 - `classes : Int[Tensor, " batch"]`
	    class labels (integer) for each sample in the batch
	 - `temperature : float`
	    temperature for scaling similarities
	    (defaults to 0.07)

	# Returns:
	 - `Float[Tensor, ""]`
	    the scalar contrastive loss

	# Usage:
	```python
	>>> batch_size = 8
	>>> latent_dim = 16
	>>> h = torch.randn(batch_size, latent_dim)
	>>> classes = torch.randint(0, 3, (batch_size,))
	>>> loss_val = contrastive_loss(h, classes, temperature=0.07)
	>>> print(loss_val)
	```

	# Raises:
	 - `ValueError` : if all samples belong to distinct classes (no positives)
	"""

	batch_size: int = h.shape[0]
	# Normalize the embeddings
	h_norm: Float[Tensor, "batch latent_dim"] = F.normalize(h, dim=1)

	# Compute pairwise cosine similarities
	sim: Float[Tensor, "batch batch"] = h_norm @ h_norm.T

	# Scale the similarities by the temperature
	sim_scaled: Float[Tensor, "batch batch"] = sim / temperature

	# Create a mask for all positives: same class and not self
	positive_mask: Bool[Tensor, "batch batch"] = (
		classes.unsqueeze(1) == classes.unsqueeze(0)
	) & (~torch.eye(batch_size, dtype=torch.bool, device=h.device))

	# Ensure there's at least one positive for each sample
	# (if there's a class with exactly 1 sample in the batch, that sample has no positives)
	# We'll allow those samples to have zero contribution, though sometimes you'd skip them or handle separately.
	if positive_mask.sum() == 0:
		warnings.warn("No positive pairs found in batch")

	# Exponentiate scaled similarities
	exp_sim: Float[Tensor, "batch batch"] = torch.exp(sim_scaled)

	# For each anchor i, we exclude itself from the denominator
	# so we zero out the diagonal
	exp_sim_masked: Float[Tensor, "batch batch"] = exp_sim * (
		~torch.eye(batch_size, device=h.device, dtype=torch.bool)
	)

	# Sum over all (masked) exponentiated similarities for the denominator
	denom: Float[Tensor, " batch"] = exp_sim_masked.sum(dim=1)

	# log_prob[i, j] = sim[i,j]/temp - log( sum_{k != i}(exp(sim[i,k]/temp)) )
	log_prob: Float[Tensor, "batch batch"] = (sim_scaled) - torch.log(denom).unsqueeze(
		1
	)

	# For each anchor i, we only want the log_probs for positives
	# We'll sum over those positives and then divide by the number of positives
	positive_log_prob: Float[Tensor, " batch"] = (
		(log_prob * positive_mask).sum(dim=1)
		/ (positive_mask.sum(dim=1) + 1e-8)  # add epsilon to avoid div by zero
	)

	# Our loss is the negative mean of these average positive log probs
	loss: Float[Tensor, ""] = -positive_log_prob.mean()

	return loss


"""
def evaluation_step(model: AttnAE) -> dict[str, float]:
	"Evaluate model on validation set"
	if VAL_LOADER is None:
		return {}

	model.eval()
	val_metrics: dict[str, float] = {
		"val/loss": 0.0,
		"val/recon_loss": 0.0,
		"val/contrast_loss": 0.0,
	}

	with torch.no_grad():
		for patterns, metadata in VAL_LOADER:
			patterns = patterns.to(DEVICE).to(torch.float32).unsqueeze(1)
			patterns_recon, embeddings = model(patterns)

			# reconstruction loss
			recon_loss = F.mse_loss(patterns_recon, patterns)

			# contrastive loss using all pairs in batch
			# compute "classes" for contrastive loss
			# classes is a tensor of the same shape as the batch, where each element is an integer
			classes: Int[torch.Tensor, " batch"] = (
				AttentionPatternMetadata.contrastive_classes(metadata).to(DEVICE)
			)

			# compute contrastive loss
			contrast_loss = contrastive_loss(
				embeddings, classes, temperature=model.config.contrast_temperature
			)

			# combined loss and backward pass
			total_loss = (
				model.config.recon_weight * recon_loss
				+ model.config.contrast_weight * contrast_loss
			)
			val_metrics["val/loss"] += total_loss.item()
			val_metrics["val/recon_loss"] += recon_loss.item()
			val_metrics["val/contrast_loss"] += contrast_loss.item()

	for k in val_metrics:
		val_metrics[k] /= len(VAL_LOADER)

	model.train()
	return val_metrics
"""

