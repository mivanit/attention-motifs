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
