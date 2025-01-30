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




def train(
	model: VitAE,
	train_loader: torch.utils.data.DataLoader,
	val_loader: torch.utils.data.DataLoader | None = None,
	num_epochs: int = 100,
	learning_rate: float = 1e-3,
	recon_weight: float = 1.0,
	contrast_weight: float = 1.0,
	project_name: str = "contrastive-ae",
	checkpoint_interval: str = "1/10 run",
	eval_interval: str = "1k samples",
	device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[VitAE, LocalLogger]:
	"""Train a contrastive autoencoder

	# Parameters:
	 - `model: ContrastiveAutoencoder`
	    Model to train
	 - `train_loader: torch.utils.data.DataLoader`
	    Training data loader returning (tensor, index_tuple) pairs
	 - `val_loader: torch.utils.data.DataLoader | None`
	    Optional validation loader
	 - `num_epochs: int`
	    Number of epochs to train
	 - `learning_rate: float`
	    Learning rate for Adam optimizer
	 - `recon_weight: float`
	    Weight for reconstruction loss
	 - `contrast_weight: float`
	    Weight for contrastive loss
	 - `project_name: str`
	    Name for logging
	    (default: "contrastive-ae")
	 - `checkpoint_interval: str`
	    When to save checkpoints (trnbl format)
	 - `eval_interval: str`
	    How often to evaluate (trnbl format)
	 - `device: torch.device`
	    Device to train on

	# Returns:
	 - `ContrastiveAutoencoder` : Trained model
	 - `LocalLogger` : Logger with training history
	"""
	model = model.to(device)
	optimizer: torch.optim.Optimizer = model.config.optimizer(
		model.parameters(),
		lr=learning_rate,
	)

	# setup logger
	logger: LocalLogger = LocalLogger(
		project=project_name,
		metric_names=[
			"train/loss",
			"train/recon_loss",
			"train/contrast_loss",
			"val/loss",
			"val/recon_loss",
			"val/contrast_loss",
		],
		train_config=dict(
			model_config=model.zanj_model_config.serialize(),
			learning_rate=learning_rate,
			recon_weight=recon_weight,
			contrast_weight=contrast_weight,
		),
	)

	def evaluation_step(model: VitAE) -> dict[str, float]:
		"""Evaluate model on validation set"""
		if val_loader is None:
			return {}

		model.eval()
		val_metrics = {"val/loss": 0.0, "val/recon_loss": 0.0, "val/contrast_loss": 0.0}

		with torch.no_grad():
			for batch_idx, (x, index_tuple) in enumerate(val_loader):
				x = x.to(device)
				index_tuple = tuple(i.to(device) for i in index_tuple)

				x_recon, z = model(x)
				recon_loss = F.mse_loss(x_recon, x)

				batch_size = x.size(0)
				z1 = z.repeat_interleave(batch_size, dim=0)
				z2 = z.repeat(batch_size, 1)
				idx1 = tuple(i.repeat_interleave(batch_size) for i in index_tuple)
				idx2 = tuple(i.repeat(batch_size) for i in index_tuple)
				contrast_loss = model.contrastive_loss(z1, z2, idx1, idx2)

				total_loss = recon_weight * recon_loss + contrast_weight * contrast_loss

				val_metrics["val/loss"] += total_loss.item()
				val_metrics["val/recon_loss"] += recon_loss.item()
				val_metrics["val/contrast_loss"] += contrast_loss.item()

		for k in val_metrics:
			val_metrics[k] /= len(val_loader)

		model.train()
		return val_metrics

	with TrainingManager(
		model=model,
		logger=logger,
		evals={
			eval_interval: evaluation_step,
		}.items(),
		checkpoint_interval=checkpoint_interval,
	) as tr:
		for epoch in tr.epoch_loop(range(num_epochs)):
			for x, index_tuple in tr.batch_loop(train_loader):
				x = x.to(device)
				index_tuple = tuple(i.to(device) for i in index_tuple)

				optimizer.zero_grad()
				x_recon, z = model(x)

				# reconstruction loss
				recon_loss = F.mse_loss(x_recon, x)

				# contrastive loss using all pairs in batch
				batch_size = x.size(0)
				z1 = z.repeat_interleave(batch_size, dim=0)
				z2 = z.repeat(batch_size, 1)
				idx1 = tuple(i.repeat_interleave(batch_size) for i in index_tuple)
				idx2 = tuple(i.repeat(batch_size) for i in index_tuple)
				contrast_loss = model.contrastive_loss(z1, z2, idx1, idx2)

				# combined loss and backward pass
				total_loss = recon_weight * recon_loss + contrast_weight * contrast_loss
				total_loss.backward()
				optimizer.step()

				# log metrics
				tr.batch_update(
					samples=len(x),
					**{
						"train/loss": total_loss.item(),
						"train/recon_loss": recon_loss.item(),
						"train/contrast_loss": contrast_loss.item(),
					},
				)

	return model, logger
