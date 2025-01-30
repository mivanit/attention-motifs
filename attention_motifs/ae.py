import warnings
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from zanj.torchutil import ConfiguredModel, set_config_class
from trnbl import TrainingManager
from trnbl.loggers.local import LocalLogger


@serializable_dataclass
class Conv2DConfig(SerializableDataclass):
	channels: int
	kernel_size: int = serializable_field(default=3)
	stride: int = serializable_field(default=1)
	padding: int = serializable_field(default=1)

	def create(self, in_channels: int) -> nn.Conv2d:
		return nn.Conv2d(
			in_channels=in_channels,
			out_channels=self.channels,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding,
		)

	def create_decoder(self, out_channels: int) -> nn.Conv2d:
		return nn.ConvTranspose2d(
			in_channels=self.channels,
			out_channels=out_channels,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=self.padding,
		)


@serializable_dataclass(kw_only=True)
class AttnAEConfig(SerializableDataclass):
	"""Configuration for square matrix contrastive autoencoder

	# Parameters:
	 - `latent_dim : int`
	    Dimension of latent space
	 - `encoder_channels : Sequence[int]`
	    Number of channels in each encoder layer
	 - `kernel_size : int`
	    Kernel size for conv layers
	 - `margin : float`
	    Margin for contrastive loss
	 - `activation : type[nn.Module]`
	    activation function to use. will define `act_fn = activation()`
	 - `pooling : str`
	    One of 'max' or 'avg'
	"""

	# architecture
	latent_dim: int
	in_channels: int = serializable_field(default=1)
	conv_encoder: list[Conv2DConfig] = serializable_field(
		default_factory=lambda: [
			Conv2DConfig(channels=16),
			Conv2DConfig(channels=64),
			Conv2DConfig(channels=64),
			Conv2DConfig(channels=64),
			Conv2DConfig(channels=128),
		],
		serialization_fn=lambda x: [c.serialize() for c in x],
		deserialize_fn=lambda x: [Conv2DConfig.load(c) for c in x],
	)

	mlp_prepool: list[int] = serializable_field(default_factory=lambda: [128, 128])
	mlp_postpool: list[int] = serializable_field(default_factory=lambda: [128, 128])

	activation: type[nn.Module] = serializable_field(
		default=nn.ReLU,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(nn, x),
	)

	# loss
	contrast_temperature: float = serializable_field(default=0.07)
	recon_weight: float = serializable_field(default=1.0)
	contrast_weight: float = serializable_field(default=1.0)

	# optimizer
	optimizer: type[torch.optim.Optimizer] = serializable_field(
		default=torch.optim.Adam,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(torch.optim, x),
	)

	# lr scheduler
	learning_rate: float = serializable_field(default=1e-4)
	lr_scheduler: type[torch.optim.lr_scheduler._LRScheduler] = serializable_field(
		default=torch.optim.lr_scheduler.ReduceLROnPlateau,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(torch.optim.lr_scheduler, x),
	)
	lr_scheduler_kwargs: dict = serializable_field(
		default_factory=lambda: dict(
			mode="min",
			factor=0.1,
			patience=10,
			threshold=1e-4,
			threshold_mode="rel",
			cooldown=0,
			min_lr=1e-6,
			eps=1e-8,
		)
	)

	# epochs
	num_epochs: int = serializable_field(default=5)

	def __post_init__(self):
		assert all(c.channels > 0 for c in self.conv_encoder)
		assert all(d > 0 for d in self.mlp_prepool)
		assert all(d > 0 for d in self.mlp_postpool)

	def get_optim_and_lrs(
		self,
		model: "AttnAE",
	) -> tuple[
		torch.optim.Optimizer,
		torch.optim.lr_scheduler._LRScheduler,
	]:
		optimizer: torch.optim.Optimizer = self.optimizer(
			model.parameters(),
			lr=self.learning_rate,
		)
		lr_scheduler: torch.optim.lr_scheduler._LRScheduler = self.lr_scheduler(
			optimizer,
			**self.lr_scheduler_kwargs,
		)
		return optimizer, lr_scheduler


@set_config_class(AttnAEConfig)
class Encoder(ConfiguredModel[AttnAEConfig]):
	def __init__(self, config: AttnAEConfig):
		super().__init__(config)
		self.config: AttnAEConfig = config

		# Convolutional encoder
		in_ch: int = config.in_channels
		conv_layers: list[nn.Module] = []

		for conv_cfg in config.conv_encoder:
			conv_layers.append(conv_cfg.create(in_ch))
			conv_layers.append(config.activation())
			in_ch = conv_cfg.channels

		self.conv: nn.Module = nn.Sequential(*conv_layers)

		# linear map on each pixel
		linear_layers_prepool: list[nn.Module] = []
		for out_dim in config.mlp_prepool:
			linear_layers_prepool.append(nn.Linear(in_ch, out_dim))
			linear_layers_prepool.append(config.activation())
			in_ch = out_dim

		self.linear_prepool: nn.Module = nn.Sequential(*linear_layers_prepool)

		# linear map on pooled features
		linear_layers_postpool: list[nn.Module] = []
		for out_dim in config.mlp_postpool:
			linear_layers_postpool.append(nn.Linear(in_ch, out_dim))
			linear_layers_postpool.append(config.activation())
			in_ch = out_dim

		self.linear_postpool: nn.Module = nn.Sequential(*linear_layers_postpool)

	def forward(
		self, x: Float[Tensor, "batch 1 n_ctx n_ctx"]
	) -> Float[Tensor, "batch latent_dim"]:
		# conv layers
		h: Float[Tensor, "batch channels n_ctx n_ctx"] = self.conv(x)
		# apply linear layers to each pixel
		# TODO: add pos embeds?
		h_reshape = h.flatten(2).reshape(h.size(0), -1, h.size(1))
		h = self.linear_prepool(h_reshape)
		# mean pool over pixels
		h = h.mean(dim=-2)
		# apply linear layers to pooled features
		h = self.linear_postpool(h)
		return h


@set_config_class(AttnAEConfig)
class Decoder(ConfiguredModel[AttnAEConfig]):
	"""Decoder stage of the AttnAE architecture

	This mirrors the `Encoder` by:
	  1. Taking a latent vector of shape (batch, latent_dim)
	  2. Passing it through the inverse MLP layers
	  3. "Un-pooling" or broadcasting back to a spatial grid
	  4. Passing the resulting feature maps through transposed convolution layers
	  5. Producing a reconstructed image of shape (batch, in_channels, H, W)

	# Parameters:
	 - `config : AttnAEConfig`
	    The model configuration

	# Usage:
	```python
	>>> decoder = Decoder(config)
	>>> z = torch.randn(16, config.latent_dim)
	>>> x_recon = decoder(z)
	>>> x_recon.shape
	torch.Size([16, config.in_channels, H, W])
	```
	"""

	def __init__(self, config: AttnAEConfig):
		super().__init__(config)
		self.config: AttnAEConfig = config

		# Inverse of post-pool MLP
		postunpool_layers: list[nn.Module] = []
		in_dim: int = config.latent_dim
		# We'll reverse the mlp_postpool layers used in the encoder
		for out_dim in reversed(config.mlp_postpool):
			postunpool_layers.append(nn.Linear(in_dim, out_dim))
			postunpool_layers.append(config.activation())
			in_dim = out_dim

		self.linear_postunpool: nn.Module = nn.Sequential(*postunpool_layers)

		# Inverse of pre-pool MLP
		preunpool_layers: list[nn.Module] = []
		for out_dim in reversed(config.mlp_prepool):
			preunpool_layers.append(nn.Linear(in_dim, out_dim))
			preunpool_layers.append(config.activation())
			in_dim = out_dim

		self.linear_preunpool: nn.Module = nn.Sequential(*preunpool_layers)

		# TODO: first conv doesn't correctly read last preunpool layer size

		# Transposed convolution layers
		rev_conv_cfgs = list(reversed(config.conv_encoder))
		conv_layers: list[nn.Module] = []
		for i, conv_cfg in enumerate(rev_conv_cfgs):
			# Decide what the output channels of this transpose conv should be
			# If not at the last reversed conv, next out is rev_conv_cfgs[i+1].channels
			# Otherwise, decode to the original in_channels
			if i < len(rev_conv_cfgs) - 1:
				out_ch = rev_conv_cfgs[i + 1].channels
			else:
				out_ch = config.in_channels

			conv_layers.append(conv_cfg.create_decoder(out_ch))
			# add activation except perhaps after the final layer
			if i < len(rev_conv_cfgs) - 1:
				conv_layers.append(config.activation())

		self.conv: nn.Module = nn.Sequential(*conv_layers)

	@classmethod
	def convert_tril_rowstoch(x: Float[Tensor, "batch in_channels n_ctx n_ctx"]) -> Float[Tensor, "batch in_channels n_ctx n_ctx"]:

		# set the upper triangle to -inf
		x += torch.triu(torch.ones_like(x) * float("-inf"), diagonal=0)
		# apply softmax
		x = F.softmax(x, dim=-2)
		return x

	def forward(
		self,
		z: Float[Tensor, "batch latent_dim"],
		n_ctx: int,
	) -> Float[Tensor, "batch in_channels n_ctx n_ctx"]:
		"""Forward pass of the Decoder"""
		# 1) Inverse of the post-pool MLP
		h: Float[Tensor, "batch mid_dim"] = self.linear_postunpool(z)  # (B, ?)
		# 2) Broadcast to spatial dimension
		h = h.unsqueeze(-1)
		h = h.expand(-1, -1, n_ctx * n_ctx)
		h = h.permute(0, 2, 1)
		# 3) Inverse of the pre-pool MLP => shape (B, channels, H*W)
		h = self.linear_preunpool(h)  # (B, channels, H*W)
		# reshape => (B, channels, H, W)
		h = h.permute(0, 2, 1)
		h = h.view(
			h.shape[0],
			h.shape[1],
			n_ctx,
			n_ctx,
		)

		# 4) Run transposed convolution => (B, in_channels, H, W)
		x_recon: Float[Tensor, "batch in_channels H W"] = self.conv(h)
		
		# 5) Convert to row-stochastic
		x_recon = self.convert_tril_rowstoch(x_recon)
		
		return x_recon


@set_config_class(AttnAEConfig)
class AttnAE(ConfiguredModel[AttnAEConfig]):
	config: AttnAEConfig

	def __init__(self, config: AttnAEConfig):
		super().__init__(config)
		self.config: AttnAEConfig = config

		self.encoder: Encoder = Encoder(config)
		self.decoder: Decoder = Decoder(config)

	def forward(
		self,
		x: Float[Tensor, "*batch n n"],
	) -> tuple[Float[Tensor, "*batch n n"], Float[Tensor, "batch latent_dim"]]:
		n_ctx: int = x.shape[-1]

		h: Float[Tensor, "batch latent_dim"] = self.encoder(x)
		x_recon: Float[Tensor, "batch 1 n n"] = self.decoder(h, n_ctx=n_ctx)
		return x_recon, h


def train(
	model: AttnAE,
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
) -> tuple[AttnAE, LocalLogger]:
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

	def evaluation_step(model: AttnAE) -> dict[str, float]:
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
