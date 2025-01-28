from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float, Int

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
	 - `decoder_channels : Sequence[int] | None`
	    Number of channels in each decoder layer. If None, mirrors encoder
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
		]
	)

	mlp_prepool: list[int] = serializable_field(default_factory=lambda: [128])
	mlp_postpool: list[int] = serializable_field(default_factory=lambda: [128, 64])

	activation: type[nn.Module] = serializable_field(
		default=nn.ReLU,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(nn, x),
	)

	# loss
	margin: float = serializable_field(default=1.0)

	# optimizer
	optimizer: type[torch.optim.Optimizer] = serializable_field(
		default=torch.optim.Adam,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(torch.optim, x),
	)

	@property
	def decoder_channels(self) -> Sequence[int]:
		return tuple(reversed(self.encoder_channels))

	def __post_init__(self):
		assert all(c.channels > 0 for c in self.encoder_channels)


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
		for out_dim in config.mlp_encoder:
			linear_layers_prepool.append(nn.Linear(in_ch, out_dim))
			linear_layers_prepool.append(config.activation())
			in_ch = out_dim

		self.linear_prepool: nn.Module = nn.Sequential(*linear_layers_prepool)

		# linear map on pooled features
		linear_layers_postpool: list[nn.Module] = []
		for out_dim in config.mlp_encoder:
			linear_layers_postpool.append(nn.Linear(in_ch, out_dim))
			linear_layers_postpool.append(config.activation())
			in_ch = out_dim

		self.linear_postpool: nn.Module = nn.Sequential(*linear_layers_postpool)

	def forward(
		self, x: Float[Tensor, "batch 1 n n"]
	) -> Float[Tensor, "batch latent_dim"]:
		# conv layers
		h: Float[Tensor, "batch channels n n"] = self.conv(x)
		# apply linear layers to each pixel
		h = self.linear_prepool(h.flatten(2))
		# mean pool over pixels
		h = h.mean(dim=2)
		# apply linear layers to pooled features
		h = self.linear_postpool(h)
		return h


@set_config_class(AttnAEConfig)
class Decoder(ConfiguredModel[AttnAEConfig]):
	def __init__(self, config: AttnAEConfig):
		super().__init__(config)
		self.config: AttnAEConfig = config


@set_config_class(AttnAEConfig)
class AttnAE(ConfiguredModel[AttnAEConfig]):
	def __init__(self, config: AttnAEConfig):
		super().__init__(config)
		self.config: AttnAEConfig = config

		self.encoder: Encoder = Encoder(config)
		self.decoder: Decoder = Decoder(config)

	def forward(
		self,
		x: Float[Tensor, "*batch n n"],
		output_with_channel_dim: bool = False,
	) -> Float[Tensor, "*batch n n"]:
		n_ctx: int = x.shape[-1]

		z: Float[Tensor, "batch latent_dim"] = self.encoder(x)
		x_recon: Float[Tensor, "batch 1 n n"] = self.decoder(z)
		return x_recon


@set_config_class(AttnAEConfig)
class AttnAE(ConfiguredModel[AttnAEConfig]):
	def __init__(self, config: AttnAEConfig):
		super().__init__(config)
		self.config: AttnAEConfig = config

		act_fn: nn.Module = config.activation()
		pool_fn: nn.Module = config.pooling()

		# Encoder stack with adaptive final pooling
		encoder_layers: list[nn.Module] = []
		in_ch: int = 1  # single channel input

		# Progressive downsampling while increasing channels
		for out_ch in config.encoder_channels:
			encoder_layers.extend(
				[
					nn.Conv2d(in_ch, out_ch, config.kernel_size, padding=1),
					act_fn(),
					pool_fn(2),
				]
			)
			in_ch = out_ch

		# Add adaptive pooling to get to fixed size before latent space
		encoder_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
		self.encoder_conv: nn.Module = nn.Sequential(*encoder_layers)

		# Linear projection to latent space
		self.final_conv_size: int = 4 * 4 * config.encoder_channels[-1]
		self.encoder_linear: nn.Module = nn.Linear(
			self.final_conv_size, config.latent_dim
		)

		# Decoder - starts from fixed size and uses interpolation
		self.decoder_linear: nn.Module = nn.Linear(
			config.latent_dim, self.final_conv_size
		)

		decoder_layers: list[nn.Module] = []
		in_ch = config.decoder_channels[0]

		# Start with reshaping layer
		self.decoder_reshape: tuple[int, int, int] = (in_ch, 4, 4)

		# Progressive upsampling while decreasing channels
		for out_ch in config.decoder_channels[1:]:
			decoder_layers.extend(
				[
					nn.ConvTranspose2d(in_ch, out_ch, config.kernel_size, padding=1),
					act_fn(),
					nn.Upsample(scale_factor=2),
				]
			)
			in_ch = out_ch

		# Final layer with adaptive interpolation
		decoder_layers.extend(
			[
				nn.ConvTranspose2d(in_ch, 1, config.kernel_size, padding=1),
				nn.Sigmoid(),
				nn.Upsample(
					mode="bilinear", align_corners=True
				),  # size set in forward pass
			]
		)

		self.decoder_conv: nn.ModuleList = nn.ModuleList(decoder_layers)

	def encode(
		self, x: Float[Tensor, "batch 1 n n"]
	) -> Float[Tensor, "batch latent_dim"]:
		h = self.encoder_conv(x)
		return self.encoder_linear(h.flatten(1))

	def decode(
		self,
		z: Float[Tensor, "batch latent_dim"],
		output_size: tuple[int, int],
	) -> Float[Tensor, "batch 1 n n"]:
		"""Decode latent vectors to reconstructions

		# Parameters:
		 - `z : Tensor`
		    Batch of latent vectors
		 - `output_size : tuple[int, int]`
		    Desired output size (height, width)
		"""
		h = self.decoder_linear(z)
		h = h.view(-1, *self.decoder_reshape)

		# All layers except last
		for layer in self.decoder_conv[:-1]:
			h = layer(h)

		# Final upsampling layer - set size
		if isinstance(self.decoder_conv[-1], nn.Upsample):
			h = self.decoder_conv[-1](h, size=output_size)
		else:
			h = self.decoder_conv[-1](h)

		return h

	def forward(
		self, x: Float[Tensor, "batch 1 n n"]
	) -> tuple[Float[Tensor, "batch 1 n n"], Float[Tensor, "batch latent_dim"]]:
		z: Float[Tensor, "batch latent_dim"] = self.encode(x)
		x_recon = self.decode(z, output_size=(x.shape[2], x.shape[3]))
		return x_recon, z

	def contrastive_loss(
		self,
		z1: Float[Tensor, "batch latent_dim"],
		z2: Float[Tensor, "batch latent_dim"],
		idx1: tuple[Int[Tensor, " batch"], ...],
		idx2: tuple[Int[Tensor, " batch"], ...],
	) -> Float[Tensor, ""]:
		"""Compute contrastive loss between pairs of embeddings"""
		distances = F.pairwise_distance(z1, z2)

		# Compare tuples of indices
		similar = torch.all(
			torch.stack([i1 == i2 for i1, i2 in zip(idx1, idx2)]), dim=0
		)

		similar_loss = distances.pow(2) * similar
		dissimilar_loss = torch.pow(
			torch.clamp(self.zanj_model_config.margin - distances, min=0.0), 2
		) * (~similar)

		return (similar_loss + dissimilar_loss).mean()


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
