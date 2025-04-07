import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from zanj.torchutil import ConfiguredModel, set_config_class

from attention_motifs.autoencoder.train_util import convert_tril_rowstoch


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
		x_recon = convert_tril_rowstoch(x_recon)

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
