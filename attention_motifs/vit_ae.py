from typing import Tuple, Dict, Any, Type

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

from attention_motifs.train import convert_tril_rowstoch


@serializable_dataclass(kw_only=True)
class VitAEConfig(SerializableDataclass):
	"""Configuration for a Vision Transformer Autoencoder

	# Parameters:
	 - `latent_dim : int`
	    Dimension of the final latent space (for contrastive usage).
	 - `in_channels : int`
	    Number of input channels.
	 - `image_size : int`
	    Assumes square images of size (image_size x image_size).
	 - `patch_size : int`
	    Size of each patch (image is split into patches).
	 - `embed_dim : int`
	    Dimension of the patch embeddings.
	 - `num_heads : int`
	    Number of attention heads.
	 - `mlp_dim : int`
	    MLP dimension inside the Transformer blocks.
	 - `encoder_depth : int`
	    Number of Transformer blocks for the encoder.
	 - `decoder_depth : int`
	    Number of Transformer blocks for the decoder.
	 - `contrast_temperature : float`
	    Temperature for contrastive loss.
	 - `recon_weight : float`
	    Weight on reconstruction loss.
	 - `contrast_weight : float`
	    Weight on contrastive loss.
	 - `optimizer : Type[torch.optim.Optimizer]`
	    Optimizer class.
	 - `learning_rate : float`
	    Initial learning rate.
	 - `lr_scheduler : Type[torch.optim.lr_scheduler._LRScheduler]`
	    LR scheduler class.
	 - `lr_scheduler_kwargs : Dict[str, Any]`
	    Keyword arguments for the LR scheduler.
	 - `num_epochs : int`
	    Number of training epochs.
	"""

	latent_dim: int
	in_channels: int = serializable_field(default=1)
	image_size: int = serializable_field(default=28)
	patch_size: int = serializable_field(default=4)

	embed_dim: int = serializable_field(default=64)
	num_heads: int = serializable_field(default=8)
	mlp_dim: int = serializable_field(default=128)
	encoder_depth: int = serializable_field(default=4)
	decoder_depth: int = serializable_field(default=4)

	contrast_temperature: float = serializable_field(default=0.07)
	recon_weight: float = serializable_field(default=1.0)
	contrast_weight: float = serializable_field(default=1.0)

	optimizer: Type[torch.optim.Optimizer] = serializable_field(
		default=torch.optim.Adam,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(torch.optim, x),
	)

	learning_rate: float = serializable_field(default=1e-4)
	lr_scheduler: Type[torch.optim.lr_scheduler._LRScheduler] = serializable_field(
		default=torch.optim.lr_scheduler.ReduceLROnPlateau,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(torch.optim.lr_scheduler, x),
	)
	lr_scheduler_kwargs: Dict[str, Any] = serializable_field(
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

	num_epochs: int = serializable_field(default=5)

	def __post_init__(self) -> None:
		"""Checks and validations after initialization."""
		assert self.latent_dim > 0, "latent_dim must be positive"
		assert self.in_channels > 0, "in_channels must be positive"
		assert self.image_size >= self.patch_size, "image_size must be >= patch_size"
		assert self.embed_dim > 0, "embed_dim must be positive"
		assert self.num_heads > 0, "num_heads must be positive"
		assert self.mlp_dim > 0, "mlp_dim must be positive"
		assert self.encoder_depth > 0, "encoder_depth must be positive"
		assert self.decoder_depth > 0, "decoder_depth must be positive"

	def get_optim_and_lrs(
		self,
		model: "VitAE",
	) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
		"""Utility to create optimizer and learning rate scheduler from config."""
		optimizer: torch.optim.Optimizer = self.optimizer(
			model.parameters(),
			lr=self.learning_rate,
		)
		lr_scheduler: torch.optim.lr_scheduler._LRScheduler = self.lr_scheduler(
			optimizer,
			**self.lr_scheduler_kwargs,
		)
		return optimizer, lr_scheduler


class PatchEmbed(nn.Module):
	"""2D Patch Embedding with linear projection.

	# Parameters:
	 - `in_channels : int`
	    Number of input channels
	 - `embed_dim : int`
	    Dimension of embedded patch
	 - `patch_size : int`
	    The patch size (square patches)
	 - `img_size : int`
	    The (square) input image size
	"""

	def __init__(
		self,
		in_channels: int,
		embed_dim: int,
		patch_size: int,
		img_size: int,
	) -> None:
		super().__init__()
		self.in_channels: int = in_channels
		self.embed_dim: int = embed_dim
		self.patch_size: int = patch_size
		self.img_size: int = img_size

		assert img_size % patch_size == 0, (
			f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
		)

		self.num_patches: int = (img_size // patch_size) * (img_size // patch_size)

		self.proj: nn.Conv2d = nn.Conv2d(
			in_channels=self.in_channels,
			out_channels=self.embed_dim,
			kernel_size=self.patch_size,
			stride=self.patch_size,
		)

	def forward(
		self,
		x: Float[Tensor, "batch in_channels H W"],
	) -> Float[Tensor, "batch num_patches embed_dim"]:
		"""Applies patch embedding to input images."""
		# batch_size: int = x.shape[0]
		x_proj: Float[Tensor, "batch embed_dim patchH patchW"] = self.proj(x)
		x_proj = x_proj.flatten(2)  # => (B, embed_dim, num_patches)
		x_proj = x_proj.transpose(1, 2)  # => (B, num_patches, embed_dim)
		return x_proj


class TransformerEncoderBlock(nn.Module):
	"""A Transformer Encoder Block (pre-LayerNorm)."""

	def __init__(
		self,
		embed_dim: int,
		num_heads: int,
		mlp_dim: int,
		drop: float = 0.0,
		attn_drop: float = 0.0,
	) -> None:
		super().__init__()
		self.norm1: nn.LayerNorm = nn.LayerNorm(embed_dim)
		self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
			embed_dim=embed_dim,
			num_heads=num_heads,
			dropout=attn_drop,
			batch_first=True,
		)
		self.drop_attn: nn.Dropout = nn.Dropout(drop)

		self.norm2: nn.LayerNorm = nn.LayerNorm(embed_dim)
		self.mlp: nn.Sequential = nn.Sequential(
			nn.Linear(embed_dim, mlp_dim),
			nn.GELU(),
			nn.Linear(mlp_dim, embed_dim),
		)
		self.drop_mlp: nn.Dropout = nn.Dropout(drop)

	def forward(
		self, x: Float[Tensor, "batch seq_len embed_dim"]
	) -> Float[Tensor, "batch seq_len embed_dim"]:
		"""Forward pass of a single Transformer encoder block."""
		h: Float[Tensor, "batch seq_len embed_dim"] = self.norm1(x)
		attn_out, _ = self.attn(h, h, h)
		x = x + self.drop_attn(attn_out)

		h = self.norm2(x)
		h = self.mlp(h)
		x = x + self.drop_mlp(h)
		return x


class TransformerDecoderBlock(nn.Module):
	"""A Transformer Decoder Block (no cross-attention, pre-LayerNorm)."""

	def __init__(
		self,
		embed_dim: int,
		num_heads: int,
		mlp_dim: int,
		drop: float = 0.0,
		attn_drop: float = 0.0,
	) -> None:
		super().__init__()
		self.norm1: nn.LayerNorm = nn.LayerNorm(embed_dim)
		self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
			embed_dim=embed_dim,
			num_heads=num_heads,
			dropout=attn_drop,
			batch_first=True,
		)
		self.drop_attn: nn.Dropout = nn.Dropout(drop)

		self.norm2: nn.LayerNorm = nn.LayerNorm(embed_dim)
		self.mlp: nn.Sequential = nn.Sequential(
			nn.Linear(embed_dim, mlp_dim),
			nn.GELU(),
			nn.Linear(mlp_dim, embed_dim),
		)
		self.drop_mlp: nn.Dropout = nn.Dropout(drop)

	def forward(
		self, x: Float[Tensor, "batch seq_len embed_dim"]
	) -> Float[Tensor, "batch seq_len embed_dim"]:
		"""Forward pass of a single Transformer decoder block."""
		h: Float[Tensor, "batch seq_len embed_dim"] = self.norm1(x)
		attn_out, _ = self.attn(h, h, h)
		x = x + self.drop_attn(attn_out)

		h = self.norm2(x)
		h = self.mlp(h)
		x = x + self.drop_mlp(h)
		return x


@set_config_class(VitAEConfig)
class VisionTransformerEncoder(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Encoder.
	Splits the input into patches, uses a stack of
	Transformer blocks, then outputs a `latent_dim` vector
	by average pooling the final patch embeddings.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		super().__init__(config)
		self.config: VitAEConfig = config

		# Patch embedding
		self.patch_embed: PatchEmbed = PatchEmbed(
			in_channels=config.in_channels,
			embed_dim=config.embed_dim,
			patch_size=config.patch_size,
			img_size=config.image_size,
		)
		self.num_patches: int = self.patch_embed.num_patches

		# Learnable position embeddings
		self.pos_embed: nn.Parameter = nn.Parameter(
			torch.zeros(1, self.num_patches, config.embed_dim),
			requires_grad=True,
		)
		nn.init.normal_(self.pos_embed, std=0.02)

		# Transformer encoder blocks
		self.blocks: nn.ModuleList = nn.ModuleList(
			[
				TransformerEncoderBlock(
					embed_dim=config.embed_dim,
					num_heads=config.num_heads,
					mlp_dim=config.mlp_dim,
					drop=0.0,
					attn_drop=0.0,
				)
				for _ in range(config.encoder_depth)
			]
		)
		self.norm: nn.LayerNorm = nn.LayerNorm(config.embed_dim)

		# Final projection to latent space
		self.to_latent: nn.Linear = nn.Linear(config.embed_dim, config.latent_dim)

	def forward(
		self, x: Float[Tensor, "batch in_channels H W"]
	) -> Float[Tensor, "batch latent_dim"]:
		"""Forward pass of the ViT encoder."""
		x_patches: Float[Tensor, "batch num_patches embed_dim"] = self.patch_embed(x)
		x_patches = x_patches + self.pos_embed  # shape => (B, num_patches, embed_dim)

		# Pass through encoder blocks
		for blk in self.blocks:
			x_patches = blk(x_patches)

		# Final layer norm
		x_patches = self.norm(x_patches)

		# Mean pool -> single vector per sample
		x_mean: Float[Tensor, "batch embed_dim"] = x_patches.mean(dim=1)
		# Project to latent_dim
		z: Float[Tensor, "batch latent_dim"] = self.to_latent(x_mean)
		return z


@set_config_class(VitAEConfig)
class VisionTransformerDecoder(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Decoder.
	Takes a (batch, latent_dim) vector, replicates it
	across the same number of patches, passes it through
	decoder Transformer blocks, then projects back to
	patches and unpatchifies to the original image shape.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		super().__init__(config)
		self.config: VitAEConfig = config

		self.num_patches: int = (config.image_size // config.patch_size) ** 2

		# Map latent -> embed_dim
		self.from_latent: nn.Linear = nn.Linear(config.latent_dim, config.embed_dim)

		# Decoder position embeddings
		self.pos_embed_dec: nn.Parameter = nn.Parameter(
			torch.zeros(1, self.num_patches, config.embed_dim),
			requires_grad=True,
		)
		nn.init.normal_(self.pos_embed_dec, std=0.02)

		# Decoder transformer blocks
		self.blocks: nn.ModuleList = nn.ModuleList(
			[
				TransformerDecoderBlock(
					embed_dim=config.embed_dim,
					num_heads=config.num_heads,
					mlp_dim=config.mlp_dim,
					drop=0.0,
					attn_drop=0.0,
				)
				for _ in range(config.decoder_depth)
			]
		)
		self.norm: nn.LayerNorm = nn.LayerNorm(config.embed_dim)

		# Final projection from embed_dim -> patch pixels
		self.patch_dim: int = config.in_channels * config.patch_size * config.patch_size
		self.head: nn.Linear = nn.Linear(config.embed_dim, self.patch_dim)

	def forward(
		self, z: Float[Tensor, "batch latent_dim"]
	) -> Float[Tensor, "batch in_channels H W"]:
		"""Forward pass of the ViT decoder."""
		batch_size: int = z.shape[0]

		# (1) Map latent -> embed_dim
		latent_embed: Float[Tensor, "batch embed_dim"] = self.from_latent(z)
		# (2) Expand across all patches => (B, num_patches, embed_dim)
		latent_embed = latent_embed.unsqueeze(1).expand(
			batch_size, self.num_patches, -1
		)
		# (3) Add decoder pos embeddings
		latent_embed = latent_embed + self.pos_embed_dec

		# (4) Pass through decoder blocks
		for blk in self.blocks:
			latent_embed = blk(latent_embed)

		# (5) Final norm
		latent_embed = self.norm(latent_embed)

		# (6) Project to patch pixels => (B, num_patches, patch_dim)
		patches: Float[Tensor, "batch num_patches patch_dim"] = self.head(latent_embed)

		# (7) Unpatchify
		# => (B, num_patches, in_channels, patch_size^2)
		patches = patches.view(
			batch_size,
			self.num_patches,
			self.config.in_channels,
			self.config.patch_size * self.config.patch_size,
		)
		# => (B, num_patches, in_channels, patch_size, patch_size)
		patches = patches.view(
			batch_size,
			self.num_patches,
			self.config.in_channels,
			self.config.patch_size,
			self.config.patch_size,
		)
		patches_per_dim: int = self.config.image_size // self.config.patch_size
		# => (B, p_per_dim, p_per_dim, in_ch, p_size, p_size)
		patches = patches.reshape(
			batch_size,
			patches_per_dim,
			patches_per_dim,
			self.config.in_channels,
			self.config.patch_size,
			self.config.patch_size,
		)
		patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
		x_recon: Float[Tensor, "batch in_channels H W"] = patches.view(
			batch_size,
			self.config.in_channels,
			self.config.image_size,
			self.config.image_size,
		)

		x_recon = convert_tril_rowstoch(x_recon)
		return x_recon


@set_config_class(VitAEConfig)
class VitAE(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Autoencoder with a
	separate encoder and decoder. Produces
	a latent vector of shape (batch, latent_dim).
	"""

	def __init__(self, config: VitAEConfig) -> None:
		super().__init__(config)
		self.config: VitAEConfig = config
		self.encoder: VisionTransformerEncoder = VisionTransformerEncoder(config)
		self.decoder: VisionTransformerDecoder = VisionTransformerDecoder(config)

	def forward(
		self,
		x: Float[Tensor, "batch in_channels H W"],
	) -> Tuple[
		Float[Tensor, "batch in_channels H W"], Float[Tensor, "batch latent_dim"]
	]:
		"""Forward pass of VitAE."""
		z: Float[Tensor, "batch latent_dim"] = self.encoder(x)
		x_recon: Float[Tensor, "batch in_channels H W"] = self.decoder(z)
		return x_recon, z
