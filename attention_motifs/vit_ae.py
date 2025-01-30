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
	"""Configuration for a Vision Transformer Autoencoder (square images of size n_ctx x n_ctx)

	# Parameters:
	 - `latent_dim : int`
	    Dimension of the final latent space (for contrastive usage).
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
	 - `num_epochs : int`
	    Number of training epochs.
	 - `optimizer : Type[torch.optim.Optimizer]`
	    Optimizer class.
	 - `learning_rate : float`
	    Initial learning rate.
	 - `lr_scheduler : Type[torch.optim.lr_scheduler._LRScheduler]`
	    LR scheduler class.
	 - `lr_scheduler_kwargs : Dict[str, Any]`
	    Keyword arguments for the LR scheduler.
	"""

	latent_dim: int
	patch_size: int = serializable_field(default=4)
	embed_dim: int = serializable_field(default=64)
	num_heads: int = serializable_field(default=8)
	mlp_dim: int = serializable_field(default=128)
	encoder_depth: int = serializable_field(default=4)
	decoder_depth: int = serializable_field(default=4)

	# loss/epochs hyperparameters
	contrast_temperature: float = serializable_field(default=0.07)
	recon_weight: float = serializable_field(default=1.0)
	contrast_weight: float = serializable_field(default=1.0)
	num_epochs: int = serializable_field(default=5)

	# optimizer and scheduler
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

	def __post_init__(self) -> None:
		"""Checks and validations after initialization."""
		assert self.latent_dim > 0, "latent_dim must be positive"
		assert self.patch_size > 0, "patch_size must be positive"
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
	"""2D Patch Embedding with linear projection (for single-channel square images).

	# Parameters:
	 - `embed_dim : int`
	    Dimension of embedded patch
	 - `patch_size : int`
	    The patch size (square patches)
	"""

	def __init__(
		self,
		embed_dim: int,
		patch_size: int,
	) -> None:
		"""summary

		extended summary

		# Parameters:
		 - `embed_dim : int`
		    dimension of the projected patch embeddings
		 - `patch_size : int`
		    patch size for each square
		"""
		super().__init__()
		self.embed_dim: int = embed_dim
		self.patch_size: int = patch_size

		# For single-channel images, in_channels=1
		in_channels: int = 1
		self.proj: nn.Conv2d = nn.Conv2d(
			in_channels=in_channels,
			out_channels=self.embed_dim,
			kernel_size=self.patch_size,
			stride=self.patch_size,
		)

	def forward(
		self,
		x: Float[Tensor, "batch n_ctx n_ctx"],
	) -> Float[Tensor, "batch num_patches embed_dim"]:
		"""summary

		extended summary

		# Parameters:
		 - `x : Float[Tensor, "batch n_ctx n_ctx"]`
		    input single-channel images of size (n_ctx x n_ctx)

		# Returns:
		 - `Float[Tensor, "batch num_patches embed_dim"]`
		    patch embeddings
		"""
		# b_size: int = x.shape[0]
		# n_ctx: int = x.shape[1]

		# Reshape to (B, 1, n_ctx, n_ctx)
		x_4d: Float[Tensor, "batch 1 n_ctx n_ctx"] = x.unsqueeze(1)

		# (B, embed_dim, n_ctx//patch_size, n_ctx//patch_size)
		x_proj: Float[Tensor, "batch embed_dim patchH patchW"] = self.proj(x_4d)

		# Flatten patch dims => (B, embed_dim, num_patches)
		x_proj = x_proj.flatten(2)

		# Transpose => (B, num_patches, embed_dim)
		x_proj = x_proj.transpose(1, 2)
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
		"""summary

		extended summary

		# Parameters:
		 - `embed_dim : int`
		    dimension of token embeddings
		 - `num_heads : int`
		    number of attention heads
		 - `mlp_dim : int`
		    dimension of hidden layer in the MLP
		 - `drop : float`
		    dropout probability
		    (defaults to 0.0)
		 - `attn_drop : float`
		    dropout probability for attention
		    (defaults to 0.0)
		"""
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
		self,
		x: Float[Tensor, "batch seq_len embed_dim"],
	) -> Float[Tensor, "batch seq_len embed_dim"]:
		"""summary

		extended summary

		# Parameters:
		 - `x : Float[Tensor, "batch seq_len embed_dim"]`
		    input token embeddings

		# Returns:
		 - `Float[Tensor, "batch seq_len embed_dim"]`
		    output token embeddings
		"""
		residual: Float[Tensor, "batch seq_len embed_dim"] = x
		h: Float[Tensor, "batch seq_len embed_dim"] = self.norm1(x)
		attn_out, _ = self.attn(h, h, h)
		x = residual + self.drop_attn(attn_out)

		residual = x
		h = self.norm2(x)
		h = self.mlp(h)
		x = residual + self.drop_mlp(h)
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
		"""summary

		extended summary

		# Parameters:
		 - `embed_dim : int`
		    dimension of token embeddings
		 - `num_heads : int`
		    number of attention heads
		 - `mlp_dim : int`
		    dimension of hidden layer in the MLP
		 - `drop : float`
		    dropout probability
		    (defaults to 0.0)
		 - `attn_drop : float`
		    dropout probability for attention
		    (defaults to 0.0)
		"""
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
		self,
		x: Float[Tensor, "batch seq_len embed_dim"],
	) -> Float[Tensor, "batch seq_len embed_dim"]:
		"""summary

		extended summary

		# Parameters:
		 - `x : Float[Tensor, "batch seq_len embed_dim"]`
		    input token embeddings

		# Returns:
		 - `Float[Tensor, "batch seq_len embed_dim"]`
		    output token embeddings
		"""
		residual: Float[Tensor, "batch seq_len embed_dim"] = x
		h: Float[Tensor, "batch seq_len embed_dim"] = self.norm1(x)
		attn_out, _ = self.attn(h, h, h)
		x = residual + self.drop_attn(attn_out)

		residual = x
		h = self.norm2(x)
		h = self.mlp(h)
		x = residual + self.drop_mlp(h)
		return x


@set_config_class(VitAEConfig)
class VisionTransformerEncoder(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Encoder (square inputs).

	Splits the input into patches (fixed `patch_size`), uses a stack of
	Transformer blocks, then outputs a `latent_dim` vector by average pooling
	the final patch embeddings.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		"""summary

		extended summary

		# Parameters:
		 - `config : VitAEConfig`
		    configuration object
		"""
		super().__init__(config)
		self.config: VitAEConfig = config

		# Patch embedding
		self.patch_embed: PatchEmbed = PatchEmbed(
			embed_dim=config.embed_dim,
			patch_size=config.patch_size,
		)

		# For square images, row/col are identical in number of patches,
		# but we still learn separate embeddings so each dimension is distinct.
		self.max_patches: int = 256  # set upper bound for patch dimension
		half_dim: int = config.embed_dim // 2
		self.row_embed: nn.Parameter = nn.Parameter(
			torch.zeros(self.max_patches, half_dim)
		)
		self.col_embed: nn.Parameter = nn.Parameter(
			torch.zeros(self.max_patches, half_dim)
		)
		nn.init.normal_(self.row_embed, std=0.02)
		nn.init.normal_(self.col_embed, std=0.02)

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
		self,
		x: Float[Tensor, "batch n_ctx n_ctx"],
	) -> Float[Tensor, "batch latent_dim"]:
		"""summary

		extended summary

		# Parameters:
		 - `x : Float[Tensor, "batch n_ctx n_ctx"]`
		    single-channel square input images

		# Returns:
		 - `Float[Tensor, "batch latent_dim"]`
		    latent representation after average pooling

		# Usage:

		```python
		>>> config = VitAEConfig(latent_dim=128)
		>>> enc = VisionTransformerEncoder(config)
		>>> x = torch.randn(4, 32, 32)  # batch=4, n_ctx=32
		>>> z = enc(x)
		>>> z.shape
		torch.Size([4, 128])
		```
		"""
		b_size: int = x.shape[0]
		n_ctx: int = x.shape[1]

		# (1) Embed patches => (B, num_patches, embed_dim)
		x_patches: Float[Tensor, "batch num_patches embed_dim"] = self.patch_embed(x)

		# (2) Number of patches in each dimension
		grid_size: int = n_ctx // self.config.patch_size
		# row_coords, col_coords each in [0..grid_size-1]
		row_coords: torch.Tensor = torch.arange(grid_size, device=x.device)
		col_coords: torch.Tensor = torch.arange(grid_size, device=x.device)

		# shape => (grid_size * grid_size,)
		row_coords = row_coords.unsqueeze(1).expand(grid_size, grid_size).reshape(-1)
		col_coords = col_coords.unsqueeze(0).expand(grid_size, grid_size).reshape(-1)

		# shape => (num_patches, embed_dim//2)
		pos_r: Float[Tensor, "num_patches half_dim"] = self.row_embed[row_coords]
		pos_c: Float[Tensor, "num_patches half_dim"] = self.col_embed[col_coords]
		# => (num_patches, embed_dim)
		pos: Float[Tensor, "num_patches embed_dim"] = torch.cat([pos_r, pos_c], dim=-1)

		# (3) Add positional embeddings => (B, num_patches, embed_dim)
		pos_expanded: Float[Tensor, "batch num_patches embed_dim"] = pos.unsqueeze(
			0
		).expand(b_size, -1, -1)
		x_patches = x_patches + pos_expanded

		# (4) Pass through encoder blocks
		for blk in self.blocks:
			x_patches = blk(x_patches)

		# (5) Final layer norm and mean pool
		x_patches = self.norm(x_patches)
		x_mean: Float[Tensor, "batch embed_dim"] = x_patches.mean(dim=1)

		# (6) Project to latent_dim
		z: Float[Tensor, "batch latent_dim"] = self.to_latent(x_mean)
		return z


@set_config_class(VitAEConfig)
class VisionTransformerDecoder(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Decoder (square outputs).

	Takes a (batch, latent_dim) vector, replicates it
	across the needed number of patches for a (n_ctx x n_ctx) image,
	passes it through decoder Transformer blocks, then projects
	back to patches and unpatchifies.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		"""summary

		extended summary

		# Parameters:
		 - `config : VitAEConfig`
		    configuration object
		"""
		super().__init__(config)
		self.config: VitAEConfig = config

		# Map latent -> embed_dim
		self.from_latent: nn.Linear = nn.Linear(config.latent_dim, config.embed_dim)

		# Separate row/col embeddings for decoding
		self.max_patches: int = 256
		half_dim: int = config.embed_dim // 2
		self.row_embed_dec: nn.Parameter = nn.Parameter(
			torch.zeros(self.max_patches, half_dim)
		)
		self.col_embed_dec: nn.Parameter = nn.Parameter(
			torch.zeros(self.max_patches, half_dim)
		)
		nn.init.normal_(self.row_embed_dec, std=0.02)
		nn.init.normal_(self.col_embed_dec, std=0.02)

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
		# For single-channel images, patch_dim = 1*(patch_size^2)
		patch_dim: int = config.patch_size * config.patch_size
		self.head: nn.Linear = nn.Linear(config.embed_dim, patch_dim)

	def forward(
		self,
		z: Float[Tensor, "batch latent_dim"],
		n_ctx: int,
	) -> Float[Tensor, "batch n_ctx n_ctx"]:
		"""summary

		extended summary

		# Parameters:
		 - `z : Float[Tensor, "batch latent_dim"]`
		    latent representation from the encoder
		 - `n_ctx : int`
		    side length of the square output image

		# Returns:
		 - `Float[Tensor, "batch n_ctx n_ctx"]`
		    reconstructed single-channel image

		# Usage:

		```python
		>>> config = VitAEConfig(latent_dim=128)
		>>> dec = VisionTransformerDecoder(config)
		>>> z = torch.randn(4, 128)
		>>> out = dec(z, 32)
		>>> out.shape
		torch.Size([4, 32, 32])
		```
		"""
		batch_size: int = z.shape[0]

		# (1) Map latent -> embed_dim
		latent_embed: Float[Tensor, "batch embed_dim"] = self.from_latent(z)

		# (2) Compute how many patches along each dimension
		grid_size: int = n_ctx // self.config.patch_size
		num_patches: int = grid_size * grid_size

		# (3) Expand (B, num_patches, embed_dim)
		latent_embed = latent_embed.unsqueeze(1).expand(batch_size, num_patches, -1)

		# (4) Build decoder positional embeddings
		row_coords: torch.Tensor = torch.arange(grid_size, device=z.device)
		col_coords: torch.Tensor = torch.arange(grid_size, device=z.device)
		row_coords = row_coords.unsqueeze(1).expand(grid_size, grid_size).reshape(-1)
		col_coords = col_coords.unsqueeze(0).expand(grid_size, grid_size).reshape(-1)

		pos_r: Float[Tensor, "num_patches half_dim"] = self.row_embed_dec[row_coords]
		pos_c: Float[Tensor, "num_patches half_dim"] = self.col_embed_dec[col_coords]
		pos: Float[Tensor, "num_patches embed_dim"] = torch.cat([pos_r, pos_c], dim=-1)
		pos = pos.unsqueeze(0).expand(batch_size, -1, -1)

		latent_embed = latent_embed + pos

		# (5) Pass through decoder blocks
		for blk in self.blocks:
			latent_embed = blk(latent_embed)

		# (6) Final norm
		latent_embed = self.norm(latent_embed)

		# (7) Project to patch pixels => (B, num_patches, patch_dim)
		patches: Float[Tensor, "batch num_patches patch_dim"] = self.head(latent_embed)

		# (8) Unpatchify:
		# => (B, num_patches, 1, patch_size, patch_size)
		patches = patches.view(
			batch_size,
			num_patches,
			1,
			self.config.patch_size,
			self.config.patch_size,
		)

		# => (B, grid_size, grid_size, 1, patch_size, patch_size)
		patches = patches.reshape(
			batch_size,
			grid_size,
			grid_size,
			1,
			self.config.patch_size,
			self.config.patch_size,
		)

		# reorder to => (B, 1, grid_size*patch_size, grid_size*patch_size)
		patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()

		# => (B, 1, n_ctx, n_ctx)
		x_recon_4d: Float[Tensor, "batch 1 n_ctx n_ctx"] = patches.view(
			batch_size,
			1,
			n_ctx,
			n_ctx,
		)

		# Optionally apply your custom transform
		x_recon_4d = convert_tril_rowstoch(x_recon_4d)

		# => (B, n_ctx, n_ctx)
		x_recon: Float[Tensor, "batch n_ctx n_ctx"] = x_recon_4d.squeeze(1)
		return x_recon


@set_config_class(VitAEConfig)
class VitAE(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Autoencoder with variable-sized square inputs.

	Produces a latent vector of shape (batch, latent_dim) from the encoder,
	and reconstructs an output of shape (batch, n_ctx, n_ctx) from the decoder.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		"""summary

		extended summary

		# Parameters:
		 - `config : VitAEConfig`
		    configuration object
		"""
		super().__init__(config)
		self.config: VitAEConfig = config
		self.encoder: VisionTransformerEncoder = VisionTransformerEncoder(config)
		self.decoder: VisionTransformerDecoder = VisionTransformerDecoder(config)

	def forward(
		self,
		x: Float[Tensor, "batch n_ctx n_ctx"],
	) -> Tuple[Float[Tensor, "batch n_ctx n_ctx"], Float[Tensor, "batch latent_dim"]]:
		"""Forward pass of VitAE (square inputs).

		# Parameters:
		 - `x : Float[Tensor, "batch n_ctx n_ctx"]`
		    input image of shape (n_ctx, n_ctx)

		# Returns:
		 - `Tuple[Float[Tensor, "batch n_ctx n_ctx"], Float[Tensor, "batch latent_dim"]]`
		    A tuple of (reconstruction, latent_vector).
		"""
		n_ctx: int = x.shape[1]

		# Encode
		z: Float[Tensor, "batch latent_dim"] = self.encoder(x)

		# Decode back to original size
		x_recon: Float[Tensor, "batch n_ctx n_ctx"] = self.decoder(z, n_ctx)
		return x_recon, z
