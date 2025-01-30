from typing import Tuple, Dict, Any, Type

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from zanj.torchutil import ConfiguredModel, set_config_class

from attention_motifs.train import convert_tril_rowstoch
from attention_motifs.dbg import dbg


@serializable_dataclass(kw_only=True)
class VitAEConfig(SerializableDataclass):
	"""Configuration for a Vision Transformer Autoencoder (square images of size n_ctx x n_ctx)

	# Parameters:
	 - `d_latent : int`
	    Dimension of the final latent space (for contrastive usage).
	 - `patch_size : int`
	    Size of each patch (image is split into patches).
	 - `d_model : int`
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

	# architecture
	# ==================================================

	act_fn: type[nn.Module] = serializable_field(
		default=nn.GELU,
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(nn, x),
	)

	# basic and patch embedding hparams
	d_latent: int
	d_model: int = serializable_field(default=64)
	patch_size: int = serializable_field(default=4)
	max_patches: int = serializable_field(default=64)

	# transformer block hparams
	num_heads: int = serializable_field(default=8)
	mlp_dim: int = serializable_field(default=256)

	# how many transformer blocks
	encoder_depth: int = serializable_field(default=2)
	decoder_depth: int = serializable_field(default=2)

	# training
	# ==================================================

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

	# ==================================================

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
	 - `d_model : int`
	    Dimension of embedded patch
	 - `patch_size : int`
	    The patch size (square patches)
	"""

	def __init__(
		self,
		d_model: int,
		patch_size: int,
		max_patches: int,
		in_channels: int = 1,
	) -> None:
		super().__init__()
		self.d_model: int = d_model
		self.patch_size: int = patch_size
		self.in_channels: int = in_channels

		self.proj: nn.Conv2d = nn.Conv2d(
			in_channels=self.in_channels,
			out_channels=self.d_model,
			kernel_size=self.patch_size,
			stride=self.patch_size,
		)

		# positional embeddings for x and y
		self.pos_embeds: nn.ModuleList = nn.ModuleList([
			nn.Embedding(
				num_embeddings=max_patches,
				embedding_dim=d_model,
			)
			for _ in range(2)
		])

	def forward(
		self,
		x: Float[Tensor, "batch channels=1 n_ctx n_ctx"],
	) -> Float[Tensor, "batch num_patches d_model"]:
		# 1) Convolutional projection
		x_proj: Float[Tensor, "batch d_model ax_patches ax_patches"] = self.proj(x)
		ax_patches: int = x_proj.shape[2]
		assert tuple(x_proj.shape) == (x.shape[0], self.d_model, ax_patches, ax_patches)

		# 2) Build separate positional embeddings for x and y, shape = (2, ax_patches, d_model)
		positions: Int[Tensor, "ax_patches"] = torch.arange(ax_patches).to(x.device)

		pos_embeds: Float[Tensor, "2 ax_patches d_model"] = torch.stack(
			[p(positions) for p in self.pos_embeds]
		)
		# pos_embeds[0] = x-embeddings (ax_patches, d_model)
		# pos_embeds[1] = y-embeddings (ax_patches, d_model)

		# 3) "Outer add" to get a 2D embedding grid for each (row, col)
		# pos2d will have shape (ax_patches, ax_patches, d_model)
		pos2d: Float[Tensor, "ax_patches ax_patches d_model"] = pos_embeds[0].unsqueeze(
			1
		) + pos_embeds[1].unsqueeze(0)

		# 4) Reshape for broadcast-add to x_proj
		# pos2d_perm: (d_model, ax_patches, ax_patches)
		# then unsqueeze -> (1, d_model, ax_patches, ax_patches)
		# pos2d_perm = pos2d.permute(2, 0, 1).unsqueeze(0)
		pos2d_perm: Float[Tensor, "1 d_model ax_patches ax_patches"] = pos2d.permute(
			2, 0, 1
		).unsqueeze(0)

		# 5) Add to the convolution outputs
		x_proj = x_proj + pos2d_perm  # broadcast over batch dim

		# 6) Flatten to patches and transpose so d_model is last
		# x_seq: Float[Tensor, "batch d_model num_patches"]
		# x_out: Float[Tensor, "batch num_patches d_model"]
		return x_proj.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
	"""A Transformer Encoder Block (pre-LayerNorm)

	# Parameters:
	 - `d_model : int`
		dimension of token embeddings
	 - `num_heads : int`
		number of attention heads
	 - `mlp_dim : int`
		dimension of hidden layer in the MLP
	"""

	def __init__(
		self,
		d_model: int,
		num_heads: int,
		mlp_dim: int,
		act_fn: type[nn.Module] = nn.GELU,
		# drop: float = 0.0,
		# attn_drop: float = 0.0,
	) -> None:
		super().__init__()
		self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
		self.attn: nn.MultiheadAttention = nn.MultiheadAttention(
			embed_dim=d_model,
			num_heads=num_heads,
			batch_first=True,
			# dropout=attn_drop,
		)
		# self.drop_attn: nn.Dropout = nn.Dropout(drop)

		self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
		self.mlp: nn.Sequential = nn.Sequential(
			nn.Linear(d_model, mlp_dim),
			act_fn(),
			nn.Linear(mlp_dim, d_model),
		)
		# self.drop_mlp: nn.Dropout = nn.Dropout(drop)

	def forward(
		self,
		x: Float[Tensor, "batch seq_len d_model"],
	) -> Float[Tensor, "batch seq_len d_model"]:
		residual: Float[Tensor, "batch seq_len d_model"] = x
		h: Float[Tensor, "batch seq_len d_model"] = self.norm1(x)
		attn_out, _ = self.attn(h, h, h)
		x = residual + attn_out

		residual = x
		h = self.norm2(x)
		h = self.mlp(h)
		x = residual + h
		return x


@set_config_class(VitAEConfig)
class VitEncoder(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Encoder (square inputs).

	Splits the input into patches (fixed `patch_size`), uses a stack of
	Transformer blocks, then outputs a `d_latent` vector by average pooling
	the final patch embeddings.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		super().__init__(config)
		self.config: VitAEConfig = config

		# Patch embedding
		self.patch_embed: PatchEmbed = PatchEmbed(
			d_model=config.d_model,
			patch_size=config.patch_size,
			max_patches=config.max_patches,
		)

		# Transformer encoder blocks
		self.blocks: nn.Module = nn.Sequential(
			*[
				TransformerBlock(
					d_model=config.d_model,
					num_heads=config.num_heads,
					mlp_dim=config.mlp_dim,
				)
				for _ in range(config.encoder_depth)
			]
		)

		# Final projection to latent space
		self.ln_final: nn.LayerNorm = nn.LayerNorm(config.d_model)
		self.to_latent: nn.Linear = nn.Linear(config.d_model, config.d_latent)

	def forward(
		self,
		x: Float[Tensor, "batch n_ctx n_ctx"],
	) -> Float[Tensor, "batch d_latent"]:
		# (1) Embed patches => (B, num_patches, d_model)
		x_patches: Float[Tensor, "batch num_patches d_model"] = self.patch_embed(x)

		# (2) Pass through encoder blocks
		x_patches = self.blocks(x_patches)

		# (3) Final layer norm and mean pool
		x_patches = self.ln_final(x_patches)
		x_mean: Float[Tensor, "batch d_model"] = x_patches.mean(dim=1)

		# (4) Project to d_latent
		z: Float[Tensor, "batch d_latent"] = self.to_latent(x_mean)
		return z


@set_config_class(VitAEConfig)
class VitDecoder(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Decoder (square outputs).

	Takes a (batch, d_latent) vector, replicates it
	across the needed number of patches for a (n_ctx x n_ctx) image,
	passes it through decoder Transformer blocks, then projects
	back to patches and unpatchifies.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		super().__init__(config)
		self.config: VitAEConfig = config

		# Map latent -> d_model
		self.from_latent: nn.Linear = nn.Linear(config.d_latent, config.d_model)

		# positional embeddings for x and y
		self.pos_embeds: nn.ModuleList = nn.ModuleList([
			nn.Embedding(
				num_embeddings=config.max_patches,
				embedding_dim=config.d_model,
			)
			for _ in range(2)
		])

		# Decoder transformer blocks
		self.blocks: nn.Module = nn.Sequential(
			*[
				TransformerBlock(
					d_model=config.d_model,
					num_heads=config.num_heads,
					mlp_dim=config.mlp_dim,
				)
				for _ in range(config.decoder_depth)
			]
		)
		self.norm: nn.LayerNorm = nn.LayerNorm(config.d_model)

		# Final projection from d_model -> patch pixels
		# For single-channel images, patch_dim = 1*(patch_size^2)
		patch_dim: int = config.patch_size * config.patch_size
		self.head: nn.Linear = nn.Linear(config.d_model, patch_dim)

	def forward(
		self,
		latent: Float[Tensor, "batch d_latent"],
		n_ctx: int,
	) -> Float[Tensor, "batch n_ctx n_ctx"]:
		batch_size: int = latent.shape[0]
		ax_patches: int = n_ctx // self.config.patch_size

		# Map latent -> d_model
		latent_embed: Float[Tensor, "batch d_model"] = self.from_latent(latent)

		# Build separate positional embeddings for x and y, shape = (2, ax_patches, d_model)
		positions: Int[Tensor, "ax_patches"] = torch.arange(
			ax_patches, device=latent.device
		)
		pos_embeds: Float[Tensor, "2 ax_patches d_model"] = torch.stack(
			[p(positions) for p in self.pos_embeds]
		)

		# "Outer add" to get a 2D embedding grid for each (row, col)
		# pos2d will have shape (ax_patches, ax_patches, d_model)
		pos2d: Float[Tensor, "ax_patches ax_patches d_model"] = pos_embeds[0].unsqueeze(
			1
		) + pos_embeds[1].unsqueeze(0)

		# Reshape for broadcast-add to x_proj
		# pos2d_perm: (d_model, ax_patches, ax_patches)
		# then unsqueeze -> (1, d_model, ax_patches, ax_patches)
		# pos2d_perm = pos2d.permute(2, 0, 1).unsqueeze(0)
		pos2d_perm: Float[Tensor, "1 d_model ax_patches ax_patches"] = pos2d.permute(
			2, 0, 1
		).unsqueeze(0)

		# patches_grid[i,j] = latent_embed + pos_embeds[0][i] + pos_embeds[1][j]
		dbg(pos2d_perm.shape)
		dbg(latent_embed.shape)
		patches_grid: Float[Tensor, "batch d_model ax_patches ax_patches"] = (
			latent_embed.unsqueeze(-1).unsqueeze(-1) + pos2d_perm
		)
		dbg(patches_grid.shape)
		patches_seq: Float[Tensor, "batch n_patches d_model"] = patches_grid.flatten(1)

		# Pass through decoder blocks
		dbg(patches_seq.shape)
		patches_seq = self.blocks(patches_seq)
		dbg(patches_seq.shape)

		# (6) Final norm
		patches_seq = self.norm(patches_seq)

		# (7) Project to patch pixels => (B, num_patches, patch_dim)
		patches: Float[Tensor, "batch num_patches patch_dim"] = self.head(latent_embed)
		num_patches: int = patches.shape[1]
		assert num_patches == ax_patches * ax_patches

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
			n_ctx,
			n_ctx,
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

		# make lower-triangular and row-stochastic
		x_recon_4d = convert_tril_rowstoch(x_recon_4d)

		return x_recon_4d


@set_config_class(VitAEConfig)
class VitAE(ConfiguredModel[VitAEConfig]):
	"""Vision Transformer Autoencoder with variable-sized square inputs.

	Produces a latent vector of shape (batch, d_latent) from the encoder,
	and reconstructs an output of shape (batch, n_ctx, n_ctx) from the decoder.
	"""

	def __init__(self, config: VitAEConfig) -> None:
		super().__init__(config)
		self.config: VitAEConfig = config
		self.encoder: VitEncoder = VitEncoder(config)
		self.decoder: VitDecoder = VitDecoder(config)

	def forward(
		self,
		x: Float[Tensor, "batch n_ctx n_ctx"],
	) -> Tuple[Float[Tensor, "batch n_ctx n_ctx"], Float[Tensor, "batch d_latent"]]:
		n_ctx: int = x.shape[1]

		# Encode
		latent: Float[Tensor, "batch d_latent"] = self.encoder(x)

		# Decode back to original size
		x_recon: Float[Tensor, "batch n_ctx n_ctx"] = self.decoder(latent, n_ctx)
		return x_recon, latent
