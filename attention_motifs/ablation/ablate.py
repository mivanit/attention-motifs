"""Core ablation infrastructure using TransformerLens hooks.

Provides tools to ablate (disable) specific attention heads to test
their causal role in model behavior.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator

import torch
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from attention_motifs.attnpedia import parse_head

# Backwards compatibility alias
parse_head_string = parse_head


class AblationMethod(Enum):
	"""Methods for ablating attention heads."""

	ZERO = "zero"  # Set head output to 0
	MEAN = "mean"  # Set head output to pre-computed mean activation


@dataclass
class HeadAblator:
	"""TransformerLens-based attention head ablation.

	Provides methods to disable specific attention heads during forward
	passes using either zero or mean ablation.

	Attributes
	----------
	model
	    TransformerLens HookedTransformer model.
	mean_cache
	    Cache of mean activations per (layer, head) for mean ablation.
	device
	    Device for tensor operations.

	Examples
	--------
	>>> model = HookedTransformer.from_pretrained("gpt2-small")
	>>> ablator = HeadAblator(model)
	>>> # Compute mean activations from calibration data
	>>> ablator.compute_mean_activations(calibration_prompts)
	>>> # Run model with head L5:H5 ablated
	>>> with ablator.ablate_heads([(5, 5)], method=AblationMethod.ZERO):
	...     output = model(input_ids)
	"""

	model: "HookedTransformer"
	mean_cache: dict[tuple[int, int], Float[Tensor, "d_head"]] = field(
		default_factory=dict
	)
	device: str = field(default="cuda")

	def __post_init__(self):
		if HookedTransformer is None:
			raise ImportError(
				"transformer_lens is required for ablation. "
				"Install with: pip install transformer-lens"
			)
		self.device = str(self.model.cfg.device)

	@property
	def n_layers(self) -> int:
		return self.model.cfg.n_layers

	@property
	def n_heads(self) -> int:
		return self.model.cfg.n_heads

	@property
	def d_head(self) -> int:
		return self.model.cfg.d_head

	def _get_hook_name(self, layer: int) -> str:
		"""Get the hook name for attention head output at a layer."""
		return f"blocks.{layer}.attn.hook_z"

	def compute_mean_activations(
		self,
		prompts: list[str] | list[Tensor],
		batch_size: int = 8,
		show_progress: bool = True,
	) -> None:
		"""Pre-compute mean activations for all heads for mean ablation.

		Runs the model on calibration prompts and caches the mean
		activation for each attention head.

		Parameters
		----------
		prompts
		    List of text prompts or tokenized tensors for calibration.
		batch_size
		    Batch size for processing prompts.
		show_progress
		    Show progress bar.
		"""
		# Initialize accumulators
		activation_sums: dict[tuple[int, int], Tensor] = {}
		activation_counts: dict[tuple[int, int], int] = {}

		for layer in range(self.n_layers):
			for head in range(self.n_heads):
				activation_sums[(layer, head)] = torch.zeros(
					self.d_head, device=self.device
				)
				activation_counts[(layer, head)] = 0

		# Define hooks to accumulate activations
		def make_accumulator_hook(
			layer: int,
		) -> Callable[[Tensor, HookPoint], Tensor]:
			def hook(
				z: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint
			) -> Tensor:
				# z shape: (batch, pos, n_heads, d_head)
				for head in range(self.n_heads):
					head_z = z[:, :, head, :]  # (batch, pos, d_head)
					# Sum over batch and position
					activation_sums[(layer, head)] += head_z.sum(dim=(0, 1))
					activation_counts[(layer, head)] += (
						head_z.shape[0] * head_z.shape[1]
					)
				return z

			return hook

		# Build hooks for all layers
		hooks = [
			(self._get_hook_name(layer), make_accumulator_hook(layer))
			for layer in range(self.n_layers)
		]

		# Process prompts in batches
		iterator = range(0, len(prompts), batch_size)
		if show_progress:
			iterator = tqdm(iterator, desc="Computing mean activations")

		with torch.no_grad():
			for i in iterator:
				batch = prompts[i : i + batch_size]

				# Tokenize if needed
				if isinstance(batch[0], str):
					# we can safely assume everything in the batch is str
					tokens = self.model.to_tokens(batch) # ty: ignore[invalid-argument-type]
				else:
					tokens = torch.stack(batch) if isinstance(batch, list) else batch

				# Run with hooks

				self.model.run_with_hooks(tokens, fwd_hooks=hooks)

		# Compute means
		for (layer, head), total in activation_sums.items():
			count = activation_counts[(layer, head)]
			if count > 0:
				self.mean_cache[(layer, head)] = total / count
			else:
				self.mean_cache[(layer, head)] = torch.zeros(
					self.d_head, device=self.device
				)

	def _create_ablation_hook(
		self,
		layer: int,
		heads: list[int],
		method: AblationMethod,
	) -> Callable[[Tensor, HookPoint], Tensor]:
		"""Create a hook function that ablates specified heads.

		Parameters
		----------
		layer
		    Layer index.
		heads
		    List of head indices to ablate.
		method
		    Ablation method (zero or mean).

		Returns
		-------
		Callable
		    Hook function for TransformerLens.
		"""

		def hook(
			z: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint
		) -> Float[Tensor, "batch pos n_heads d_head"]:
			# Clone to avoid modifying original
			z_modified = z.clone()

			for head in heads:
				if method == AblationMethod.ZERO:
					z_modified[:, :, head, :] = 0.0
				elif method == AblationMethod.MEAN:
					if (layer, head) not in self.mean_cache:
						raise ValueError(
							f"Mean activation for layer {layer}, head {head} not cached. "
							"Call compute_mean_activations() first."
						)
					mean_val = self.mean_cache[(layer, head)]
					z_modified[:, :, head, :] = mean_val

			return z_modified

		return hook

	@contextmanager
	def ablate_heads(
		self,
		heads: list[tuple[int, int]],
		method: AblationMethod = AblationMethod.ZERO,
	) -> Iterator[None]:
		"""Context manager to temporarily ablate specified heads.

		Parameters
		----------
		heads
		    List of (layer, head) tuples to ablate.
		method
		    Ablation method (zero or mean).

		Yields
		------
		None
		    Use within a `with` block to run model with ablated heads.

		Examples
		--------
		>>> with ablator.ablate_heads([(5, 5), (6, 9)], AblationMethod.ZERO):
		...     logits = model(tokens)
		"""
		# Group heads by layer for efficient hook creation
		heads_by_layer: dict[int, list[int]] = {}
		for layer, head in heads:
			if layer not in heads_by_layer:
				heads_by_layer[layer] = []
			heads_by_layer[layer].append(head)

		# Create hooks
		hooks = [
			(
				self._get_hook_name(layer),
				self._create_ablation_hook(layer, layer_heads, method),
			)
			for layer, layer_heads in heads_by_layer.items()
		]

		# Register hooks by adding to the hook point's fwd_hooks list
		for hook_name, hook_fn in hooks:
			self.model.hook_dict[hook_name].add_hook(hook_fn)

		try:
			yield
		finally:
			# Remove our hooks (they are the last ones added to each hook point)
			for hook_name, _ in hooks:
				hook_point = self.model.hook_dict[hook_name]
				if hook_point.fwd_hooks:
					hook_point.fwd_hooks.pop()

	def run_with_ablation(
		self,
		tokens: Tensor,
		heads: list[tuple[int, int]],
		method: AblationMethod = AblationMethod.ZERO,
		return_type: str = "logits",
	) -> Tensor:
		"""Run model with specified heads ablated.

		Convenience method that wraps the context manager.

		Parameters
		----------
		tokens
		    Input token tensor.
		heads
		    List of (layer, head) tuples to ablate.
		method
		    Ablation method.
		return_type
		    What to return ("logits", "loss", etc.).

		Returns
		-------
		Tensor
		    Model output with ablated heads.
		"""
		with self.ablate_heads(heads, method):
			return self.model(tokens, return_type=return_type)

	def run_with_hooks_ablation(
		self,
		tokens: Tensor,
		heads: list[tuple[int, int]],
		method: AblationMethod = AblationMethod.ZERO,
		return_type: str = "logits",
	) -> Tensor:
		"""Run model with ablation using run_with_hooks.

		Alternative implementation using TransformerLens's run_with_hooks
		instead of persistent hook registration.

		Parameters
		----------
		tokens
		    Input token tensor.
		heads
		    List of (layer, head) tuples to ablate.
		method
		    Ablation method.
		return_type
		    What to return.

		Returns
		-------
		Tensor
		    Model output.
		"""
		# Group heads by layer
		heads_by_layer: dict[int, list[int]] = {}
		for layer, head in heads:
			if layer not in heads_by_layer:
				heads_by_layer[layer] = []
			heads_by_layer[layer].append(head)

		# Create hooks
		hooks = [
			(
				self._get_hook_name(layer),
				self._create_ablation_hook(layer, layer_heads, method),
			)
			for layer, layer_heads in heads_by_layer.items()
		]

		return self.model.run_with_hooks(
			tokens, fwd_hooks=hooks, return_type=return_type
		)
