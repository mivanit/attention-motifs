"""Core ablation infrastructure using TransformerLens hooks.

Provides tools to ablate (disable) specific attention heads to test
their causal role in model behavior.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Iterator, cast

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
	PATTERN_PRESERVING = (
		"pattern_preserving"  # Zero head output, freeze all attn patterns
	)


@dataclass
class HeadAblator:
	"""TransformerLens-based attention head ablation.

	Provides methods to disable specific attention heads during forward
	passes using zero, mean, or pattern-preserving ablation.

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
	>>> model = load_model("gpt2-small")
	>>> ablator = HeadAblator(model)
	>>> # Zero ablation
	>>> with ablator.ablate_heads([(5, 5)], method=AblationMethod.ZERO):
	...     output = model(input_ids)
	>>> # Pattern-preserving ablation
	>>> ablator.set_clean_patterns(tokens)
	>>> with ablator.ablate_heads([(5, 5)], method=AblationMethod.PATTERN_PRESERVING):
	...     output = model(input_ids)
	"""

	model: "HookedTransformer"
	mean_cache: dict[tuple[int, int], Float[Tensor, "d_head"]] = field(
		default_factory=dict
	)
	device: str = field(default="cuda")
	_clean_patterns: dict[str, Float[Tensor, "batch n_heads dest src"]] | None = field(
		default=None, init=False, repr=False
	)

	def __post_init__(self):
		if HookedTransformer is None:
			raise ImportError(
				"transformer_lens is required for ablation. "
				"Install with: pip install transformer-lens"
			)
		self.device = str(self.model.cfg.device)
		self.model.eval()

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

	def _get_pattern_hook_name(self, layer: int) -> str:
		"""Get the hook name for attention patterns at a layer."""
		return f"blocks.{layer}.attn.hook_pattern"

	def cache_clean_patterns(
		self,
		tokens: Tensor,
		prepend_bos: bool = False,
	) -> dict[str, Float[Tensor, "batch n_heads dest src"]]:
		"""Run a clean forward pass and cache all attention patterns.

		.. warning:: Must be called **outside** any ``ablate_heads``
		   context — the model must have no active hooks so the forward
		   pass is genuinely clean.

		Parameters
		----------
		tokens
		    Input token tensor.
		prepend_bos
		    Whether the model should prepend BOS.

		Returns
		-------
		dict[str, Tensor]
		    Mapping from hook name to attention pattern tensor.
		"""
		pattern_names: list[str] = [
			self._get_pattern_hook_name(layer) for layer in range(self.n_layers)
		]
		with torch.no_grad():
			_, cache = self.model.run_with_cache(
				tokens, names_filter=pattern_names, prepend_bos=prepend_bos
			)
		clean_patterns: dict[str, Float[Tensor, "batch n_heads dest src"]] = {
			name: cache[name].clone() for name in pattern_names
		}
		return clean_patterns

	def set_clean_patterns(
		self,
		tokens: Tensor,
		prepend_bos: bool = False,
	) -> None:
		"""Cache clean attention patterns for pattern-preserving ablation.

		Must be called before using ``AblationMethod.PATTERN_PRESERVING``
		with the ``ablate_heads`` context manager.

		Parameters
		----------
		tokens
		    Input token tensor (same tokens that will be used in the ablated pass).
		prepend_bos
		    Whether the model should prepend BOS.
		"""
		self._clean_patterns = self.cache_clean_patterns(
			tokens, prepend_bos=prepend_bos
		)

	def clear_clean_patterns(self) -> None:
		"""Clear cached clean patterns to free memory."""
		self._clean_patterns = None

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
					head_z: Tensor = z[:, :, head, :]  # (batch, pos, d_head)
					# Sum over batch and position
					activation_sums[(layer, head)] += head_z.sum(dim=(0, 1))
					activation_counts[(layer, head)] += (
						head_z.shape[0] * head_z.shape[1]
					)
				return z

			return hook

		# Build hooks for all layers
		hooks: list[tuple[str, Callable]] = [
			(self._get_hook_name(layer), make_accumulator_hook(layer))
			for layer in range(self.n_layers)
		]

		# Process prompts in batches
		batch_indices: Iterable[int] = range(0, len(prompts), batch_size)
		if show_progress:
			batch_indices = tqdm(batch_indices, desc="Computing mean activations")

		with torch.no_grad():
			for i in batch_indices:
				batch = prompts[i : i + batch_size]

				# Tokenize if needed
				if isinstance(batch[0], str):
					# isinstance on batch[0] doesn't narrow the list type
					tokens: Tensor = self.model.to_tokens(cast(list[str], batch))
				else:
					tokens = torch.stack(cast(list[Tensor], batch)) if isinstance(batch, list) else batch

				# Run with hooks

				self.model.run_with_hooks(tokens, fwd_hooks=hooks)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

		# Compute means
		for (layer, head), total in activation_sums.items():
			count: int = activation_counts[(layer, head)]
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
		    Ablation method (zero, mean, or pattern_preserving — all zero
		    the head output; the difference is in whether patterns are frozen).

		Returns
		-------
		Callable
		    Hook function for TransformerLens.
		"""

		def hook(
			z: Float[Tensor, "batch pos n_heads d_head"], hook: HookPoint
		) -> Float[Tensor, "batch pos n_heads d_head"]:
			# Clone to avoid modifying original
			z_modified: Tensor = z.clone()

			for head in heads:
				if method in (AblationMethod.ZERO, AblationMethod.PATTERN_PRESERVING):
					z_modified[:, :, head, :] = 0.0
				elif method == AblationMethod.MEAN:
					if (layer, head) not in self.mean_cache:
						raise ValueError(
							f"Mean activation for layer {layer}, head {head} not cached. "
							"Call compute_mean_activations() first."
						)
					mean_val: Tensor = self.mean_cache[(layer, head)]
					z_modified[:, :, head, :] = mean_val

			return z_modified

		return hook

	def _create_pattern_freeze_hook(
		self,
		clean_pattern: Float[Tensor, "batch n_heads dest src"],
	) -> Callable[[Tensor, HookPoint], Tensor]:
		"""Create a hook that replaces attention patterns with cached clean ones.

		Parameters
		----------
		clean_pattern
		    Clean attention pattern to substitute.

		Returns
		-------
		Callable
		    Hook function that returns the clean pattern.
		"""

		def hook(
			pattern: Float[Tensor, "batch n_heads dest src"], hook: HookPoint
		) -> Float[Tensor, "batch n_heads dest src"]:
			# Slice to actual batch size — cached pattern may have a larger batch
			# dim when the experiment micro-batches the forward pass.
			return clean_pattern[: pattern.shape[0]]

		return hook

	@contextmanager
	def ablate_heads(
		self,
		heads: list[tuple[int, int]],
		method: AblationMethod = AblationMethod.ZERO,
	) -> Iterator[None]:
		"""Context manager to temporarily ablate specified heads.

		For ``PATTERN_PRESERVING``, ``set_clean_patterns(tokens)`` must be
		called first to cache the clean-run attention patterns.

		Parameters
		----------
		heads
		    List of (layer, head) tuples to ablate.
		method
		    Ablation method (zero, mean, or pattern_preserving).

		Yields
		------
		None
		    Use within a ``with`` block to run model with ablated heads.

		Examples
		--------
		>>> with ablator.ablate_heads([(5, 5), (6, 9)], AblationMethod.ZERO):
		...     logits = model(tokens)
		"""
		if method == AblationMethod.PATTERN_PRESERVING and self._clean_patterns is None:
			raise ValueError(
				"Call set_clean_patterns(tokens) before using PATTERN_PRESERVING. "
				"Pattern-preserving ablation requires a clean forward pass first."
			)

		# Group heads by layer for efficient hook creation
		heads_by_layer: dict[int, list[int]] = {}
		for layer, head in heads:
			if layer not in heads_by_layer:
				heads_by_layer[layer] = []
			heads_by_layer[layer].append(head)

		# Create z-ablation hooks
		all_hooks: list[tuple[str, Callable[[Tensor, HookPoint], Tensor]]] = [
			(
				self._get_hook_name(layer),
				self._create_ablation_hook(layer, layer_heads, method),
			)
			for layer, layer_heads in heads_by_layer.items()
		]

		# For pattern-preserving: add pattern-freeze hooks for ALL layers
		if method == AblationMethod.PATTERN_PRESERVING:
			assert self._clean_patterns is not None
			for layer in range(self.n_layers):
				pattern_hook_name: str = self._get_pattern_hook_name(layer)
				clean_pattern: Tensor = self._clean_patterns[pattern_hook_name]
				all_hooks.append(
					(
						pattern_hook_name,
						self._create_pattern_freeze_hook(clean_pattern),
					)
				)

		# Register hooks
		for hook_name, hook_fn in all_hooks:
			self.model.hook_dict[hook_name].add_hook(hook_fn)  # ty: ignore[invalid-argument-type] # pyright: ignore[reportArgumentType]

		try:
			yield
		finally:
			# Remove our hooks (they are the last ones added to each hook point).
			# NOTE: assumes no nested ablate_heads contexts — our hooks are
			# always the last added to each hook point.
			# Must call handle.hook.remove() to deregister the underlying PyTorch
			# forward hook — just popping from fwd_hooks only removes TransformerLens's
			# bookkeeping entry but leaves the hook active in nn.Module._forward_hooks.
			for hook_name, _ in all_hooks:
				hook_point = self.model.hook_dict[hook_name]
				if hook_point.fwd_hooks:
					handle = hook_point.fwd_hooks.pop()
					handle.hook.remove()

	def run_with_ablation(
		self,
		tokens: Tensor,
		heads: list[tuple[int, int]],
		method: AblationMethod = AblationMethod.ZERO,
		return_type: str = "logits",
		prepend_bos: bool = True,
	) -> Tensor:
		"""Run model with specified heads ablated.

		For ``PATTERN_PRESERVING``, automatically caches clean patterns
		if not already cached.

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
		prepend_bos
		    Whether the model should prepend BOS.

		Returns
		-------
		Tensor
		    Model output with ablated heads.
		"""
		if method == AblationMethod.PATTERN_PRESERVING and self._clean_patterns is None:
			self.set_clean_patterns(tokens, prepend_bos=prepend_bos)

		with self.ablate_heads(heads, method), torch.no_grad():
			return self.model(tokens, return_type=return_type, prepend_bos=prepend_bos)
