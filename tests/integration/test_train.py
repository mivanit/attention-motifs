# tests/integration/test_small_vitae_integration.py
import pytest
import torch
from jaxtyping import Float

from attention_motifs.dataset.util import (
	AttentionPatternMetadataArray,
	AttentionPatternMetadata,
)
from attention_motifs.dataset.dataset import (
	AttentionPatternDataset,
	CollectedAttentionPatternDataloader,
)
from attention_motifs.dataset.prompts import PromptDataset, PromptDatasetConfig
from attention_motifs.autoencoder.vit_ae import VitAEConfig


@pytest.mark.integration
def test_small_vit_ae_integration() -> None:
	"""Integration test for VitAE with a very small synthetic dataset

	We:
	 - create 4 random attention patterns of shape (4,4)
	 - define minimal metadata
	 - wrap them in a `CollectedAttentionPatternDataloader`
	 - build a tiny `VitAE` model
	 - train for 2 epochs, 2 steps total
	 - check no NaNs appear in loss, outputs, or model params

	# Parameters:
	 - None

	# Returns:
	 - None

	# Usage:
	```python
	# From your project root:
	# run pytest with:
	pytest tests/integration/test_small_vitae_integration.py -v
	```

	# Raises:
	 - AssertionError : if any NaNs occur in loss, outputs, or params
	"""

	# ------------------------------
	# 1) Create a tiny synthetic dataset
	# ------------------------------
	n_patterns: int = 4
	n_ctx: int = 4
	# [4,4,4] => 4 samples, each a 4×4 attention pattern
	patterns: Float[torch.Tensor, "n_patterns n_ctx n_ctx"] = torch.randn(
		n_patterns, n_ctx, n_ctx
	)

	# minimal metadata for each pattern
	meta_list: list[AttentionPatternMetadata] = []
	for i in range(n_patterns):
		meta_list.append(
			AttentionPatternMetadata(
				model_name="test-model",
				idx_layer=0,
				idx_head=i,
				n_ctx=n_ctx,
				prompt_hash=i,
			)
		)

	# wrap in an AttentionPatternDataset
	dataset: AttentionPatternDataset = AttentionPatternDataset(
		n_ctx=n_ctx,
		n_patterns=n_patterns,
		patterns=patterns,
		metadata=AttentionPatternMetadataArray.from_list(meta_list),
		raw_scores=False,
	)

	# wrap in a "collected" dataloader, with a dummy PromptDataset
	prompts_config: PromptDatasetConfig = PromptDatasetConfig(
		name="dummy",
		source_path=torch.randn(0),  # not used, but must be set
		source_info={},
	)
	dummy_prompts: PromptDataset = PromptDataset(
		config=prompts_config,
		prompts=[],
		hash_map={},
	)
	dl: CollectedAttentionPatternDataloader = CollectedAttentionPatternDataloader(
		config=VitAEConfig(d_latent=8),  # not used, but must exist
		prompts=dummy_prompts,
		datasets={n_ctx: dataset},
	)

	# ------------------------------
	# 2) Build a minimal VitAE model
	# ------------------------------
	config: VitAEConfig = VitAEConfig(
		d_latent=2,
		d_model=2,
		patch_size=1,  # ensures minimal "patch" logic
		max_patches=4,  # must be >= n_ctx/patch_size
		num_heads=2,
		mlp_dim=2,
		encoder_depth=1,
		decoder_depth=1,
		contrast_temperature=0.1,
		recon_weight=1.0,
		contrast_weight=0.0,  # keep things simpler (pure reconstruction)
		num_epochs=2,  # we'll do 2 epochs below
		learning_rate=1e-3,
	)

	# TODO: finish this test to use the train() function
