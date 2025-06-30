from attn_embed.pipeline.s1_activations import generate_activations
from attn_embed.pipeline.s1b_render_patterns import render_patterns
from attn_embed.pipeline.s2_features import compute_features
from attn_embed.pipeline.s4_head_dist import head_dists
from attn_embed.util.pipeline_cfg import PipelineConfig
from attn_embed.pipeline.s3_feat_proc import feat_proc


def full_pipeline(cfg: PipelineConfig) -> None:
	"""Run the full attention motifs pipeline."""
	print(f"Running full pipeline with config:\n{cfg}")

	# Step 1: Generate activations
	generate_activations(cfg)

	# Step 1.b: Render patterns
	render_patterns(cfg)

	# Step 2: Compute features
	compute_features(cfg)

	# Step 3: Process features (PCA, etc.)
	feat_proc(cfg)

	# Step 4: head distances
	head_dists(cfg)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	full_pipeline(cfg)