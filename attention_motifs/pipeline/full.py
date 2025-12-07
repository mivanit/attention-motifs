from attention_motifs.pipeline.s1_activations import generate_activations
from attention_motifs.pipeline.s1b_render_patterns import render_patterns
from attention_motifs.pipeline.s1c_write_idxs import write_idxs
from attention_motifs.pipeline.s2_features import compute_features
from attention_motifs.pipeline.s3b_feat_fig import feat_figures
from attention_motifs.pipeline.s4_head_dist import head_dists
from attention_motifs.pipeline.s4b_write_frontend import write_frontend
from attention_motifs.pipeline.s5_head_embed import head_embed
from attention_motifs.pipeline.s5b_head_embed_plots import head_embed_plots
from attention_motifs.pipeline.cfg import PipelineConfig
from attention_motifs.pipeline.s3_feat_proc import feat_proc


def full_pipeline(cfg: PipelineConfig) -> None:
	"""Run the full attention motifs pipeline."""
	print(f"Running full pipeline with config:\n{cfg}")

	# Step 1: Generate activations
	generate_activations(cfg)

	# Step 1.b: Render patterns
	render_patterns(cfg)

	# Step 1.c: Write indexes
	write_idxs(cfg)

	# Step 2: Compute features
	compute_features(cfg)

	# Step 3: Process features (PCA, etc.)
	feat_proc(cfg)

	# Step 3b: render figures related to features
	feat_figures(cfg)

	# Step 4: head distances
	head_dists(cfg)

	# Step 4b: write frontend files
	write_frontend(cfg)

	# Step 5: head embeddings
	head_embed(cfg)

	# Step 5b: head embedding plots
	head_embed_plots(cfg)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	full_pipeline(cfg)
