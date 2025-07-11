from pattern_lens.indexes import write_html_index

from attn_embed.pipeline.cfg import PipelineConfig


def write_idxs(cfg: PipelineConfig) -> None:
	"""Writes the indexes for the attention patterns."""
	write_html_index(
		path=cfg.patterns_dir,
		cfg_single={
			"data": {
				"attentionFilename": "attn.png",
				"links": {
					# Navigate to Pattern Lens with prompt selected
					"prompt": "index.html?prompts={prompt_hash}",
					# Navigate to AttentionPedia with head selected
					"head": "../vis/attnpedia/index.html?head_viewing={model}~L{layer}~H{head}",
				},
			},
		},
	)
	print(f"# Indexes written to {cfg.patterns_dir}")


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	write_idxs(cfg)
