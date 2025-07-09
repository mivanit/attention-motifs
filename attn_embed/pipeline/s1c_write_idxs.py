from pattern_lens.indexes import write_html_index

from attn_embed.util.pipeline_cfg import PipelineConfig


def write_idxs(cfg: PipelineConfig) -> None:
	"""Writes the indexes for the attention patterns."""
	write_html_index(
		path=cfg.patterns_dir,
		cfg_single={
			"data": {
				"attentionFilename": "attn.png",
			},
		},
	)
	print(f"# Indexes written to {cfg.patterns_dir}")


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	write_idxs(cfg)
