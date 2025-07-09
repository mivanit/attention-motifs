import json
from pathlib import Path
import importlib.resources


import attn_embed
from attn_embed.util.pipeline_cfg import PipelineConfig


def write_frontend(cfg: PipelineConfig) -> None:
	"""Writes the indexes for the attention patterns."""
	frontend_resources_path: Path = Path(
		importlib.resources.files(attn_embed).joinpath("frontend"),  # type: ignore[arg-type]
	)

	ap_html: str = (frontend_resources_path / "attnpedia/index.html").read_text()
	ap_data: str = (frontend_resources_path / "attnpedia/ap.json").read_text()
	embed_html: str = (frontend_resources_path / "embeds/index.html").read_text()
	head_embed_table_html: str = (
		frontend_resources_path / "head_embed_table/index.html"
	).read_text()

	vis_output_path: Path = cfg.vis_dir

	# write attentionpedia
	ap_vis_cfg: dict = cfg.vis_configs["attentionpedia"]
	ap_dir: Path = vis_output_path / ap_vis_cfg["path"]
	ap_dir.mkdir(parents=True, exist_ok=True)
	(ap_dir / "index.html").write_text(ap_html)
	(ap_dir / "ap.json").write_text(ap_data)

	(ap_dir / ap_vis_cfg["cfg_path"]).write_text(
		json.dumps(ap_vis_cfg["cfg"], indent="\t")
	)

	# write pattern embedding
	embed_pattern_vis_cfg: dict = cfg.vis_configs["embed_pattern"]
	embed_pattern_dir: Path = vis_output_path / embed_pattern_vis_cfg["path"]
	embed_pattern_dir.mkdir(parents=True, exist_ok=True)
	(embed_pattern_dir / "index.html").write_text(embed_html)
	(embed_pattern_dir / embed_pattern_vis_cfg["cfg_path"]).write_text(
		json.dumps(embed_pattern_vis_cfg["cfg"], indent="\t")
	)

	# write head embedding
	embed_head_vis_cfg: dict = cfg.vis_configs["embed_head"]
	embed_head_dir: Path = vis_output_path / embed_head_vis_cfg["path"]
	embed_head_dir.mkdir(parents=True, exist_ok=True)
	(embed_head_dir / "index.html").write_text(embed_html)
	(embed_head_dir / embed_head_vis_cfg["cfg_path"]).write_text(
		json.dumps(embed_head_vis_cfg["cfg"], indent="\t")
	)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	write_frontend(cfg)
