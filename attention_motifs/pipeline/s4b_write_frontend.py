import json
from pathlib import Path
import importlib.resources

from js_embedding_vis import fetch_jev

import attention_motifs
from attention_motifs.pipeline.cfg import PipelineConfig


def write_frontend(cfg: PipelineConfig) -> None:
	"""Writes the indexes for the attention patterns."""
	frontend_resources_path: Path = Path(
		importlib.resources.files(attention_motifs).joinpath("frontend"),  # type: ignore[arg-type]
	)

	ap_html: str = (frontend_resources_path / "attnpedia/index.html").read_text()
	ap_data: str = (frontend_resources_path / "attnpedia/ap.json").read_text()
	embed_html: str = fetch_jev()
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

	# write head embedding table and classifications page (only used after s5b)
	head_embed_table_html: str = (
		frontend_resources_path / "head_embed_table/index.html"
	).read_text()
	(cfg.figures_dir / "head_embed_table.html").write_text(head_embed_table_html)

	classes_html: str = (frontend_resources_path / "classes/index.html").read_text()
	(cfg.figures_dir / "classifications.html").write_text(classes_html)

	# write clustering dendrogram
	clustering_html: str = (frontend_resources_path / "clustering/index.html").read_text()
	clustering_dir: Path = cfg.vis_dir / "clustering"
	clustering_dir.mkdir(parents=True, exist_ok=True)
	(clustering_dir / "index.html").write_text(clustering_html)

	# write main index and diagram svg
	main_index_html: str = (frontend_resources_path / "index.html").read_text()
	(cfg.figures_dir / "../index.html").write_text(main_index_html)
	diagram_svg: str = (frontend_resources_path / "diagram.svg").read_text()
	(cfg.figures_dir / "../diagram.svg").write_text(diagram_svg)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	write_frontend(cfg)
