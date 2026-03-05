import json
from pathlib import Path
import importlib.resources
from typing import Any

from js_embedding_vis import fetch_jev
from js_embedding_vis.inline_cfg import inline_hooks  # type: ignore[import-untyped]

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

	vis_output_path: Path = cfg.vis_dir

	# write attentionpedia
	ap_vis_cfg: dict[str, Any] = cfg.vis_configs["attentionpedia"]
	ap_dir: Path = vis_output_path / ap_vis_cfg["path"]
	ap_dir.mkdir(parents=True, exist_ok=True)
	(ap_dir / "index.html").write_text(ap_html)
	(ap_dir / "ap.json").write_text(ap_data)

	(ap_dir / ap_vis_cfg["cfg_path"]).write_text(
		json.dumps(ap_vis_cfg["cfg"], indent="\t")
	)

	# write pattern embedding
	embed_pattern_vis_cfg: dict[str, Any] = cfg.vis_configs["embed_pattern"]
	embed_pattern_dir: Path = vis_output_path / embed_pattern_vis_cfg["path"]
	embed_pattern_dir.mkdir(parents=True, exist_ok=True)
	(embed_pattern_dir / "index.html").write_text(embed_html)
	(embed_pattern_dir / embed_pattern_vis_cfg["cfg_path"]).write_text(
		json.dumps(embed_pattern_vis_cfg["cfg"], indent="\t")
	)

	# write head embedding (with clustering hooks injected)
	embed_head_vis_cfg: dict[str, Any] = cfg.vis_configs["embed_head"]
	embed_head_dir: Path = vis_output_path / embed_head_vis_cfg["path"]
	embed_head_dir.mkdir(parents=True, exist_ok=True)

	# Build hooks JS: cluster_utils + ClusteringLoader + setup script
	cluster_utils_js: str = (
		frontend_resources_path / "shared/cluster_utils.js"
	).read_text()
	clustering_js: str = (
		frontend_resources_path / "attnpedia/src/clustering.js"
	).read_text()
	setup_js: str = (
		frontend_resources_path / "shared/embed_clustering_setup.js"
	).read_text()
	hooks_js: str = cluster_utils_js + "\n" + clustering_js + "\n" + setup_js

	# Inject hooks into jev HTML
	embed_head_html: str = inline_hooks(hooks_js, embed_html)
	(embed_head_dir / "index.html").write_text(embed_head_html)
	(embed_head_dir / embed_head_vis_cfg["cfg_path"]).write_text(
		json.dumps(embed_head_vis_cfg["cfg"], indent="\t")
	)

	# NOTE: clustering + cluster_trends frontends are written by s6c_write_cluster_frontend

	# write head embedding table, classifications page, and main index (only if figures enabled)
	if cfg.do_figures:
		assert cfg.figures_dir is not None

		head_embed_table_html: str = (
			frontend_resources_path / "head_embed_table/index.html"
		).read_text()
		(cfg.figures_dir / "head_embed_table.html").write_text(head_embed_table_html)

		classes_html: str = (frontend_resources_path / "classes/index.html").read_text()
		(cfg.figures_dir / "classifications.html").write_text(classes_html)

		# write main index, style.css, and diagram svg
		main_index_html: str = (frontend_resources_path / "index.html").read_text()
		(cfg.figures_dir / "../index.html").write_text(main_index_html)
		style_css: str = (frontend_resources_path / "style.css").read_text()
		(cfg.figures_dir / "../style.css").write_text(style_css)
		diagram_svg: str = (frontend_resources_path / "diagram.svg").read_text()
		(cfg.figures_dir / "../diagram.svg").write_text(diagram_svg)


if __name__ == "__main__":
	import sys

	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	write_frontend(cfg)
