"""Pipeline step 6c: Write clustering and cluster-trends frontends to data/vis/."""

import importlib.resources
import sys
from pathlib import Path

import attention_motifs
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def write_cluster_frontend(cfg: PipelineConfig) -> None:
	"""Copy clustering and cluster-trends frontend HTML (and libs) to vis/.

	Writes:
	- clustering/index.html  → cfg.vis_dir / "clustering/"
	- cluster_trends/index.html → cfg.vis_dir / "cluster_trends/"
	- libs/d3.min.js → cfg.vis_dir.parent / "libs/"
	"""
	pipeline_step_major("pipeline step 6c: write cluster frontends")

	frontend_resources_path: Path = Path(
		importlib.resources.files(attention_motifs).joinpath("frontend"),  # type: ignore[arg-type]
	)

	# clustering frontend
	clustering_html: str = (
		frontend_resources_path / "clustering/index.html"
	).read_text()
	clustering_dir: Path = cfg.vis_dir / "clustering"
	clustering_dir.mkdir(parents=True, exist_ok=True)
	(clustering_dir / "index.html").write_text(clustering_html)

	# cluster_trends frontend
	trends_html: str = (
		frontend_resources_path / "cluster_trends/index.html"
	).read_text()
	trends_dir: Path = cfg.vis_dir / "cluster_trends"
	trends_dir.mkdir(parents=True, exist_ok=True)
	(trends_dir / "index.html").write_text(trends_html)

	# d3.min.js (too large for bundler to inline)
	libs_src: Path = frontend_resources_path / "libs" / "d3.min.js"
	libs_dst: Path = cfg.vis_dir.parent / "libs" / "d3.min.js"
	libs_dst.parent.mkdir(parents=True, exist_ok=True)
	libs_dst.write_bytes(libs_src.read_bytes())

	if cfg.verbose > 0:
		print(f"Wrote clustering frontend to {clustering_dir}")
		print(f"Wrote cluster_trends frontend to {trends_dir}")
		print(f"Wrote d3.min.js to {libs_dst}")


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	write_cluster_frontend(cfg)
