"""Pipeline step 7b: Deploy ablation frontend HTML (and libs) to data/ablations/."""

import sys
from pathlib import Path

from attention_motifs.ablation.frontend import deploy_ablation_frontend
from attention_motifs.pipeline.cfg import PipelineConfig, pipeline_step_major


def write_ablation_frontend(cfg: PipelineConfig) -> None:
	"""Copy ablation frontend HTML and chart.min.js to the ablation output dir.

	Reads ``output_dir`` from ``cfg.raw_cfg["ablation"]`` (default
	``data/ablations``).  Fixes the ``chart.min.js`` script path for the
	ablation deploy location and copies the library to ``data/libs/``.
	"""
	pipeline_step_major("pipeline step 7b: write ablation frontend")

	ablation_dict: dict = cfg.ablation
	output_dir: Path = Path(ablation_dict.get("output_dir", "data/ablations"))

	deploy_ablation_frontend(output_dir, verbose=cfg.verbose)


if __name__ == "__main__":
	cfg: PipelineConfig = PipelineConfig.from_cli(sys.argv[1:])
	write_ablation_frontend(cfg)
