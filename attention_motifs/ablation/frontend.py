"""Write ablation results frontend: HTML + data JSON."""

import importlib.resources
import json
from pathlib import Path
from typing import Any

import attention_motifs
from attention_motifs.ablation.experiment import AblationResults


def _get_bundled_html() -> str:
	"""Load the bundled ablation frontend HTML from package resources.

	Returns
	-------
	str
	    Bundled HTML string.
	"""
	frontend_resources_path: Path = Path(
		importlib.resources.files(attention_motifs).joinpath("frontend"),  # type: ignore[arg-type]
	)
	html_path: Path = frontend_resources_path / "ablation" / "index.html"
	return html_path.read_text()


def _serialize_results(
	all_results: dict[str, AblationResults],
) -> dict[str, Any]:
	"""Serialize multiple AblationResults into a JSON-compatible dict.

	Parameters
	----------
	all_results
	    Mapping of model_name to AblationResults.

	Returns
	-------
	dict[str, Any]
	    ``{"models": {"model_name": {...}, ...}}``
	"""
	models: dict[str, Any] = {
		model_name: exp_results.serialize()
		for model_name, exp_results in all_results.items()
	}
	return {"models": models}


def deploy_ablation_frontend(
	output_dir: Path | str,
	verbose: int = 1,
) -> Path:
	"""Copy bundled ablation HTML and libs to *output_dir*.

	Fixes the ``d3.min.js`` path for the ablation deploy location
	(depth 1 under ``data/``, not depth 2 like ``data/vis/X/``), and
	copies the library file to ``output_dir.parent / "libs/"``.

	Parameters
	----------
	output_dir
	    Directory to write ``index.html`` into (e.g. ``data/ablations``).
	verbose
	    Print output paths when > 0.

	Returns
	-------
	Path
	    Path to the written ``index.html``.
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Copy bundled HTML (fix d3.min.js path for runtime location)
	# The bundler keeps ../../libs/d3.min.js (build-time path), but at
	# runtime the HTML is at data/ablations/ (depth 1), not data/vis/X/ (depth 2),
	# so the correct runtime path is ../libs/d3.min.js → data/libs/.
	html: str = _get_bundled_html()
	html = html.replace(
		'src="../../libs/d3.min.js"',
		'src="../libs/d3.min.js"',
	)
	html_path: Path = output_dir / "index.html"
	html_path.write_text(html)

	# Ensure d3.min.js is available at data/libs/
	frontend_resources_path: Path = Path(
		importlib.resources.files(attention_motifs).joinpath("frontend"),  # type: ignore[arg-type]
	)
	libs_src: Path = frontend_resources_path / "libs" / "d3.min.js"
	libs_dst: Path = output_dir.parent / "libs" / "d3.min.js"
	libs_dst.parent.mkdir(parents=True, exist_ok=True)
	libs_dst.write_bytes(libs_src.read_bytes())

	if verbose > 0:
		print(f"Wrote ablation frontend to {html_path}")
		print(f"Wrote d3.min.js to {libs_dst}")

	return html_path


def write_ablation_frontend(
	all_results: dict[str, AblationResults],
	output_dir: Path | str,
) -> Path:
	"""Write ablation frontend HTML and results data to *output_dir*.

	Copies the bundled ``index.html`` (with path fixes) and writes
	serialized results as a separate ``ablation_results.json`` file.
	The frontend fetches the JSON at runtime.

	Parameters
	----------
	all_results
	    Mapping of model_name to AblationResults.
	output_dir
	    Directory to write files into.

	Returns
	-------
	Path
	    Path to the written ``index.html``.
	"""
	output_dir = Path(output_dir)

	# Deploy HTML + libs
	html_path: Path = deploy_ablation_frontend(output_dir, verbose=0)

	# Write data JSON
	data: dict[str, Any] = _serialize_results(all_results)
	data_path: Path = output_dir / "ablation_results.json"
	data_path.write_text(json.dumps(data))

	print(f"Wrote ablation frontend to {html_path}")
	print(f"Wrote ablation data to {data_path}")
	return html_path
