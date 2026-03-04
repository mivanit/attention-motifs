"""Write self-contained ablation results HTML."""

import importlib.resources
import json
from pathlib import Path
from typing import Any

import attention_motifs
from attention_motifs.ablation.experiment import ExperimentResults


_DATA_PLACEHOLDER: str = "__ABLATION_DATA__"


def _get_bundled_html() -> str:
	"""Load the bundled ablation frontend HTML from package resources.

	Returns
	-------
	str
	    Bundled HTML string with ``__ABLATION_DATA__`` placeholder.
	"""
	frontend_resources_path: Path = Path(
		importlib.resources.files(attention_motifs).joinpath("frontend"),  # type: ignore[arg-type]
	)
	html_path: Path = frontend_resources_path / "ablation" / "index.html"
	return html_path.read_text()


def _serialize_results(
	all_results: dict[str, ExperimentResults],
) -> dict[str, Any]:
	"""Serialize multiple ExperimentResults into a JSON-compatible dict.

	Parameters
	----------
	all_results
	    Mapping of model_name to ExperimentResults.

	Returns
	-------
	dict[str, Any]
	    ``{"models": {"model_name": {...}, ...}}``
	"""
	models: dict[str, dict[str, Any]] = {}
	for model_name, exp_results in all_results.items():
		models[model_name] = {
			"model_name": exp_results.model_name,
			"config": {
				"n_sequences": exp_results.config.n_sequences,
				"seq_length": exp_results.config.seq_length,
				"n_repetitions": exp_results.config.n_repetitions,
				"ablation_methods": [
					m.value for m in exp_results.config.ablation_methods
				],
				"n_calibration_prompts": exp_results.config.n_calibration_prompts,
				"seed": exp_results.config.seed,
			},
			"baseline_loss": exp_results.baseline_loss,
			"baseline_icl": exp_results.baseline_icl,
			"results": [r.serialize() for r in exp_results.results],
		}
	return {"models": models}


def write_ablation_frontend(
	all_results: dict[str, ExperimentResults],
	output_dir: Path | str,
	filename: str = "ablation_results.html",
) -> Path:
	"""Write a self-contained HTML report of ablation results.

	Reads the bundled HTML template, embeds serialized results by
	replacing the ``__ABLATION_DATA__`` placeholder, and writes the
	resulting file to *output_dir*.

	Parameters
	----------
	all_results
	    Mapping of model_name to ExperimentResults.
	output_dir
	    Directory to write the HTML file.
	filename
	    Name of the output HTML file.

	Returns
	-------
	Path
	    Path to the written HTML file.
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	html: str = _get_bundled_html()
	data: dict[str, Any] = _serialize_results(all_results)
	data_json: str = json.dumps(data)

	if _DATA_PLACEHOLDER not in html:
		raise ValueError(
			f"Bundled HTML does not contain placeholder '{_DATA_PLACEHOLDER}'. "
			"Run 'make am-frontend-bundle' to rebuild the frontend."
		)

	html = html.replace(_DATA_PLACEHOLDER, data_json)

	output_path: Path = output_dir / filename
	output_path.write_text(html)
	print(f"Wrote ablation frontend to {output_path}")
	return output_path
