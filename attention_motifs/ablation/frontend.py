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
				"calibration_prompts_file": exp_results.config.calibration_prompts_file,
				"seed": exp_results.config.seed,
			},
			"baseline_loss": exp_results.baseline_loss,
			"baseline_icl": exp_results.baseline_icl,
			"results": [r.serialize() for r in exp_results.results],
		}
	return {"models": models}


def write_ablation_frontend(
	all_results: dict[str, AblationResults],
	output_dir: Path | str,
) -> Path:
	"""Write ablation frontend HTML and results data to *output_dir*.

	Copies the bundled ``index.html`` and writes serialized results as
	a separate ``ablation_results.json`` file.  The frontend fetches
	the JSON at runtime.

	Parameters
	----------
	all_results
	    Mapping of model_name to ExperimentResults.
	output_dir
	    Directory to write files into.

	Returns
	-------
	Path
	    Path to the written ``index.html``.
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Copy bundled HTML
	html: str = _get_bundled_html()
	html_path: Path = output_dir / "index.html"
	html_path.write_text(html)

	# Write data JSON
	data: dict[str, Any] = _serialize_results(all_results)
	data_path: Path = output_dir / "ablation_results.json"
	data_path.write_text(json.dumps(data))

	print(f"Wrote ablation frontend to {html_path}")
	print(f"Wrote ablation data to {data_path}")
	return html_path
