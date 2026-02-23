"""Fetch and cache TransformerLens model parameter table from GitHub.

Used by the parallel model scheduler to estimate VRAM requirements
based on parameter counts.
"""

import csv
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODEL_TABLE_URL: str = "https://raw.githubusercontent.com/mivanit/transformerlens-model-table/main/docs/model_table.csv"
MODEL_TABLE_CACHE: Path = (
	Path.home() / ".cache" / "attention_motifs" / "model_table.csv"
)


@dataclass(frozen=True)
class ModelInfo:
	"""Basic model metadata from the TransformerLens model table."""

	name: str
	n_params: int


def _download_csv(url: str, cache_path: Path) -> str:
	"""Download CSV from URL and save to cache path.

	Returns the CSV content as a string.
	"""
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	with urllib.request.urlopen(url) as response:  # noqa: S310
		content: str = response.read().decode("utf-8")
	cache_path.write_text(content)
	return content


def _parse_csv(content: str) -> dict[str, ModelInfo]:
	"""Parse model table CSV into a dict keyed by model name."""
	reader: csv.DictReader = csv.DictReader(io.StringIO(content))
	table: dict[str, ModelInfo] = {}
	for row in reader:
		name: str = row["name.default_alias"]
		n_params_str: str = row["n_params.as_int"]
		if not name or not n_params_str:
			continue
		n_params: int = int(n_params_str)
		table[name] = ModelInfo(name=name, n_params=n_params)
	return table


def fetch_model_table(force_refresh: bool = False) -> dict[str, ModelInfo]:
	"""Fetch model table from GitHub, using local cache if available.

	Downloads the CSV on first call, then reads from cache on subsequent calls.
	Pass ``force_refresh=True`` to re-download.
	"""
	content: str
	if MODEL_TABLE_CACHE.exists() and not force_refresh:
		content = MODEL_TABLE_CACHE.read_text()
	else:
		print(f"Downloading model table from {MODEL_TABLE_URL}")
		content = _download_csv(MODEL_TABLE_URL, MODEL_TABLE_CACHE)
		print(f"Cached model table to {MODEL_TABLE_CACHE}")
	return _parse_csv(content)


def get_model_params(model_name: str, table: dict[str, ModelInfo]) -> int:
	"""Look up parameter count for a model by name.

	Raises ``KeyError`` if the model is not found in the table.
	"""
	if model_name in table:
		return table[model_name].n_params
	raise KeyError(
		f"Model {model_name!r} not found in model table. "
		f"Available models: {sorted(table.keys())}"
	)
