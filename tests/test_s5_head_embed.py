"""Tests for s5_head_embed: add_model_metadata_columns and add_metadata_to_existing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from attention_motifs.pipeline.model_table import ModelInfo
from attention_motifs.pipeline.s5_head_embed import (
	add_metadata_to_existing,
	add_model_metadata_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_MODEL_TABLE: dict[str, ModelInfo] = {
	"gpt2-small": ModelInfo(name="gpt2-small", n_params=85_000_000, n_layers=12, n_heads=12),
	"gpt2-medium": ModelInfo(name="gpt2-medium", n_params=345_000_000, n_layers=24, n_heads=16),
	"pythia-14m": ModelInfo(name="pythia-14m", n_params=14_000_000, n_layers=6, n_heads=8),
}


def _make_df(
	models: list[str] | None = None,
	include_metadata: bool = False,
) -> pl.DataFrame:
	"""Build a minimal head embedding DataFrame for testing."""
	if models is None:
		models = ["gpt2-small", "gpt2-medium", "pythia-14m"]

	rows: list[dict[str, object]] = []
	for model in models:
		info: ModelInfo = MOCK_MODEL_TABLE[model]
		for layer in range(min(info.n_layers, 2)):
			for head in range(min(info.n_heads, 2)):
				row: dict[str, object] = {
					"cls": f"{model}:L{layer}:H{head}",
					"model": model,
					"layer": layer,
					"head": head,
					"type.primary": "test",
					"type.group": "test-group",
					"embed.pca.d2.b4.dim.0": float(layer),
					"embed.pca.d2.b4.dim.1": float(head),
				}
				if include_metadata:
					row["model_family"] = "old_value"
					row["layer_depth"] = -1.0
					row["model_size"] = -1
				rows.append(row)

	return pl.DataFrame(rows)


@pytest.fixture()
def mock_model_table() -> object:
	"""Patch fetch_model_table to return test data without network access."""
	with patch(
		"attention_motifs.pipeline.s5_head_embed.fetch_model_table",
		return_value=MOCK_MODEL_TABLE,
	) as m:
		yield m


# ---------------------------------------------------------------------------
# Tests: add_model_metadata_columns
# ---------------------------------------------------------------------------


class TestAddModelMetadataColumns:
	def test_adds_columns(self, mock_model_table: object) -> None:
		"""Metadata columns are added to a DataFrame that lacks them."""
		df: pl.DataFrame = _make_df()
		result: pl.DataFrame = add_model_metadata_columns(df)

		assert "model_family" in result.columns
		assert "model_size" in result.columns
		assert "layer_depth" in result.columns

	def test_column_order(self, mock_model_table: object) -> None:
		"""Output columns are ordered: base, type.*, embed.*"""
		df: pl.DataFrame = _make_df()
		result: pl.DataFrame = add_model_metadata_columns(df)

		cols: list[str] = result.columns
		# Base columns first
		assert cols[:7] == [
			"cls", "model", "layer", "head",
			"model_family", "layer_depth", "model_size",
		]
		# Then type.* columns
		type_start: int = 7
		assert all(c.startswith("type.") for c in cols[type_start:type_start + 2])
		# Then embed.* columns
		embed_cols: list[str] = [c for c in cols if c.startswith("embed.")]
		assert len(embed_cols) == 2

	def test_model_family_values(self, mock_model_table: object) -> None:
		"""model_family is correctly extracted from model names."""
		df: pl.DataFrame = _make_df()
		result: pl.DataFrame = add_model_metadata_columns(df)

		families: dict[str, str] = dict(
			zip(
				result["model"].to_list(),
				result["model_family"].to_list(),
				strict=False,
			)
		)
		assert families["gpt2-small"] == "gpt2"
		assert families["gpt2-medium"] == "gpt2"
		assert families["pythia-14m"] == "pythia"

	def test_model_size_values(self, mock_model_table: object) -> None:
		"""model_size is correctly looked up from model table."""
		df: pl.DataFrame = _make_df()
		result: pl.DataFrame = add_model_metadata_columns(df)

		sizes: dict[str, int] = dict(
			zip(
				result["model"].to_list(),
				result["model_size"].to_list(),
				strict=False,
			)
		)
		assert sizes["gpt2-small"] == 85_000_000
		assert sizes["gpt2-medium"] == 345_000_000
		assert sizes["pythia-14m"] == 14_000_000

	def test_layer_depth_values(self, mock_model_table: object) -> None:
		"""layer_depth is layer / n_layers."""
		df: pl.DataFrame = _make_df(models=["gpt2-small"])
		result: pl.DataFrame = add_model_metadata_columns(df)

		# gpt2-small has 12 layers; our test data has layers 0 and 1
		depths: list[float] = result["layer_depth"].to_list()
		assert depths[0] == pytest.approx(0.0 / 12)
		assert depths[2] == pytest.approx(1.0 / 12)

	def test_idempotent(self, mock_model_table: object) -> None:
		"""Running twice produces the same columns and values."""
		df: pl.DataFrame = _make_df()
		first: pl.DataFrame = add_model_metadata_columns(df)
		second: pl.DataFrame = add_model_metadata_columns(first)

		assert first.columns == second.columns
		assert first.shape == second.shape
		assert first["model_family"].to_list() == second["model_family"].to_list()
		assert first["model_size"].to_list() == second["model_size"].to_list()

	def test_replaces_existing_columns(self, mock_model_table: object) -> None:
		"""Existing metadata columns are replaced, not duplicated."""
		df: pl.DataFrame = _make_df(include_metadata=True)
		assert df["model_family"].to_list()[0] == "old_value"

		result: pl.DataFrame = add_model_metadata_columns(df)

		# Should have fresh values, not "old_value"
		assert result["model_family"].to_list()[0] != "old_value"
		# No duplicate columns
		assert result.columns.count("model_family") == 1
		assert result.columns.count("model_size") == 1
		assert result.columns.count("layer_depth") == 1


# ---------------------------------------------------------------------------
# Tests: add_metadata_to_existing
# ---------------------------------------------------------------------------


class TestAddMetadataToExisting:
	def test_round_trip(self, mock_model_table: object, tmp_path: Path) -> None:
		"""Write JSONL, add metadata, read back — columns are present."""
		df: pl.DataFrame = _make_df()
		jsonl_path: Path = tmp_path / "head_embed.jsonl"
		df.write_ndjson(jsonl_path)

		add_metadata_to_existing(jsonl_path)

		result: pl.DataFrame = pl.read_ndjson(jsonl_path)
		assert "model_family" in result.columns
		assert "model_size" in result.columns
		assert "layer_depth" in result.columns
		assert result.shape[0] == df.shape[0]

	def test_idempotent_file(self, mock_model_table: object, tmp_path: Path) -> None:
		"""Running add_metadata_to_existing twice keeps column count stable."""
		df: pl.DataFrame = _make_df()
		jsonl_path: Path = tmp_path / "head_embed.jsonl"
		df.write_ndjson(jsonl_path)

		add_metadata_to_existing(jsonl_path)
		cols_after_first: int = pl.read_ndjson(jsonl_path).shape[1]

		add_metadata_to_existing(jsonl_path)
		cols_after_second: int = pl.read_ndjson(jsonl_path).shape[1]

		assert cols_after_first == cols_after_second
