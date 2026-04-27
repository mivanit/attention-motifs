"""Tests for add_pattern_metadata_columns and add_metadata_to_pattern_files."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from attention_motifs.features.feature_table import add_pattern_metadata_columns
from attention_motifs.pipeline.model_table import ModelInfo
from attention_motifs.pipeline.s3_feat_proc import add_metadata_to_pattern_files


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_MODEL_TABLE: dict[str, ModelInfo] = {
	"gpt2-small": ModelInfo(
		name="gpt2-small", n_params=85_000_000, n_layers=12, n_heads=12
	),
	"gpt2-medium": ModelInfo(
		name="gpt2-medium", n_params=345_000_000, n_layers=24, n_heads=16
	),
	"pythia-14m": ModelInfo(
		name="pythia-14m", n_params=14_000_000, n_layers=6, n_heads=8
	),
}


def _make_pattern_df(
	models: list[str] | None = None,
	include_metadata: bool = False,
) -> pl.DataFrame:
	"""Build a minimal pattern embedding DataFrame for testing."""
	if models is None:
		models = ["gpt2-small", "gpt2-medium", "pythia-14m"]

	rows: list[dict[str, object]] = []
	for model in models:
		info: ModelInfo = MOCK_MODEL_TABLE[model]
		for layer in range(min(info.n_layers, 2)):
			for head in range(min(info.n_heads, 2)):
				row: dict[str, object] = {
					"activation.model": model,
					"activation.layer": str(layer),
					"activation.head": head,
					"activation.cls": f"{model}:L{layer}:H{head}",
					"activation.prompt": "test_prompt_hash",
					"activation.n_ctx": 128,
					"activation.cache_key": f"blocks.{layer}.attn.hook_pattern",
					"activation.layer_depth": float(layer) / float(info.n_layers - 1),
					"pc.0": float(layer),
					"pc.1": float(head),
				}
				if include_metadata:
					row["activation.model_family"] = "old_value"
					row["activation.model_size"] = -1
				rows.append(row)

	return pl.DataFrame(rows)


@pytest.fixture()
def mock_model_table() -> object:
	"""Patch fetch_model_table to return test data without network access."""
	with patch(
		"attention_motifs.pipeline.model_table.fetch_model_table",
		return_value=MOCK_MODEL_TABLE,
	) as m:
		yield m


# ---------------------------------------------------------------------------
# Tests: add_pattern_metadata_columns
# ---------------------------------------------------------------------------


class TestAddPatternMetadataColumns:
	def test_adds_columns(self, mock_model_table: object) -> None:
		"""Metadata columns are added to a DataFrame that lacks them."""
		df: pl.DataFrame = _make_pattern_df()
		result: pl.DataFrame = add_pattern_metadata_columns(df)

		assert "activation.model_family" in result.columns
		assert "activation.model_size" in result.columns

	def test_preserves_existing_columns(self, mock_model_table: object) -> None:
		"""Original activation.* and pc.* columns are preserved."""
		df: pl.DataFrame = _make_pattern_df()
		result: pl.DataFrame = add_pattern_metadata_columns(df)

		for col in (
			"activation.model",
			"activation.layer",
			"activation.head",
			"activation.cls",
			"activation.prompt",
			"activation.n_ctx",
			"activation.layer_depth",
			"pc.0",
			"pc.1",
		):
			assert col in result.columns, f"Missing column: {col}"

	def test_model_family_values(self, mock_model_table: object) -> None:
		"""activation.model_family is correctly extracted from model names."""
		df: pl.DataFrame = _make_pattern_df()
		result: pl.DataFrame = add_pattern_metadata_columns(df)

		families: dict[str, str] = dict(
			zip(
				result["activation.model"].to_list(),
				result["activation.model_family"].to_list(),
				strict=False,
			)
		)
		assert families["gpt2-small"] == "gpt2"
		assert families["gpt2-medium"] == "gpt2"
		assert families["pythia-14m"] == "pythia"

	def test_model_size_values(self, mock_model_table: object) -> None:
		"""activation.model_size is correctly looked up from model table."""
		df: pl.DataFrame = _make_pattern_df()
		result: pl.DataFrame = add_pattern_metadata_columns(df)

		sizes: dict[str, int] = dict(
			zip(
				result["activation.model"].to_list(),
				result["activation.model_size"].to_list(),
				strict=False,
			)
		)
		assert sizes["gpt2-small"] == 85_000_000
		assert sizes["gpt2-medium"] == 345_000_000
		assert sizes["pythia-14m"] == 14_000_000

	def test_idempotent(self, mock_model_table: object) -> None:
		"""Running twice produces the same columns and values."""
		df: pl.DataFrame = _make_pattern_df()
		first: pl.DataFrame = add_pattern_metadata_columns(df)
		second: pl.DataFrame = add_pattern_metadata_columns(first)

		assert first.columns == second.columns
		assert first.shape == second.shape
		assert (
			first["activation.model_family"].to_list()
			== second["activation.model_family"].to_list()
		)
		assert (
			first["activation.model_size"].to_list()
			== second["activation.model_size"].to_list()
		)

	def test_replaces_existing_columns(self, mock_model_table: object) -> None:
		"""Existing metadata columns are replaced, not duplicated."""
		df: pl.DataFrame = _make_pattern_df(include_metadata=True)
		assert df["activation.model_family"].to_list()[0] == "old_value"

		result: pl.DataFrame = add_pattern_metadata_columns(df)

		# Should have fresh values, not "old_value"
		assert result["activation.model_family"].to_list()[0] != "old_value"
		# No duplicate columns
		assert result.columns.count("activation.model_family") == 1
		assert result.columns.count("activation.model_size") == 1

	def test_unknown_model_gets_unknown_family(self, mock_model_table: object) -> None:
		"""Models not in get_model_family heuristics get 'unknown'."""
		# Add a row with a model name that doesn't match any known family
		df: pl.DataFrame = pl.DataFrame(
			[
				{
					"activation.model": "mystery-model-7b",
					"activation.layer": "0",
					"activation.head": 0,
					"activation.cls": "mystery-model-7b:L0:H0",
					"activation.prompt": "test",
					"activation.n_ctx": 64,
					"activation.layer_depth": 0.0,
					"pc.0": 1.0,
				}
			]
		)
		result: pl.DataFrame = add_pattern_metadata_columns(df)
		assert result["activation.model_family"].to_list()[0] == "unknown"
		assert result["activation.model_size"].to_list()[0] is None

	def test_row_count_unchanged(self, mock_model_table: object) -> None:
		"""Adding metadata does not change the number of rows."""
		df: pl.DataFrame = _make_pattern_df()
		result: pl.DataFrame = add_pattern_metadata_columns(df)
		assert result.shape[0] == df.shape[0]


# ---------------------------------------------------------------------------
# Tests: add_metadata_to_pattern_files
# ---------------------------------------------------------------------------


class TestAddMetadataToPatternFiles:
	def test_patches_jsonl(self, mock_model_table: object, tmp_path: Path) -> None:
		"""JSONL files get metadata columns added."""
		df: pl.DataFrame = _make_pattern_df()
		jsonl_path: Path = tmp_path / "raw.jsonl"
		df.write_ndjson(jsonl_path)

		add_metadata_to_pattern_files(tmp_path)

		result: pl.DataFrame = pl.read_ndjson(jsonl_path)
		assert "activation.model_family" in result.columns
		assert "activation.model_size" in result.columns
		assert result.shape[0] == df.shape[0]

	def test_patches_csv(self, mock_model_table: object, tmp_path: Path) -> None:
		"""CSV files get metadata columns added."""
		df: pl.DataFrame = _make_pattern_df()
		csv_path: Path = tmp_path / "pca.csv"
		df.write_csv(csv_path, float_precision=6)

		add_metadata_to_pattern_files(tmp_path)

		result: pl.DataFrame = pl.read_csv(csv_path)
		assert "activation.model_family" in result.columns
		assert "activation.model_size" in result.columns
		assert result.shape[0] == df.shape[0]

	def test_patches_parquet(self, mock_model_table: object, tmp_path: Path) -> None:
		"""Parquet files get metadata columns added."""
		df: pl.DataFrame = _make_pattern_df()
		parquet_path: Path = tmp_path / "pca.parquet"
		df.write_parquet(parquet_path)

		add_metadata_to_pattern_files(tmp_path)

		result: pl.DataFrame = pl.read_parquet(parquet_path)
		assert "activation.model_family" in result.columns
		assert "activation.model_size" in result.columns
		assert result.shape[0] == df.shape[0]

	def test_skips_missing_files(
		self, mock_model_table: object, tmp_path: Path
	) -> None:
		"""Does not error when expected files are absent."""
		# Empty directory — should just print skip messages, not raise
		add_metadata_to_pattern_files(tmp_path)

	def test_idempotent_file(self, mock_model_table: object, tmp_path: Path) -> None:
		"""Running twice keeps column count stable."""
		df: pl.DataFrame = _make_pattern_df()
		jsonl_path: Path = tmp_path / "pca.jsonl"
		df.write_ndjson(jsonl_path)

		add_metadata_to_pattern_files(tmp_path)
		cols_after_first: int = pl.read_ndjson(jsonl_path).shape[1]

		add_metadata_to_pattern_files(tmp_path)
		cols_after_second: int = pl.read_ndjson(jsonl_path).shape[1]

		assert cols_after_first == cols_after_second
