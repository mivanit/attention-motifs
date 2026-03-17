"""Tests for string parsers and utility functions across modules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from attention_motifs.attnpedia.attnpedia import parse_cls, parse_head
from attention_motifs.pipeline.cfg import PipelineConfig, deep_merge_dicts
from attention_motifs.pipeline.s3_feat_proc import _sample_prompts
from attention_motifs.pipeline.s5_head_embed import (
	create_plots_metadata,
	get_embedding_prefixes,
	parse_prefix_info,
)
from attention_motifs.util.util import prefix_dict


# ===========================================================================
# attnpedia parsers
# ===========================================================================


class TestParseHead:
	def test_two_parts(self) -> None:
		"""'L3:H7' → (3, 7)."""
		assert parse_head("L3:H7") == (3, 7)

	def test_three_parts(self) -> None:
		"""'gpt2:L3:H7' → (3, 7), model prefix stripped."""
		assert parse_head("gpt2:L3:H7") == (3, 7)

	def test_zero_indexed(self) -> None:
		assert parse_head("L0:H0") == (0, 0)

	def test_invalid_format(self) -> None:
		with pytest.raises(ValueError, match="Invalid head string"):
			parse_head("L3")

	def test_too_many_parts(self) -> None:
		with pytest.raises(ValueError, match="Invalid head string"):
			parse_head("a:b:c:d")


class TestParseCls:
	def test_basic(self) -> None:
		"""'gpt2:L3:H7' → ('gpt2', 3, 7)."""
		assert parse_cls("gpt2:L3:H7") == ("gpt2", 3, 7)

	def test_invalid_two_parts(self) -> None:
		with pytest.raises(ValueError):
			parse_cls("L3:H7")


# ===========================================================================
# s5_head_embed parsers
# ===========================================================================


class TestParsePrefix:
	def test_valid(self) -> None:
		result: dict[str, str] = parse_prefix_info("embed.umap.d2.b16")
		assert result == {"method": "umap", "n_components": "2", "n_neighbors": "16"}

	def test_pca(self) -> None:
		result: dict[str, str] = parse_prefix_info("embed.pca.d3.b0")
		assert result["method"] == "pca"
		assert result["n_components"] == "3"

	def test_invalid_format(self) -> None:
		with pytest.raises(ValueError, match="Invalid prefix"):
			parse_prefix_info("bad.prefix")

	def test_invalid_not_embed(self) -> None:
		with pytest.raises(ValueError, match="Invalid prefix"):
			parse_prefix_info("other.umap.d2.b16")


class TestGetEmbeddingPrefixes:
	def test_extracts_unique_prefixes(self) -> None:
		df: pl.DataFrame = pl.DataFrame(
			{
				"embed.umap.d2.b16.dim.0": [1.0],
				"embed.umap.d2.b16.dim.1": [2.0],
				"embed.pca.d3.b0.dim.0": [3.0],
				"embed.pca.d3.b0.dim.1": [4.0],
				"embed.pca.d3.b0.dim.2": [5.0],
				"other_col": [6.0],
			}
		)
		prefixes: list[str] = get_embedding_prefixes(df)
		assert prefixes == ["embed.pca.d3.b0", "embed.umap.d2.b16"]

	def test_no_embed_columns(self) -> None:
		df: pl.DataFrame = pl.DataFrame({"col_a": [1], "col_b": [2]})
		assert get_embedding_prefixes(df) == []


class TestCreatePlotsMetadata:
	def test_basic(self) -> None:
		prefixes: list[str] = ["embed.umap.d2.b16", "embed.pca.d3.b0"]
		result: dict = create_plots_metadata(prefixes)

		assert result["metadata"]["total_plots"] == 2
		assert len(result["plots"]) == 2
		assert set(result["metadata"]["methods"]) == {"umap", "pca"}


# ===========================================================================
# util.util
# ===========================================================================


class TestPrefixDict:
	def test_string_prefix(self) -> None:
		result: dict[str, int] = prefix_dict({"a": 1, "b": 2}, "x")
		assert result == {"x.a": 1, "x.b": 2}

	def test_list_prefix(self) -> None:
		result: dict[str, int] = prefix_dict({"a": 1}, ["x", "y"])
		assert result == {"x.y.a": 1}

	def test_custom_separator(self) -> None:
		result: dict[str, int] = prefix_dict({"a": 1}, "x", sep="/")
		assert result == {"x/a": 1}

	def test_empty_dict(self) -> None:
		result: dict = prefix_dict({}, "x")
		assert result == {}


# ===========================================================================
# cfg.py utilities
# ===========================================================================


class TestDeepMergeDicts:
	def test_basic_merge(self) -> None:
		result: dict = deep_merge_dicts({"a": 1}, {"b": 2})
		assert result == {"a": 1, "b": 2}

	def test_overwrite(self) -> None:
		result: dict = deep_merge_dicts({"a": 1}, {"a": 2})
		assert result == {"a": 2}

	def test_nested(self) -> None:
		d1: dict = {"x": {"a": 1, "b": 2}}
		d2: dict = {"x": {"b": 3, "c": 4}}
		result: dict = deep_merge_dicts(d1, d2)
		assert result == {"x": {"a": 1, "b": 3, "c": 4}}

	def test_no_mutation(self) -> None:
		d1: dict = {"x": {"a": 1}}
		d2: dict = {"x": {"b": 2}}
		d1_copy: dict = deepcopy(d1)
		d2_copy: dict = deepcopy(d2)
		deep_merge_dicts(d1, d2)
		assert d1 == d1_copy
		assert d2 == d2_copy

	def test_non_dict_overwrites_dict(self) -> None:
		result: dict = deep_merge_dicts({"x": {"a": 1}}, {"x": 99})
		assert result == {"x": 99}


class TestComputeHash:
	def test_stable(self) -> None:
		"""Same config → same hash."""
		cfg: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		h1: str = cfg.compute_hash()
		h2: str = cfg.compute_hash()
		assert h1 == h2
		assert len(h1) == 64  # SHA-256 hex

	def test_different_models_different_hash(self) -> None:
		cfg1: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		cfg2: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		cfg2.models = ["different-model"]
		assert cfg1.compute_hash() != cfg2.compute_hash()


# ===========================================================================
# s3 web PCA sampling
# ===========================================================================


class TestSamplePrompts:
	"""Tests for _sample_prompts used to create the reduced pca_web.csv."""

	@pytest.fixture()
	def df(self) -> pl.DataFrame:
		"""3 models × 5 prompts × 2 heads = 30 rows."""
		rows: list[dict[str, str | float]] = []
		for model in ["model-a", "model-b", "model-c"]:
			for prompt in ["p1", "p2", "p3", "p4", "p5"]:
				for head in [0, 1]:
					rows.append(
						{
							"activation.model": model,
							"activation.prompt": prompt,
							"activation.head": float(head),
							"pc.0": 1.0,
						}
					)
		return pl.DataFrame(rows)

	def test_samples_correct_prompt_count(self, df: pl.DataFrame) -> None:
		result: pl.DataFrame = _sample_prompts(df, n_prompts=2, seed=42)
		prompts: list[str] = result["activation.prompt"].unique().sort().to_list()
		assert len(prompts) == 2

	def test_same_prompts_across_models(self, df: pl.DataFrame) -> None:
		result: pl.DataFrame = _sample_prompts(df, n_prompts=2, seed=42)
		prompts_per_model: dict[str, set[str]] = {}
		for model in result["activation.model"].unique().to_list():
			model_prompts: set[str] = set(
				result.filter(pl.col("activation.model") == model)[
					"activation.prompt"
				].to_list()
			)
			prompts_per_model[model] = model_prompts
		# all models should have the exact same prompt set
		values: list[set[str]] = list(prompts_per_model.values())
		assert all(v == values[0] for v in values)

	def test_all_heads_preserved(self, df: pl.DataFrame) -> None:
		"""Each prompt should keep all its heads (2 per model)."""
		result: pl.DataFrame = _sample_prompts(df, n_prompts=2, seed=42)
		# 2 prompts × 3 models × 2 heads = 12 rows
		assert len(result) == 12

	def test_deterministic(self, df: pl.DataFrame) -> None:
		r1: pl.DataFrame = _sample_prompts(df, n_prompts=2, seed=42)
		r2: pl.DataFrame = _sample_prompts(df, n_prompts=2, seed=42)
		assert r1.equals(r2)

	def test_different_seed_different_result(self, df: pl.DataFrame) -> None:
		r1: pl.DataFrame = _sample_prompts(df, n_prompts=2, seed=42)
		r2: pl.DataFrame = _sample_prompts(df, n_prompts=2, seed=99)
		p1: list[str] = sorted(r1["activation.prompt"].unique().to_list())
		p2: list[str] = sorted(r2["activation.prompt"].unique().to_list())
		# with 5 prompts and sample of 2, different seeds should (very likely) differ
		assert p1 != p2

	def test_n_prompts_exceeds_available(self, df: pl.DataFrame) -> None:
		"""If n_prompts >= available prompts, return all rows unchanged."""
		result: pl.DataFrame = _sample_prompts(df, n_prompts=100, seed=42)
		assert len(result) == len(df)

	def test_config_defaults_none(self) -> None:
		"""web_pca_n_prompts defaults to None when not in TOML."""
		cfg: PipelineConfig = PipelineConfig.read(Path("tests/pipeline_cfg_test.toml"))
		assert cfg.web_pca_n_prompts is None
		assert cfg.web_pca_seed == 42
