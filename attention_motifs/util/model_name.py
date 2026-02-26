"""Disk-backed cache for model name resolution and sanitization.

Caches both :func:`resolve_model_name` (any variant → default alias) and
:func:`sanitize_model_name` (any variant → filesystem-safe name) from
``pattern_lens`` so that the heavy import is only needed on cache misses.

CLI usage::

    python -m attention_motifs.util.model_name config.toml
    python -m attention_motifs.util.model_name --cache-path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from attention_motifs.util.cache_path import resolve_cache_path

_CACHE_FILENAME: str = "model_name_map.json"


class _CacheEntry(TypedDict):
	original: str
	resolved: str
	sanitized: str
	hf_repo_id: str
	all_match: bool


# Old cache entries may lack hf_repo_id; accept entries with at least these keys.
_CACHE_ENTRY_KEYS_REQUIRED: frozenset[str] = frozenset(
	{"original", "resolved", "sanitized", "all_match"}
)


_cache_path: Path | None = None
_cache: dict[str, _CacheEntry] | None = None


def _get_cache_path() -> Path:
	"""Lazy-init and return the cache file path."""
	global _cache_path  # noqa: PLW0603
	if _cache_path is None:
		_cache_path = resolve_cache_path(_CACHE_FILENAME)
	return _cache_path


def _get_cache() -> dict[str, _CacheEntry]:
	"""Lazy-load the disk cache into memory.

	Old cache entries missing ``hf_repo_id`` are accepted and will be
	backfilled lazily in :func:`_ensure_entry`.

	Raises
	------
	ValueError
		If the cache file exists but contains malformed entries.
	"""
	global _cache  # noqa: PLW0603
	if _cache is None:
		path: Path = _get_cache_path()
		if path.exists():
			raw: dict[str, object] = json.loads(path.read_text())
			for key, entry in raw.items():
				if (
					not isinstance(entry, dict)
					or not _CACHE_ENTRY_KEYS_REQUIRED.issubset(entry.keys())
				):
					msg: str = (
						f"Corrupt model name cache at {path}: "
						f"entry {key!r} has keys {set(entry.keys()) if isinstance(entry, dict) else type(entry).__name__}, "
						f"expected at least {_CACHE_ENTRY_KEYS_REQUIRED}. "
						f"Delete the file or re-run with --refresh."
					)
					raise ValueError(msg)
			_cache = raw  # type: ignore[assignment]
		else:
			_cache = {}
	assert _cache is not None
	return _cache


def _persist_cache() -> None:
	"""Write the in-memory cache to disk."""
	cache: dict[str, _CacheEntry] = _get_cache()
	path: Path = _get_cache_path()
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(cache, indent=2))


def _resolve_hf_repo_id(name: str) -> str:
	"""Resolve *name* to a HuggingFace Hub repo ID.

	Uses TransformerLens's alias map.  Falls back to *name* unchanged
	if it is not a known TransformerLens alias (it may already be a
	valid HF repo ID like ``meta-llama/Llama-3.2-1B``).
	"""
	from transformer_lens.loading_from_pretrained import (  # noqa: PLC0415
		get_official_model_name,
	)

	try:
		return get_official_model_name(name)
	except ValueError:
		return name


def _ensure_entry(name: str) -> _CacheEntry:
	"""Return the cache entry for *name*, computing on miss.

	Old cache entries missing ``hf_repo_id`` are backfilled in-place.
	"""
	cache: dict[str, _CacheEntry] = _get_cache()
	if name in cache:
		entry: _CacheEntry = cache[name]
		if "hf_repo_id" not in entry:  # type: ignore[operator]
			entry["hf_repo_id"] = _resolve_hf_repo_id(entry["original"])
			_persist_cache()
		return entry

	from pattern_lens.consts import sanitize_name_str  # noqa: PLC0415
	from pattern_lens.load_model import resolve_model_name  # noqa: PLC0415

	resolved: str = resolve_model_name(name)
	sanitized: str = sanitize_name_str(resolved)
	hf_repo_id: str = _resolve_hf_repo_id(name)
	entry = _CacheEntry(
		original=name,
		resolved=resolved,
		sanitized=sanitized,
		hf_repo_id=hf_repo_id,
		all_match=(name == resolved == sanitized),
	)
	# store under all variants so lookups by sanitized name also hit
	for key in {name, resolved, sanitized}:
		cache[key] = entry
	_persist_cache()
	return entry


def cached_resolve_model_name(name: str) -> str:
	"""Return the TransformerLens default alias for *name* (cached)."""
	return _ensure_entry(name)["resolved"]


def cached_sanitize_model_name(name: str) -> str:
	"""Return the filesystem-safe form of *name* (cached)."""
	return _ensure_entry(name)["sanitized"]


def cached_get_hf_repo_id(name: str) -> str:
	"""Return the HuggingFace Hub repo ID for *name* (cached).

	This is the ``repo_id`` that can be passed to
	``huggingface_hub.snapshot_download(repo_id=...)``.
	"""
	return _ensure_entry(name)["hf_repo_id"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
	"""Warm the model-name cache for all models in a pipeline config."""
	import argparse  # noqa: PLC0415
	import tomllib  # noqa: PLC0415

	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Warm the model-name resolve/sanitize cache from a pipeline config.",
	)
	parser.add_argument(
		"config",
		nargs="?",
		type=Path,
		help="Path to a pipeline TOML config file.",
	)
	parser.add_argument(
		"--cache-path",
		action="store_true",
		help="Print the resolved cache file path and exit.",
	)
	parser.add_argument(
		"-f",
		"--refresh",
		action="store_true",
		help="Delete existing cache before warming.",
	)
	args: argparse.Namespace = parser.parse_args()

	if args.cache_path:
		print(_get_cache_path())
		return

	if args.config is None:
		parser.error("config path is required (unless --cache-path is given)")

	if args.refresh:
		global _cache  # noqa: PLW0603
		path: Path = _get_cache_path()
		if path.exists():
			path.unlink()
			print(f"Deleted {path}")
		_cache = None

	with args.config.open("rb") as f:
		data: dict = tomllib.load(f)

	models: list[str] = data["models"]
	print(f"Warming cache for {len(models)} model(s)...")
	for model_name in models:
		resolved: str = cached_resolve_model_name(model_name)
		sanitized: str = cached_sanitize_model_name(model_name)
		hf_repo_id: str = cached_get_hf_repo_id(model_name)
		print(
			f"  {model_name} -> resolved={resolved!r}, "
			f"sanitized={sanitized!r}, hf_repo_id={hf_repo_id!r}"
		)
	print(f"Cache written to {_get_cache_path()}")


if __name__ == "__main__":
	main()
