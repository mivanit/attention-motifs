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
	all_match: bool


_cache_path: Path | None = None
_cache: dict[str, _CacheEntry] | None = None


def _get_cache_path() -> Path:
	"""Lazy-init and return the cache file path."""
	global _cache_path  # noqa: PLW0603
	if _cache_path is None:
		_cache_path = resolve_cache_path(_CACHE_FILENAME)
	return _cache_path


_CACHE_ENTRY_KEYS: frozenset[str] = frozenset(_CacheEntry.__required_keys__)


def _get_cache() -> dict[str, _CacheEntry]:
	"""Lazy-load the disk cache into memory.

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
				if not isinstance(entry, dict) or set(entry.keys()) != _CACHE_ENTRY_KEYS:
					msg: str = (
						f"Corrupt model name cache at {path}: "
						f"entry {key!r} has keys {set(entry.keys()) if isinstance(entry, dict) else type(entry).__name__}, "
						f"expected {_CACHE_ENTRY_KEYS}. "
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


def _ensure_entry(name: str) -> _CacheEntry:
	"""Return the cache entry for *name*, computing on miss."""
	cache: dict[str, _CacheEntry] = _get_cache()
	if name in cache:
		return cache[name]

	from pattern_lens.consts import sanitize_name_str  # noqa: PLC0415
	from pattern_lens.load_model import resolve_model_name  # noqa: PLC0415

	resolved: str = resolve_model_name(name)
	sanitized: str = sanitize_name_str(resolved)
	entry: _CacheEntry = _CacheEntry(
		original=name,
		resolved=resolved,
		sanitized=sanitized,
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
		print(f"  {model_name} -> resolved={resolved!r}, sanitized={sanitized!r}")
	print(f"Cache written to {_get_cache_path()}")


if __name__ == "__main__":
	main()
