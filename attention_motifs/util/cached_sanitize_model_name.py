from pathlib import Path
import json

_MODEL_NAME_CACHE_PATH: Path = (
	Path.home() / ".cache" / "attention-motifs" / "model_name_map.json"
)
_model_name_cache: dict[str, str] | None = None


def cached_sanitize_model_name(name: str) -> str:
	"""Return the sanitized (filesystem-safe) form of a model name.

	Uses a disk-backed cache so that ``pattern_lens`` is only imported on
	cache misses.
	"""
	global _model_name_cache
	if _model_name_cache is None:
		if _MODEL_NAME_CACHE_PATH.exists():
			_model_name_cache = json.loads(_MODEL_NAME_CACHE_PATH.read_text())
		else:
			_model_name_cache = {}

	assert _model_name_cache is not None
	cache: dict[str, str] = _model_name_cache

	if name in cache:
		return cache[name]

	from pattern_lens.load_model import sanitize_model_name

	sanitized: str = sanitize_model_name(name)
	cache[name] = sanitized
	_MODEL_NAME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
	_MODEL_NAME_CACHE_PATH.write_text(json.dumps(cache, indent=2))
	return sanitized
