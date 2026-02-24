"""Shared cache-path resolution for attention-motifs.

Uses ``.meta/local/`` when running from a repo clone (detected by the
presence of a ``.meta/`` directory next to the package root), otherwise
falls back to ``~/.cache/attention_motifs/``.
"""

import importlib.resources
from pathlib import Path


def resolve_cache_path(filename: str) -> Path:
	"""Return the cache path for *filename*, preferring repo-local storage.

	Parameters
	----------
	filename:
		Basename of the cache file (e.g. ``"model_table.csv"``).
	"""
	import attention_motifs  # noqa: PLC0415

	pkg_root: Path = Path(str(importlib.resources.files(attention_motifs)))
	repo_root: Path = pkg_root.parent
	if (repo_root / ".meta").is_dir():
		return repo_root / ".meta" / "local" / filename
	return Path.home() / ".cache" / "attention_motifs" / filename
