"""Scan patterns directories for corrupt activations.npz files.

npz files can be corrupted when one of the activation-writing processes from s1 crashes or is killed while in the middle of a write. this utility lets us scan for and delete those files, letting us re-run s1 which (when `force_overwrite=False`) will skip already-complete prompts and only re-run the ones with missing npz files.

Usage:
    python -m attention_motifs.util.find_corrupt_npz
    python -m attention_motifs.util.find_corrupt_npz --model gpt2-small
    python -m attention_motifs.util.find_corrupt_npz --verbose-files
    python -m attention_motifs.util.find_corrupt_npz --verbose-errors
    python -m attention_motifs.util.find_corrupt_npz --delete
    python -m attention_motifs.util.find_corrupt_npz --delete-cached
    python -m attention_motifs.util.find_corrupt_npz --cache-info
    python -m attention_motifs.util.find_corrupt_npz --workers 16 --chunksize 128
"""

import argparse
import multiprocessing
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

CACHE_FILENAME: str = ".corrupt_npz_cache.txt"


def find_npz(patterns_dir: Path, models: list[str]) -> list[Path]:
	"""Collect all activations.npz paths for the given models."""
	all_npz: list[Path] = []
	model: str
	for model in models:
		model_dir: Path = patterns_dir / model / "prompts"
		if not model_dir.is_dir():
			print(f"skipping {model}: {model_dir} not found", file=sys.stderr)
			continue
		all_npz.extend(sorted(model_dir.rglob("activations.npz")))
	return all_npz


def _check_one(npz: Path) -> tuple[Path, str] | None:
	"""Check a single npz file for corruption. Returns (path, error) or None."""
	try:
		with zipfile.ZipFile(npz) as z:
			bad: str | None = z.testzip()
			if bad is not None:
				return (npz, f"bad entry: {bad}")
	except Exception as e:
		return (npz, str(e))
	return None


def check_npz_corrupt(
	files: list[Path],
	workers: int | None = None,
	chunksize: int = 64,
) -> list[tuple[Path, str]]:
	"""Check files for corruption in parallel, return list of (path, error)."""
	corrupt: list[tuple[Path, str]] = []
	pbar: tqdm = tqdm(total=len(files), desc="checking npz files", unit="file")
	with multiprocessing.Pool(processes=workers) as pool:
		result: tuple[Path, str] | None
		for result in pool.imap_unordered(_check_one, files, chunksize=chunksize):
			if result is not None:
				corrupt.append(result)
				pbar.set_postfix(corrupt=len(corrupt))
			pbar.update()
	pbar.close()
	return corrupt


def _cache_path(patterns_dir: Path) -> Path:
	"""Return the cache file path for a given patterns directory."""
	return patterns_dir / CACHE_FILENAME


def save_cache(patterns_dir: Path, corrupt: list[tuple[Path, str]]) -> None:
	"""Save corrupt file paths to a newline-separated cache file."""
	cache: Path = _cache_path(patterns_dir)
	lines: str = "\n".join(str(p) for p, _ in corrupt)
	cache.write_text((lines + "\n") if lines else "")


def load_cache(patterns_dir: Path) -> list[Path]:
	"""Load corrupt file paths from cache. Raises FileNotFoundError if missing."""
	cache: Path = _cache_path(patterns_dir)
	text: str = cache.read_text().strip()
	if not text:
		return []
	return [Path(line) for line in text.splitlines()]


def _model_from_path(path: Path) -> str:
	"""Extract model name from .../patterns/{model}/prompts/{hash}/activations.npz."""
	return path.parent.parent.parent.name


def _cache_info(patterns_dir: Path, workers: int | None, chunksize: int) -> None:
	"""Re-check cached corrupt files and print TOML-style error details."""
	try:
		cached_paths: list[Path] = load_cache(patterns_dir)
	except FileNotFoundError:
		print(f"no cache found at {_cache_path(patterns_dir)}", file=sys.stderr)
		sys.exit(2)

	if not cached_paths:
		print("cache is empty")
		sys.exit(0)

	# re-check only the cached files
	corrupt: list[tuple[Path, str]] = check_npz_corrupt(
		cached_paths, workers=workers, chunksize=chunksize
	)

	# files that were cached but no longer exist
	missing: list[Path] = [p for p in cached_paths if not p.exists()]

	for path, error in corrupt:
		print(f"\"{path}\" = '''\n{error}'''")

	for path in missing:
		print(f"\"{path}\" = 'missing'")

	n_still_corrupt: int = len(corrupt)
	n_missing: int = len(missing)
	n_fixed: int = len(cached_paths) - n_still_corrupt - n_missing
	print(
		f"\n{len(cached_paths)} cached, {n_still_corrupt} still corrupt, {n_missing} missing, {n_fixed} fixed"
	)
	sys.exit(1 if n_still_corrupt else 0)


def _delete_cached(patterns_dir: Path) -> None:
	"""Interactive deletion of previously-cached corrupt files."""
	try:
		cached_paths: list[Path] = load_cache(patterns_dir)
	except FileNotFoundError:
		print(f"no cache found at {_cache_path(patterns_dir)}", file=sys.stderr)
		sys.exit(2)

	if not cached_paths:
		print("cache is empty, nothing to delete")
		sys.exit(0)

	model_counts: Counter[str] = Counter(_model_from_path(p) for p in cached_paths)

	print(f"{len(cached_paths)} corrupt file(s) cached ({_cache_path(patterns_dir)}):")
	model: str
	count: int
	for model, count in sorted(model_counts.items()):
		print(f"  {model}: {count}")

	answer: str = input("\ndelete these files? [y/N] ").strip().lower()
	if answer != "y":
		print("aborted")
		sys.exit(0)

	deleted: int = 0
	path: Path
	for path in cached_paths:
		if path.exists():
			path.unlink()
			deleted += 1

	print(f"deleted {deleted} file(s)")
	_cache_path(patterns_dir).unlink(missing_ok=True)
	sys.exit(0)


def main(
	patterns_dir: Path = Path("data/patterns"),
	model: str | None = None,
	verbose_files: bool = False,
	verbose_errors: bool = False,
	delete: bool = False,
	delete_cached: bool = False,
	cache_info: bool = False,
	workers: int | None = None,
	chunksize: int = 64,
) -> None:
	"""Find and optionally delete corrupt activations.npz files."""
	if not patterns_dir.is_dir():
		print(f"error: {patterns_dir} is not a directory", file=sys.stderr)
		sys.exit(2)

	if cache_info:
		_cache_info(patterns_dir, workers=workers, chunksize=chunksize)
		return

	if delete_cached:
		_delete_cached(patterns_dir)
		return

	if model is not None:
		models: list[str] = [model]
	else:
		models = sorted(d.name for d in patterns_dir.iterdir() if d.is_dir())

	files: list[Path] = find_npz(patterns_dir, models)
	corrupt: list[tuple[Path, str]] = check_npz_corrupt(
		files, workers=workers, chunksize=chunksize
	)

	# always save cache for later --delete-cached
	save_cache(patterns_dir, corrupt)

	if verbose_files:
		for path, _error in corrupt:
			print(path, file=sys.stderr)

	if verbose_errors:
		for path, error in corrupt:
			print(f"\"{path}\" = '''\n{error}'''", file=sys.stderr)

	if delete:
		for path, _error in corrupt:
			path.unlink()
		if corrupt:
			print(f"deleted {len(corrupt)} corrupt file(s)")

	n_corrupt: int = len(corrupt)
	if n_corrupt:
		print(f"{n_corrupt} corrupt file(s) (cached to {_cache_path(patterns_dir)})")
		sys.exit(1)
	else:
		print("all files OK")
		sys.exit(0)


def cli() -> None:
	"""Parse CLI arguments and call main."""
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Find corrupt activations.npz files under a patterns directory.",
	)
	parser.add_argument(
		"--patterns-dir",
		type=Path,
		default=Path("data/patterns"),
		help="Root patterns directory (default: data/patterns)",
	)
	parser.add_argument(
		"--model",
		type=str,
		default=None,
		help="Scan a single model (default: all models)",
	)
	parser.add_argument(
		"--verbose-files",
		action="store_true",
		help="Print corrupt file paths to stderr, one per line",
	)
	parser.add_argument(
		"--verbose-errors",
		action="store_true",
		help="Print corrupt files as TOML key=value pairs with multiline literal strings",
	)
	parser.add_argument(
		"--delete",
		action="store_true",
		help="Delete corrupt files after finding them",
	)
	parser.add_argument(
		"--delete-cached",
		action="store_true",
		help="Delete corrupt files from a previous scan (interactive)",
	)
	parser.add_argument(
		"--cache-info",
		action="store_true",
		help="Re-check cached corrupt files and print TOML-style error details",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=None,
		help=f"Number of parallel worker processes (default: os.cpu_count()={os.cpu_count()})",
	)
	parser.add_argument(
		"--chunksize",
		type=int,
		default=64,
		help="Files per IPC chunk sent to each worker (default: 64)",
	)

	args: argparse.Namespace = parser.parse_args()
	main(
		patterns_dir=args.patterns_dir,
		model=args.model,
		verbose_files=args.verbose_files,
		verbose_errors=args.verbose_errors,
		delete=args.delete,
		delete_cached=args.delete_cached,
		cache_info=args.cache_info,
		workers=args.workers,
		chunksize=args.chunksize,
	)


if __name__ == "__main__":
	cli()
