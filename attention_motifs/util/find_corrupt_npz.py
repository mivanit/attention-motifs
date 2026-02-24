"""Scan patterns directories for corrupt activations.npz files.

Usage:
    python -m attention_motifs.util.find_corrupt_npz
    python -m attention_motifs.util.find_corrupt_npz --model gpt2-small
    python -m attention_motifs.util.find_corrupt_npz --verbose-files
    python -m attention_motifs.util.find_corrupt_npz --verbose-errors
    python -m attention_motifs.util.find_corrupt_npz --delete
    python -m attention_motifs.util.find_corrupt_npz --workers 16 --chunksize 128
"""

import argparse
import multiprocessing
import os
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm


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
	with multiprocessing.Pool(processes=workers) as pool:
		result: tuple[Path, str] | None
		for result in tqdm(
			pool.imap_unordered(_check_one, files, chunksize=chunksize),
			total=len(files),
			desc="checking npz files",
			unit="file",
		):
			if result is not None:
				corrupt.append(result)
	return corrupt


def main() -> None:
	"""CLI entrypoint."""
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
	patterns_dir: Path = args.patterns_dir

	if not patterns_dir.is_dir():
		print(f"error: {patterns_dir} is not a directory", file=sys.stderr)
		sys.exit(2)

	if args.model is not None:
		models: list[str] = [args.model]
	else:
		models = sorted(d.name for d in patterns_dir.iterdir() if d.is_dir())

	files: list[Path] = find_npz(patterns_dir, models)
	corrupt: list[tuple[Path, str]] = check_npz_corrupt(
		files, workers=args.workers, chunksize=args.chunksize
	)

	if args.verbose_files:
		for path, _error in corrupt:
			print(path, file=sys.stderr)

	if args.verbose_errors:
		for path, error in corrupt:
			print(f"\"{path}\" = '''\n{error}'''", file=sys.stderr)

	if args.delete:
		for path, _error in corrupt:
			path.unlink()
		if corrupt:
			print(f"deleted {len(corrupt)} corrupt file(s)")

	n_corrupt: int = len(corrupt)
	if n_corrupt:
		print(f"{n_corrupt} corrupt file(s)")
		sys.exit(1)
	else:
		print("all files OK")
		sys.exit(0)


if __name__ == "__main__":
	main()
