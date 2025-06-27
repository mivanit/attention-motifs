from pathlib import Path


# muutils

# attention-motifs
from attn_embed.features.features import (
	scalar_feature_table,
	compute_scalar_features,
)


if __name__ == "__main__":
	import argparse

	arg_parser: argparse.ArgumentParser = argparse.ArgumentParser()
	arg_parser.add_argument(
		"-a",
		"--act-path",
		type=str,
		default="data/activations",
		help="Path to the activations directory, should be generated with `python -m pattern_lens.activations <args>",
	)
	arg_parser.add_argument(
		"-m",
		"--models",
		type=str,
		default=None,
		help="Comma separated list of models to process. If None, all models in the activations directory will be processed",
	)
	arg_parser.add_argument(
		"-o",
		"--out-path",
		type=str,
		default="data/features",
		help="Path to save the output files",
	)
	arg_parser.add_argument(
		"-p",
		"--processes",
		type=int,
		default=None,
		help="number of processes to use for parallel processing. If None, use all available cores",
	)
	args: argparse.Namespace = arg_parser.parse_args()
	models: list[str] | None = args.models.split(",") if args.models else None

	scalar_feature_table(
		features_func=compute_scalar_features,
		act_path=Path(args.act_path),
		models=models,
		out_path=Path(args.out_path),
		processes=args.processes,
	)
