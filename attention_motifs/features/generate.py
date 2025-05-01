from pathlib import Path

from jaxtyping import Float
import numpy as np
import polars as pl
from sklearn.decomposition import PCA

# muutils
import muutils.tensor_info
from muutils.dbg import dbg_tensor

# attention-motifs
from attention_motifs.features.analysis import null_stats, filter_data, normalize_data
from attention_motifs.features.plotting import (
	plot_embedding,
	apply_pca,
)
from attention_motifs.bins import Bins
from attention_motifs.features.features import scalar_feature_table, compute_scalar_features
from attention_motifs.features.hist_beta_fit import hist_beta_fit
from attention_motifs.util import prefix_dict
from attention_motifs.features.transition_tensor import tt_features
from attention_motifs.features.vec_features import vec_features
from attention_motifs.math.cos_sim import cosine_similarity_matrix
from attention_motifs.math.math import skew_lt


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
	models: list[str]|None = args.models.split(",") if args.models else None

	scalar_feature_table(
		features_func=compute_scalar_features,
		act_path=Path(args.act_path),
		models=models,
		out_path=Path(args.out_path),
		processes=args.processes,
	)
	