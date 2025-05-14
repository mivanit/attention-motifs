import warnings


def filter_umap_warns():
	warnings.filterwarnings(
		action="ignore",
		category=FutureWarning,
		message=r"'force_all_finite' was renamed to 'ensure_all_finite'",
		module=r"^sklearn\.utils\.deprecation$",
	)
	warnings.filterwarnings(
		"ignore",
		category=UserWarning,
		message=r"using precomputed metric; inverse_transform will be unavailable",
		module=r"^umap\.umap_$",
	)
	warnings.filterwarnings(
		"ignore",
		category=UserWarning,
		message=r"n_jobs value .* overridden .* random_state",
		module=r"^umap\.umap_$",
	)
