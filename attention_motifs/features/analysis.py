import polars as pl

# plotting

# scipy

# muutils
from muutils.dbg import dbg
from muutils.tensor_info import array_summary


def null_stats(df: pl.DataFrame) -> pl.DataFrame:
	"""Returns a dataframe with the count of missing values (NaN, null, etc.) for each column.

	# Parameters:
	 - `df : pl.DataFrame`
	   Input dataframe.
	
	# Returns:
	 - `pl.DataFrame`
	   DataFrame with columns `feature`, `missing_count`, and `missing_frac`.
	"""
	nan_counts: dict[str, int] = {}
	for col in df.columns:
		col_dtype: pl.DataType = df.schema[col]
		# For columns whose dtype string contains "float", check both nulls and NaNs.
		if "float" in str(col_dtype).lower():
			missing_expr: pl.Expr = pl.col(col).is_null() | pl.col(col).is_nan()
		else:
			missing_expr: pl.Expr = pl.col(col).is_null()
		count: int = df.select(missing_expr.sum()).item()
		nan_counts[col] = count
	return pl.DataFrame({
		"feature": list(nan_counts.keys()),
		"missing_count": list(nan_counts.values()),
		"missing_frac": [count / df.shape[0] for count in nan_counts.values()],
	}).sort("missing_count", descending=True)


def filter_data(
	df: pl.DataFrame,
	remove_models: list[str] | None = None,
	missing_threshold: float | int = 10,
	threshold_is_percent: bool = False,
) -> pl.DataFrame:
	"""Remove 'feat.*' columns that have zero variance or too many missing values,
	and remove rows corresponding to specified models.

	Columns are dropped if their variance is 0 or if their missing count (or fraction) exceeds the provided threshold.

	# Parameters:
	 - `df : pl.DataFrame`
	   Input dataframe.
	 - `remove_models : list[str] | None`
	   List of model names to remove rows where `activation.model` matches any of these values.
	 - `missing_threshold : float | int`
	   The threshold for missing values. If `threshold_is_percent` is False, this is interpreted as an absolute count.
	   If `threshold_is_percent` is True, then this is interpreted as a fraction (e.g., 0.1 for 10%).
	 - `threshold_is_percent : bool`
	   Whether to treat the `missing_threshold` as a fraction (default is False).

	# Returns:
	 - `pl.DataFrame`
	   Dataframe with filtered rows and columns.
	"""
	df_models_dropped: pl.DataFrame
	if remove_models is not None:
		print(f"Removing {len(remove_models)} models: {remove_models}")
		df_models_dropped = df.filter(~pl.col("activation.model").is_in(set(remove_models)))
		print(f"Removed {len(df) - len(df_models_dropped)}/{len(df)} rows")
	else:
		print("Not removing any models")
		df_models_dropped = df

	# Use null_stats to get missing counts and fractions for all columns.
	stats: pl.DataFrame = null_stats(df_models_dropped)
	# Build a dictionary mapping column name to a tuple (missing_count, missing_frac)
	stats_dict: dict[str, tuple[int, float]] = {
		row["feature"]: (row["missing_count"], row["missing_frac"]) for row in stats.to_dicts()
	}

	# Determine which feature columns to remove.
	feat_cols: list[str] = [col for col in df_models_dropped.columns if col.startswith("feat.")]
	remove_cols: list[str] = []

	for col in feat_cols:
		# Calculate variance for the column.
		var: float = df_models_dropped.select(pl.col(col).var()).item()
		# Retrieve missing statistics for the column (defaulting to 0 if not found).
		missing_count, missing_frac = stats_dict.get(col, (0, 0.0))
		# Check if missing values exceed the configured threshold.
		remove_due_to_missing: bool = (missing_frac > missing_threshold) if threshold_is_percent else (missing_count > missing_threshold)
		if var == 0 or remove_due_to_missing:
			remove_cols.append(col)

	print(f"Removing {len(remove_cols)}/{len(feat_cols)} feat.* columns due to zero variance or excessive missing values")
	for col in remove_cols:
		# Assuming array_summary is defined elsewhere.
		print(f"{col:<60} {array_summary(df_models_dropped[col].to_numpy())}")

	df_filtered: pl.DataFrame = df_models_dropped.drop(remove_cols)
	return df_filtered


def normalize_data(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
	"""Normalize the data in the DataFrame by subtracting the mean and dividing by the standard deviation."""
	return df.with_columns(
		[
			((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
			for col in feature_cols
		]
	)
