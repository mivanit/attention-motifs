from pathlib import Path
from typing import Optional, Any, Union, Callable

import numpy as np
import polars as pl
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

# scipy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# muutils
from muutils.dbg import dbg, dbg_tensor
import muutils.tensor_info
from muutils.tensor_info import array_summary

def nan_stats(df: pl.DataFrame) -> pl.DataFrame:
	return (
		df.null_count()
		.transpose(include_header=True, header_name="feature", column_names=["count"])
		.sort("count", descending=True)
	)


def filter_data(df: pl.DataFrame, remove_models: list[str]|None = None) -> pl.DataFrame:
	"remove cols 'feat.*' which are all nan or have variance 0, rows with the listed models"
	# remove all rows where "activation.model" is in remove_models
	df_models_dropped: pl.DataFrame
	if remove_models is not None:
		print(f"Removing {len(remove_models)} models")
		print(f"Models: {remove_models}")
		df_models_dropped = df.filter(~pl.col("activation.model").is_in(set(remove_models)))
		print(f"Removed {len(df) - len(df_models_dropped)}/{len(df)} rows")
	else:
		print("Not removing any models")
		df_models_dropped = df


	# remove cols whose names start with "feat." and they have 0 variance
	feat_cols: list[str] = [col for col in df.columns if col.startswith("feat.")]
	remove_cols: list[str] = []

	for col in feat_cols:
		# Calculate variance, handling potential NaN values
		var = df_models_dropped.select(pl.col(col).var()).item()
		# print(f"{col = }, {df[col].is_null().sum() = }")
		if var == 0 or df_models_dropped[col].is_null().sum() > 10:
			remove_cols.append(col)

	print(f"Removing {len(remove_cols)}/{len(feat_cols)} cols")
	dbg(remove_cols)
	for col in remove_cols:
		print(f"{col:<60} {array_summary(df_models_dropped[col].to_numpy())}")

	# remove cols that are all nan
	df_cols_dropped: pl.DataFrame = df_models_dropped.drop(remove_cols)

	return df_cols_dropped


def normalize_data(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
	"""Normalize the data in the DataFrame by subtracting the mean and dividing by the standard deviation."""
	return df.with_columns(
		[
			((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
			for col in feature_cols
		]
	)