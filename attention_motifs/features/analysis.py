from functools import cached_property
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence
import math
from collections import defaultdict
from statistics import median
from typing import Self, Any

import numpy as np
import polars as pl
from jaxtyping import Float
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# muutils
from muutils.tensor_info import array_summary
from muutils.json_serialize import json_serialize
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
from zanj import ZANJ

from attention_motifs.attnpedia import parse_cls


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
		missing_expr: pl.Expr
		if "float" in str(col_dtype).lower():
			missing_expr = pl.col(col).is_null() | pl.col(col).is_nan()
		else:
			missing_expr = pl.col(col).is_null()
		count: int = df.select(missing_expr.sum()).item()
		nan_counts[col] = count
	return pl.DataFrame(
		{
			"feature": list(nan_counts.keys()),
			"missing_count": list(nan_counts.values()),
			"missing_frac": [count / df.shape[0] for count in nan_counts.values()],
		}
	).sort("missing_count", descending=True)


def filter_data(
	df: pl.DataFrame,
	remove_models: list[str] | None = None,
	missing_threshold: float | int = 0,
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
		df_models_dropped = df.filter(
			~pl.col("activation.model").is_in(set(remove_models))
		)
		print(f"Removed {len(df) - len(df_models_dropped)}/{len(df)} rows")
	else:
		print("Not removing any models")
		df_models_dropped = df

	# Use null_stats to get missing counts and fractions for all columns.
	stats: pl.DataFrame = null_stats(df_models_dropped)
	# Build a dictionary mapping column name to a tuple (missing_count, missing_frac)
	stats_dict: dict[str, tuple[int, float]] = {
		row["feature"]: (row["missing_count"], row["missing_frac"])
		for row in stats.to_dicts()
	}

	# Determine which feature columns to remove.
	feat_cols: list[str] = [
		col for col in df_models_dropped.columns if col.startswith("feat.")
	]
	remove_cols: list[str] = []

	for col in feat_cols:
		# Calculate variance for the column.
		var: float = df_models_dropped.select(pl.col(col).var()).item()
		# Retrieve missing statistics for the column (defaulting to 0 if not found).
		missing_count, missing_frac = stats_dict.get(col, (0, 0.0))
		# Check if missing values exceed the configured threshold.
		remove_due_to_missing: bool = (
			(missing_frac > missing_threshold)
			if threshold_is_percent
			else (missing_count > missing_threshold)
		)
		if var == 0 or remove_due_to_missing:
			remove_cols.append(col)

	print(
		f"Removing {len(remove_cols)}/{len(feat_cols)} feat.* columns due to zero variance or excessive missing values"
	)
	for col in remove_cols:
		# Assuming array_summary is defined elsewhere.
		print(
			f"{col:<60} {array_summary(df_models_dropped[col].to_numpy(), as_list=False)}"
		)

	df_filtered: pl.DataFrame = df_models_dropped.drop(remove_cols)
	return df_filtered


def normalize_data(
	df: pl.DataFrame,
	feature_cols: Iterable[str] | None = None,
	stats: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
	"""Return `(normalized_df, stats_df)` where every chosen column has zero
	mean and unit variance.

	If `stats` is provided, those numbers are *used* (reproducible
	preprocessing). Otherwise they are *computed* from `df`.

	# Parameters
	 - `df : pl.DataFrame`
	 - `feature_cols : Iterable[str] | None`
	   Columns to normalize (defaults to all that start with `"feat."`).
	 - `stats : pl.DataFrame | None`
	   Optional three-column frame with schema
	   `{"feature": pl.Utf8, "mean": pl.Float64, "std": pl.Float64}`.

	# Returns
	 - `pl.DataFrame` normalized data
	 - `pl.DataFrame` the statistics that were actually used to normalize it
	"""
	if feature_cols is None:
		feature_cols = [c for c in df.columns if c.startswith("feat.")]
	feature_cols = list(feature_cols)
	if not feature_cols:
		raise ValueError("No feature columns supplied / detected.")

	# ------------------------------------------------------------------ stats
	if stats is None:
		stats_df = pl.DataFrame(
			{
				"feature": feature_cols,
				"mean": [df[col].mean() for col in feature_cols],
				"std": [df[col].std() for col in feature_cols],
			}
		)
	else:
		required = {"feature", "mean", "std"}
		if set(stats.columns) != required:
			raise ValueError(
				f"`stats` must have columns {required}, got {stats.columns}"
			)
		missing = set(feature_cols) - set(stats["feature"])
		if missing:
			raise ValueError(f"`stats` lacks entries for {sorted(missing)}")
		stats_df = stats

	mean_map = dict(zip(stats_df["feature"], stats_df["mean"]))
	std_map = dict(zip(stats_df["feature"], stats_df["std"]))

	# ----------------------------------------------------------- normalise df
	norm_df = df.with_columns(
		[
			((pl.col(col) - mean_map[col]) / std_map[col]).alias(col)
			for col in feature_cols
		]
	)

	return norm_df, stats_df


def pca_importance_table(
	pca_obj: PCA,
	feature_names: list[str],
) -> pl.DataFrame:
	"""Return PCA loadings plus simple importance stats.

	# Parameters:
	 - `pca_obj : PCA`
		Fitted PCA.
	 - `feature_names : list[str]`
		Columns used to fit the PCA.

	# Returns:
	 - `pl.DataFrame`
		One row per feature; columns =
		`PC0 … PCk`, `abs_sum`, `abs_mean`, `abs_max`, `abs_var`, `var_weighted`.
	"""
	# raw loadings → (n_features × n_components)
	loadings = pca_obj.components_.T
	comp_cols = [f"PC{i}" for i in range(pca_obj.n_components_)]
	df = (
		pl.DataFrame(loadings, schema=comp_cols)
		.with_columns(pl.Series("feature", feature_names))
		.select(["feature", *comp_cols])
	)

	abs_exprs = [pl.col(c).abs() for c in comp_cols]
	vw_exprs = [
		pl.col(f"PC{i}").abs() * w
		for i, w in enumerate(pca_obj.explained_variance_ratio_)
	]

	return df.with_columns(
		abs_mean=pl.sum_horizontal(abs_exprs) / len(comp_cols),
		abs_max=pl.max_horizontal(abs_exprs),
		abs_var=pl.concat_list(abs_exprs).list.var(ddof=0),  # replacement
		var_weighted=pl.sum_horizontal(vw_exprs),
	).sort("abs_mean", descending=True)


def aggregate_feature_stats(
	df: pl.DataFrame,
	*,
	side: int | set[int] = -1,
	abs_col: str = "abs_mean",
	feature_col: str = "feature",
) -> pl.DataFrame:
	"""Group by the first *or* last token of a dot-separated *feature* column
	and compute summary statistics for `abs_mean`.

	A minimal loop + `defaultdict` keeps things readable; only the final
	conversion to a `polars.DataFrame` uses Polars internals.

	# Parameters
	 - `df : pl.DataFrame`
	   Input data with at least `feature_col` & `abs_col`.
	 - `side : Literal["first", "last"]`
	   Which token to group on. (defaults to `"last"`)
	 - `abs_col : str`
	   Numeric column to aggregate. (defaults to `"abs_mean"`)
	 - `feature_col : str`
	   Dot-delimited feature names. (defaults to `"feature"`)

	# Returns
	 - `pl.DataFrame`
	   One row per token, with the requested statistics.

	# Usage
	```python
	out = aggregate_feature_stats(df, side="first", stats=("mean", "median"))
	```

	# Raises
	 - `ValueError` : invalid `side` or `stats` entry
	"""
	# --- simple loop, no Polars “wizardry” ---------------------------------
	groups: defaultdict[str, list[float]] = defaultdict(list)

	feat_series = df[feature_col]  # pl.Series
	val_series = df[abs_col]  # pl.Series

	for feat, val in zip(feat_series, val_series, strict=False):
		if val is None or (isinstance(val, float) and math.isnan(val)):
			continue  # skip nulls/NaNs
		token: str
		if isinstance(side, int):
			token = feat.split(".")[side]
		else:
			token = ".".join([s for i, s in enumerate(feat.split(".")) if i in side])
		groups[token].append(float(val))

	# --- compute requested statistics --------------------------------------
	rows: list[dict[str, float | str]] = []
	for token, values in groups.items():
		vals_np: np.ndarray = np.asarray(values, dtype=float)
		row: dict[str, float | str] = {"token": token}
		row["mean"] = float(vals_np.mean())
		row["min"] = float(vals_np.min())
		row["max"] = float(vals_np.max())
		row["median"] = float(median(values))  # statistics.median faster than np
		rows.append(row)

	# convert back to Polars for downstream work
	return pl.DataFrame(rows)


# ------------------------------------------------------------------------
MONO_FONT: str = "DejaVu Sans Mono"  # any monospace installed on your system
PAD_CHARS: int = 2  # add this many spaces **beyond** max length
# ------------------------------------------------------------------------


def plot_importance_covariance(
	data: pl.DataFrame,
	importance_df: pl.DataFrame,
	metrics: Sequence[str] = ["abs_sum", "abs_max"],
	descending: bool = True,
	feature_order: list[str] | None = None,  # << NEW
	bins: int | None = None,
	cmap: str = "coolwarm",
	feat_strip_prefix: str = "feat.",
	figsize: tuple[int, int] = (25, 22),
	trim_frac: float = 0.028,
	tick_pad: int = 30,
	importance_threshold: float | None = None,
	fontsize: int = 8,
	imp_legend_align: float = -0.3,
) -> tuple[list[str], np.ndarray]:
	# ---------- pick feature sequence ------------------------------------
	if feature_order is not None:
		# Use the given list exactly as provided
		features = feature_order
		# Grab importance scores in that order (fill with NaNs if missing)
		scores_df = pl.DataFrame({"feature": features}).join(
			importance_df.select("feature", metrics[0]), on="feature", how="left"
		)
	else:
		# Fall back to importance-sorted order
		scores_df = importance_df.sort(metrics[0], descending=descending).select(
			"feature",
			*metrics,
		)
		features = scores_df["feature"].to_list()

	if importance_threshold is not None:
		# Filter features based on the importance threshold
		scores_df = scores_df.filter(pl.col(metrics[0]) > importance_threshold)
		features = scores_df["feature"].to_list()

	labels = [f.removeprefix(feat_strip_prefix) for f in features]

	# ---------- monospace + right-padding for labels ----------------------
	max_len = max(len(lbl) for lbl in labels) + PAD_CHARS
	pad_lbls = [lbl.ljust(max_len) for lbl in labels]

	# ---------- covariance ------------------------------------------------
	cov = np.cov(data[features].to_numpy().T, bias=False)
	vmax = np.nanmax(np.abs(cov)) or 1.0

	# ---------- figure / grid --------------------------------------------
	ncols = 2 if bins else 1
	gs = gridspec.GridSpec(
		2,
		ncols,
		height_ratios=[1, 8],
		width_ratios=[20, 5] if bins else [20],
		hspace=0.04,
		wspace=0.3,
	)
	fig = plt.figure(figsize=figsize)

	# ---------- covariance heat-map --------------------------------------
	ax_cov = fig.add_subplot(gs[1, 0])
	im = ax_cov.imshow(cov, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

	ax_cov.set_xticks(range(len(features)))
	ax_cov.set_yticks(range(len(features)))
	ax_cov.set_xticklabels(
		pad_lbls,
		rotation=90,
		rotation_mode="anchor",
		fontfamily=MONO_FONT,
		fontsize=fontsize,
	)
	ax_cov.set_yticklabels(
		pad_lbls,
		fontfamily=MONO_FONT,
		fontsize=fontsize,
	)
	ax_cov.tick_params(axis="x", pad=tick_pad)  # << shift labels downward

	ax_cov.set_xlabel("Features")
	ax_cov.set_ylabel("Features")

	divider = make_axes_locatable(ax_cov)
	cax = divider.append_axes("right", size="2.5%", pad=0.05)
	fig.colorbar(im, cax=cax).set_label("Covariance")

	# ---------- importance dot-plot (shares x) ---------------------------
	ax_imp = fig.add_subplot(gs[0, 0], sharex=ax_cov)
	ax_imp.set_ylabel("Importance")
	for m in metrics:
		s = scores_df[m].to_numpy()
		ax_imp.plot(np.arange(len(s)), s, "o", markersize=5, label=m)
	ax_imp.legend(
		loc="center left",
		bbox_to_anchor=(imp_legend_align, 0.5),  # x,y in axes fraction units
		borderaxespad=0,
	)
	ax_imp.tick_params(axis="x", labelbottom=False)
	ax_imp.set_yscale("log")
	ax_imp.grid(which="major", axis="y")
	ax_imp.grid(which="minor", axis="y", linestyle="--", linewidth=1, alpha=0.4)
	ax_imp.grid(which="major", axis="x", linestyle="--", linewidth=1, alpha=0.4)

	if 0 < trim_frac < 0.5:
		pos = ax_imp.get_position()
		ax_imp.set_position((pos.x0, pos.y0, pos.width * (1 - trim_frac), pos.height))

	# ---------- optional histogram ---------------------------------------
	# if bins:
	# 	ax_hist = fig.add_subplot(gs[:, 1])
	# 	ax_hist.hist(scores, bins=bins, orientation="horizontal", color="gray")
	# 	ax_hist.set_xlabel("count")
	# 	ax_hist.set_ylabel(metric)
	# 	ax_hist.set_title("Importance distribution")
	# 	ax_hist.invert_yaxis()

	plt.tight_layout()

	return features, cov


@serializable_dataclass(methods_no_override=["serialize", "load"])
class DistanceTensorResult(SerializableDataclass):
	"""Return object for `build_distance_tensor`."""

	cls_values: list[str]
	prompt_values: list[str]
	distances: Float[np.ndarray, "h h p"] | Float[np.ndarray, "h h"]
	is_reduced: bool = serializable_field(default=False)

	def serialize(self) -> dict:
		return dict(
			__muutils_format__="DistanceTensorResult(SerializableDataclass)",
			cls_values=self.cls_values,
			prompt_values=self.prompt_values,
			distances=self.mean_dists,
			is_reduced=True,
		)

	# this violates Liskov but its fine
	@classmethod
	def load(cls, data: dict[str, Any] | Self) -> "DistanceTensorResult":  # ty: ignore[invalid-method-override]
		"""Load a `DistanceTensorResult` from a dictionary."""
		if isinstance(data, cls):
			return data
		assert data["is_reduced"], (
			"data must be reduced when loading -- non-reduced would be huge!"
		)
		return cls(
			cls_values=data["cls_values"],
			prompt_values=data["prompt_values"],
			distances=data["distances"],
			is_reduced=True,
		)

	def save(self, path: Path | str, zanj: ZANJ | None = None) -> None:
		if zanj is None:
			zanj = ZANJ()
		zanj.save(
			self.serialize(),
			path,
		)

	def save_raw(self, path: Path | str) -> None:
		"""Save the raw data to json and npy"""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)

		json_meta: str = json.dumps(
			dict(
				cls_values=self.cls_values,
				prompt_values=self.prompt_values,
			),
		)
		with open(path / "dists_meta.json", "w") as f:
			f.write(json_meta)

		# save the mean distances
		np.save(
			path / "distances.npy",
			self.mean_dists,
		)
		# reduced precision, to be faster
		np.save(
			path / "distances_f32.npy",
			self.mean_dists.astype(np.float32),
		)
		np.save(
			path / "distances_f16.npy",
			self.mean_dists.astype(np.float16),
		)

	@classmethod
	def read(cls, path: Path | str, zanj: ZANJ | None = None) -> "DistanceTensorResult":
		if zanj is None:
			zanj = ZANJ()
		return zanj.read(path)

	@classmethod
	def read_raw(
		cls,
		path: Path | str,
		precision: str = "f64",
	) -> "DistanceTensorResult":
		"""Load from raw json + npy files saved by save_raw.

		Parameters
		----------
		path
			Directory containing dists_meta.json and distances*.npy files.
		precision
			Which precision file to load: "f64" (default), "f32", or "f16".

		Returns
		-------
		DistanceTensorResult
			Loaded result with is_reduced=True.
		"""
		path = Path(path)

		# Load metadata
		with open(path / "dists_meta.json", "r") as f:
			meta = json.load(f)

		# Load distances based on precision
		if precision == "f64":
			distances = np.load(path / "distances.npy")
		elif precision == "f32":
			distances = np.load(path / "distances_f32.npy")
		elif precision == "f16":
			distances = np.load(path / "distances_f16.npy")
		else:
			raise ValueError(
				f"Unknown precision: {precision}. Use 'f64', 'f32', or 'f16'."
			)

		return cls(
			cls_values=meta["cls_values"],
			prompt_values=meta["prompt_values"],
			distances=distances,
			is_reduced=True,
		)

	@property
	def n_heads(self) -> int:
		n: int = len(self.cls_values)
		assert self.distances.shape[0] == n
		assert self.distances.shape[1] == n
		return n

	def match_cls(
		self, cls_filter: str | Callable[[str], bool]
	) -> "DistanceTensorResult":
		"""Return a new DistanceTensorResult with only the matching classes."""
		cls_filter_func: Callable[[str], bool]
		if isinstance(cls_filter, str):
			cls_filter_func = lambda cls: cls.startswith(cls_filter)  # noqa: E731
		else:
			cls_filter_func = cls_filter

		matching_cls_idxs: list[int] = [
			i for i, cls in enumerate(self.cls_values) if cls_filter_func(cls)
		]
		matching_cls_idxs_np = np.array(matching_cls_idxs, dtype=int)
		matching_cls: list[str] = [self.cls_values[i] for i in matching_cls_idxs]
		matching_dists: Float[np.ndarray, "h h *p"]

		if self.is_reduced:
			matching_dists = self.distances[
				np.ix_(matching_cls_idxs_np, matching_cls_idxs_np)
			]
		else:
			p: int = self.distances.shape[2]
			matching_dists = self.distances[
				np.ix_(matching_cls_idxs_np, matching_cls_idxs_np, np.arange(p))
			]

		return DistanceTensorResult(
			cls_values=matching_cls,
			prompt_values=self.prompt_values,
			distances=matching_dists,
			is_reduced=self.is_reduced,
		)

	@cached_property
	def mean_dists(self) -> Float[np.ndarray, "h h"]:
		if self.is_reduced:
			return self.distances
		else:
			return self.distances.mean(axis=-1)

	def get_closest_heads(
		self, head: str, n_closest: int = 5
	) -> list[tuple[str, float]]:
		head_idx: int = self.cls_values.index(head)
		dists: Float[np.ndarray, "h h"] = self.mean_dists[head_idx]

		# sort by distance
		sorted_indices: np.ndarray = np.argsort(dists)
		closest_indices: np.ndarray = sorted_indices[:n_closest]
		closest_dists: Float[np.ndarray, " h"] = dists[closest_indices]

		return [
			(self.cls_values[i], float(d))
			for i, d in zip(closest_indices, closest_dists)
		]

	@classmethod
	def build_distance_tensor(
		cls,
		df: pl.DataFrame,
		*,
		cls_col: str = "activation.cls",
		prompt_col: str = "activation.prompt",
		feature_prefix: str = "feat.",
		order: int = 2,
		include_missing_prompts: bool = False,
	) -> "DistanceTensorResult":
		"""
		Compute a (h, h, p) distance tensor grouped by
		(`activation.cls`, `activation.prompt`).

		Parameters
		----------
		df
			Polars DataFrame containing feature columns and two categorical columns.
		cls_col, prompt_col
			Column names holding the categorical identifiers.
		feature_prefix
			Prefix that marks feature columns.
		order
			Order of the L‑p norm (1 → L₁/Manhattan, 2 → L₂/Euclidean).
		include_missing_prompts
			If ``False`` (default), *drop* any prompt that lacks a row for at
			least one class; the output tensor then contains **no** NaNs.
			If ``True``, keep all prompts and leave distances with missing rows
			as ``NaN``.

		Returns
		-------
		DistanceTensorResult
			* ``cls_values``	(list[str]) – first‑occurrence order of cls values
			* ``prompt_values`` (list[str]) – first‑occurrence order of prompts
			* ``distances``	 (Float[Array, 'h h p']) – distance tensor
			(may include ``NaN`` depending on *include_missing_prompts*).
		"""
		# gather feature columns and unique keys (order‑preserving)
		feat_cols: list[str] = [c for c in df.columns if c.startswith(feature_prefix)]

		cls_values: list[str] = (
			df.select(cls_col).get_column(cls_col).unique(maintain_order=True).to_list()
		)
		prompt_values: list[str] = (
			df.select(prompt_col)
			.get_column(prompt_col)
			.unique(maintain_order=True)
			.to_list()
		)

		# cache vectors keyed by (cls, prompt)
		vectors: dict[tuple[str, str], np.ndarray] = {}
		for row in df.select(feat_cols + [cls_col, prompt_col]).iter_rows(named=True):
			key: tuple[str, str] = (row[cls_col], row[prompt_col])
			vectors[key] = np.array([row[c] for c in feat_cols], dtype=float)

		# optionally drop prompts with missing class rows
		if not include_missing_prompts:
			prompt_values = [
				p
				for p in prompt_values
				if all((cls, p) in vectors for cls in cls_values)
			]

		h: int = len(cls_values)
		p: int = len(prompt_values)
		cls_to_i: dict[str, int] = {c: i for i, c in enumerate(cls_values)}

		# build tensor prompt‑by‑prompt
		D: Float[np.ndarray, "h h p"] = np.full((h, h, p), np.nan, dtype=float)

		for k, prompt in tqdm(enumerate(prompt_values), desc="prompts", total=p):
			existing_cls = [cls for cls in cls_values if (cls, prompt) in vectors]
			idxs = [cls_to_i[cls] for cls in existing_cls]
			if len(idxs) < 2:  # 0 or 1 row → nothing to compare
				continue

			X = np.vstack([vectors[(cls, prompt)] for cls in existing_cls])  # m × d
			diff = X[:, None, :] - X[None, :, :]  # m × m × d
			dist = np.linalg.norm(diff, ord=order, axis=-1)  # m × m

			for a, i in enumerate(idxs):
				D[i, idxs, k] = dist[a]

		return DistanceTensorResult(
			cls_values=cls_values,
			prompt_values=prompt_values,
			distances=D,
		)

	def save_means(self, path: Path) -> None:
		with open(path, "w") as f:
			json.dump(
				json_serialize(
					dict(
						labels=self.cls_values,
						prompts=self.prompt_values,
						mean_dists=self.mean_dists,
					)
				),
				f,
			)

	def plot_hists(
		self,
		bins: int = 50,
		alpha: float = 0.01,
		n_samples: int | None = 128,
	) -> Axes:
		assert not self.is_reduced, "plot_hists() only works if we haven't reduced"
		max_dist: float = np.max(self.distances)
		bin_edges: np.ndarray = np.linspace(0, max_dist, bins)
		n_heads: int = self.n_heads
		print(f"{n_heads=}, {max_dist=}")
		if n_samples is not None:
			n_heads = n_samples

		fig, ax = plt.subplots(figsize=(10, 6))

		for i in range(n_heads):
			for j in range(i + 1, n_heads):
				hist, _ = np.histogram(
					self.distances[i, j],
					bins=bin_edges,
					density=True,
				)
				ax.plot(
					bin_edges[:-1] - self.mean_dists[i, j],
					hist,
					color="black",
					alpha=alpha,
				)

		ax.set_xlabel("Normalized Distances")
		ax.set_ylabel("Density")

		return ax

	def plot_heatmap(
		self,
		*,
		figsize: tuple[int, int] = (9, 9),
		stripe_thickness: float = 0.018,
		top_label_space: float = 0.04,
		left_label_space: float = 0.08,
		model_font: int = 8,  # ← pass size here (default a bit smaller)
		layer_font: int = 7,  # kept but no longer used
		major_grid_colour: str = "red",
		minor_grid_colour: str = "red",
		show: bool = True,
	) -> Figure:
		"""
		Draw the distance matrix with:

		* rows / columns sorted by (model, layer, head);
		* colour stripes encoding model (hue) + depth (lightness);
		* model labels on both axes (no layer numbers);
		* red grid (major at model boundaries).
		"""
		# ---------------- sort -------------------------------------------------
		parsed_entries: list[tuple[str, int, int]] = [
			parse_cls(t) for t in self.cls_values
		]
		sort_indices: list[int] = sorted(
			range(len(self.cls_values)), key=lambda i: parsed_entries[i]
		)
		sorted_entries: list[tuple[str, int, int]] = [
			parsed_entries[i] for i in sort_indices
		]
		distance_matrix: np.ndarray = self.mean_dists[
			np.ix_(sort_indices, sort_indices)
		]
		size: int = len(sorted_entries)

		# ---------------- colours ---------------------------------------------
		models: list[str] = sorted({model for model, _, _ in sorted_entries})
		model_rgb: dict[str, tuple[float, float, float]] = {
			model: plt.get_cmap("tab10")(i)[:3] for i, model in enumerate(models)
		}

		max_layer_for_model: dict[str, int] = {}
		for model, layer, _ in sorted_entries:
			max_layer_for_model[model] = max(max_layer_for_model.get(model, -1), layer)

		def entry_rgba(
			entry: tuple[str, int, int],
		) -> tuple[float, float, float, float]:
			model, layer, _ = entry
			base_rgb = np.asarray(model_rgb[model])
			depth_scale = (
				0.3 + 0.7 * (layer / max_layer_for_model[model])
				if max_layer_for_model[model]
				else 0.5
			)
			blended_rgb = base_rgb * depth_scale + (1.0 - depth_scale)
			return (*blended_rgb, 1.0)

		rgba_entries: list[tuple[float, float, float, float]] = [
			entry_rgba(e) for e in sorted_entries
		]
		stripe_top: np.ndarray = np.array(rgba_entries).reshape(1, -1, 4)
		stripe_left: np.ndarray = np.array(rgba_entries).reshape(
			-1, 1, 4
		)  # ← no reverse

		# ---------------- figure / main heat-map -------------------------------
		figure, axis_main = plt.subplots(figsize=figsize)
		heat_img = axis_main.matshow(distance_matrix)
		axis_main.set_xticks([])
		axis_main.set_yticks([])
		axis_main.set_aspect("equal")

		# ---------------- stripes ------------------------------------------------
		axis_top = axis_main.inset_axes(
			(0, 1.0 + top_label_space * 0.4, 1, stripe_thickness),
			transform=axis_main.transAxes,
			sharex=axis_main,
		)
		axis_top.imshow(stripe_top, aspect="auto", origin="upper")
		axis_top.set_axis_off()

		axis_left = axis_main.inset_axes(
			(-stripe_thickness - left_label_space * 0.4, 0, stripe_thickness, 1),
			transform=axis_main.transAxes,
			sharey=axis_main,
		)
		axis_left.imshow(stripe_left, aspect="auto", origin="upper")
		axis_left.set_axis_off()

		# ---------------- model-label axes --------------------------------------
		axis_top_labels = axis_main.inset_axes(
			(
				0,
				1.0 + stripe_thickness + top_label_space * 0.2,
				1,
				top_label_space * 0.8,
			),
			transform=axis_main.transAxes,
			sharex=axis_main,
		)
		axis_top_labels.set_axis_off()

		axis_left_labels = axis_main.inset_axes(
			(-stripe_thickness - left_label_space, 0, top_label_space * 0.8, 1),
			transform=axis_main.transAxes,
			sharey=axis_main,
		)
		axis_left_labels.set_axis_off()

		# ---------------- spans & gridlines -------------------------------------
		model_spans: dict[str, list[int]] = {}
		for index, (model, layer, _) in enumerate(sorted_entries):
			model_spans.setdefault(model, [index, index])[1] = index  # extend end

		# boundaries
		model_boundaries: list[float] = [
			end + 0.5 for _, (_, end) in model_spans.items() if end + 1 < size
		]

		for boundary in model_boundaries:
			# manual adjustment here, idk why it's needed
			axis_main.axvline(boundary + 2, color=major_grid_colour, lw=0.1, zorder=2)
			axis_main.axhline(boundary, color=major_grid_colour, lw=0.1, zorder=2)

		# ---------------- annotations (model names only) ------------------------
		for model, (start, end) in model_spans.items():
			center: float = (start + end) / 2

			axis_top_labels.text(
				center, 0.5, model, ha="center", va="center", fontsize=model_font
			)
			axis_left_labels.text(
				0.5,
				center,
				model,
				ha="center",
				va="center",
				fontsize=model_font,
				rotation=90,
			)

		# ---------------- colour-bar --------------------------------------------
		colour_bar = figure.colorbar(heat_img, ax=axis_main, fraction=0.045, pad=0.03)
		colour_bar.ax.set_ylabel("Mean distance", rotation=270, labelpad=13)

		if show:
			plt.show()
		return figure


def groups_by_covariance(
	features: list[str],
	cov: Float[np.ndarray, "n n"],
	threshold: float,
	use_abs: bool = True,
) -> list[list[str]]:
	n: int = len(features)
	assert n == cov.shape[0] == cov.shape[1], (
		"Covariance matrix must be square and match the number of features."
	)

	# |cov| if desired, then a boolean matrix marking edges above threshold
	comparison_matrix: Float[np.ndarray, "n n"] = np.abs(cov) if use_abs else cov
	adjacency_matrix: Float[np.ndarray, "n n"] = comparison_matrix > threshold

	visited_indices: set[int] = set()
	groups: list[list[str]] = []

	# Depth-first search over the implicit graph
	for start_idx in range(n):
		if start_idx in visited_indices:
			continue
		stack: list[int] = [start_idx]
		component_indices: list[int] = []

		while stack:
			idx: int = stack.pop()
			if idx in visited_indices:
				continue
			visited_indices.add(idx)
			component_indices.append(idx)

			# All indices j where adjacency_matrix[idx, j] is True
			neighbors: np.ndarray = np.where(adjacency_matrix[idx])[0]
			stack.extend([j for j in neighbors if j not in visited_indices])

		groups.append([features[j] for j in component_indices])

	return groups


def print_covariance_groups(
	cov_feats: list[str],
	cov_mat: Float[np.ndarray, "n n"],
	df_importance: pl.DataFrame,
	threshold: float = 0.95,
) -> None:
	groups: list[list[str]] = groups_by_covariance(
		cov_feats, cov_mat, threshold=threshold
	)
	singletons: list[list[str]] = [g for g in groups if len(g) == 1]
	# print(f"{singletons = }")
	# print singleton features with their importance
	# get the row from df_importance that matches the feature name
	singleton_info = [
		(
			g[0],
			df_importance.filter(pl.col("feature").is_in(g)).to_dicts()[0]["abs_max"],
		)
		for g in singletons
	]
	for feat, imp in sorted(singleton_info, key=lambda x: x[1], reverse=True):
		print(f"{feat:60} {imp:.3f}")

	for g in groups:
		if len(g) == 1:
			continue
		print(g)
