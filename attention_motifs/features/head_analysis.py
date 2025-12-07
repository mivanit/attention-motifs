import warnings
from pathlib import Path
from typing import Any, Literal, Sequence
import itertools

from jaxtyping import Float
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors
from tqdm import tqdm

# scipy
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import sklearn.base
import umap

# attention-motifs
from attention_motifs.features.analysis import parse_cls, DistanceTensorResult
from attention_motifs.attnpedia.attnpedia import AttentionPedia


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


EmbeddingMethod = Literal["isomap", "umap", "tsne", "pca"]


def create_head_embedding_df(
	head_dists: DistanceTensorResult,
	attnpedia: AttentionPedia | None = None,
	embedding_method: EmbeddingMethod = "isomap",
	n_components: int = 3,
	n_neighbors: int = 15,
	reducer_kwargs: dict[str, Any] | None = None,
	match_model: str | None = None,
	save_path: Path | None = None,
) -> pl.DataFrame:
	"""
	Create a polars DataFrame from head distance data, with embeddings and type information.

	# Parameters:
	 - `head_dists : DistanceTensorResult`
		The head distances object containing cls_values and distances
	 - `attnpedia : Optional[AttentionPedia]`
		AttentionPedia object for head type lookups, if None a new one will be created
	 - `embedding_method : Literal["isomap", "umap", "tsne", "pca"]`
		Method to use for dimensionality reduction
	 - `n_components : int`
		Number of dimensions for the embedding (default: 3)
	 - `n_neighbors : int`
		Number of neighbors to consider for manifold methods (default: 15)
	 - `random_state : int`
		Random seed for reproducibility (default: 42)

	# Returns:
	 - `pl.DataFrame`
		A dataframe with one row per head, containing:
		- cls string
		- model, layer, head extracted from cls
		- embedding dimensions (embed.0, embed.1, etc.)
		- head types ("type.primary", "type.group", "type.all")
	"""
	# kwargs
	reducer_kwargs_: dict[str, Any] = reducer_kwargs or {}

	# Create AttentionPedia if not provided
	if attnpedia is None:
		attnpedia = AttentionPedia()

	# match cls if provided
	if match_model is not None:
		# Filter cls values based on match_cls
		head_dists = head_dists.match_cls(match_model)

	# Get distance matrix
	distance_matrix: Float[np.ndarray, "n_heads n_heads"] = head_dists.mean_dists

	# Create embedding based on specified method
	reducer: sklearn.base.BaseEstimator
	match embedding_method:
		case "isomap":
			reducer = Isomap(
				n_neighbors=n_neighbors,
				n_components=n_components,
				metric="precomputed",
				**reducer_kwargs_,
			)
		case "umap":
			filter_umap_warns()
			reducer = umap.UMAP(
				n_neighbors=n_neighbors,
				n_components=n_components,
				metric="precomputed",
				**reducer_kwargs_,
			)
		case "tsne":
			# t-SNE with precomputed distances needs 'random' init, not 'pca'
			# Add this to reducer_kwargs_ if not already specified
			if "init" not in reducer_kwargs_:
				reducer_kwargs_["init"] = "random"

			reducer = TSNE(
				n_components=n_components,
				metric="precomputed",
				**reducer_kwargs_,
			)
		case "pca":
			# For PCA, we need to convert distances to similarity
			# This is a simple approach - might need more sophisticated methods
			similarity_matrix = 1 / (0.001 + distance_matrix)
			# normalize so all values are between 0 and 1
			similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (
				np.max(similarity_matrix) - np.min(similarity_matrix)
			)
			# set diagonal to 1 (self is most similar to self)
			np.fill_diagonal(similarity_matrix, 1.0)
			# Create PCA reducer
			reducer = PCA(
				n_components=n_components,
				**reducer_kwargs_,
			)
		case _:
			raise ValueError(f"Unknown embedding method: {embedding_method}")

	# Perform embedding
	if embedding_method == "pca":
		embedding: np.ndarray = reducer.fit_transform(similarity_matrix)
	else:
		embedding: np.ndarray = reducer.fit_transform(distance_matrix)

	# Parse cls values
	parsed_cls: list[tuple[str, int, int]] = [
		parse_cls(cls_) for cls_ in head_dists.cls_values
	]
	models: list[str] = [model for model, _, _ in parsed_cls]
	layers: list[int] = [layer for _, layer, _ in parsed_cls]
	heads: list[int] = [head for _, _, head in parsed_cls]

	# Get head types
	head_to_type: dict[str, str] = attnpedia.head_to_type()
	head_to_types: dict[str, list[str]] = attnpedia.head_to_types()

	# Get head type groups
	head_type_groups: dict[str, str] = attnpedia.head_type_groups()

	type_primary: list[str] = [
		head_to_type.get(cls, "unknown") for cls in head_dists.cls_values
	]
	type_all: list[str] = [
		", ".join(head_to_types.get(cls, ["unknown"])) for cls in head_dists.cls_values
	]

	# Add the type_group column using the mapping from the JSON file
	type_groups: list[str] = [
		head_type_groups.get(ptype, "unknown") for ptype in type_primary
	]

	# Create basic dataframe
	df: pl.DataFrame = pl.DataFrame(
		{
			"cls": head_dists.cls_values,
			"model": models,
			"layer": layers,
			"head": heads,
			"type.primary": type_primary,
			"type.group": type_groups,
			"type.all": type_all,
		}
	)

	# Add embedding columns
	for i in range(n_components):
		df = df.with_columns(pl.Series(f"embed.{i}", embedding[:, i]))

	# HACK: add metadata
	df._embed_meta = dict(
		embedding_method=embedding_method,
		n_components=n_components,
		n_neighbors=n_neighbors,
		reducer_kwargs=reducer_kwargs,
		match_model=match_model,
	)

	if save_path:
		match_model_str: str = match_model or "ALL"
		fname: str = f"m_{match_model_str}-{embedding_method}-d{n_components}-nb{n_neighbors}.jsonl"
		df.write_ndjson(save_path / fname)

	return df


def create_embedding_df_multi(
	head_dists: DistanceTensorResult,
	attnpedia: AttentionPedia | None = None,
	embedding_methods: Sequence[EmbeddingMethod] = ("isomap", "umap", "tsne", "pca"),
	n_components_list: Sequence[int] = (3,),
	n_neighbors_list: Sequence[int] = (15,),
	match_model: str | None = None,
	save_path: Path | None = None,
) -> pl.DataFrame:
	"""Create a unified polars DataFrame with multiple embedding methods and parameters.

	Creates individual embeddings and combines them into a single DataFrame.

	# Parameters:
	 - `head_dists : DistanceTensorResult`
	    The head distances object containing cls_values and distances
	 - `attnpedia : Optional[AttentionPedia]`
	    AttentionPedia object for head type lookups, if None a new one will be created
	 - `embedding_methods : Sequence[Literal["isomap", "umap", "tsne", "pca"]]`
	    List of methods to use for dimensionality reduction
	 - `n_components_list : Sequence[int]`
	    List of dimensions for the embeddings
	 - `n_neighbors_list : Sequence[int]`
	    List of neighbors values to try for manifold methods
	 - `match_model : str | None`
	    Model name pattern to filter cls values
	 - `save_path : Path | None`
	    Path to save the resulting DataFrame

	# Returns:
	 - `pl.DataFrame`
	    A unified dataframe with all embeddings combined
	"""
	# Create a list to store individual DataFrames
	dfs: list[pl.DataFrame] = []
	match_model_str: str = match_model or "ALL"

	# Create embeddings for all combinations
	for method, n_components, n_neighbors in tqdm(
		list(itertools.product(embedding_methods, n_components_list, n_neighbors_list))
	):
		# Create embedding for this configuration
		df: pl.DataFrame = create_head_embedding_df(
			head_dists=head_dists,
			attnpedia=attnpedia,
			embedding_method=method,
			n_components=n_components,
			n_neighbors=n_neighbors,
			match_model=match_model,
			save_path=None,  # Don't save individual embeddings
		)

		# Rename embedding columns to the desired format
		renamed_df: pl.DataFrame = df.clone()
		for i in range(n_components):
			old_col: str = f"embed.{i}"
			new_col: str = f"embed.{method}.d{n_components}.b{n_neighbors}.dim.{i}"
			renamed_df = renamed_df.rename({old_col: new_col})

		# Store the DataFrame
		dfs.append(renamed_df)

	# Extract base columns (non-embedding columns) from the first DataFrame
	base_cols: list[str] = [
		col for col in dfs[0].columns if not col.startswith("embed.")
	]

	# Start with the first DataFrame
	result_df: pl.DataFrame = dfs[0]

	# Add embedding columns from all other DataFrames
	for df in dfs[1:]:
		embed_cols: list[str] = [col for col in df.columns if col.startswith("embed.")]
		result_df = result_df.with_columns(df.select(embed_cols))

	# Save if a path is provided
	if save_path:
		components_str: str = "-".join(map(str, n_components_list))
		neighbors_str: str = "-".join(map(str, n_neighbors_list))
		fname: str = f"embed.mdl.{match_model_str}.ndim.{components_str}.nb.{neighbors_str}.jsonl"
		result_df.write_ndjson(save_path / fname)

	return result_df


EMBED_CMAP: str = "gist_ncar"


def plot_head_embeddings(
	df: pl.DataFrame,
	prefix: str,
	attnpedia: AttentionPedia | None = None,
	color_by: str = "type.primary",
	dims: tuple[int, int] = (0, 1),
	alphas: tuple[float, float] = (0.7, 0.3),
	sizes: tuple[int, int] = (60, 20),
	ax: plt.Axes | None = None,
	figsize: tuple[int, int] = (12, 10),
	title: str | None = None,
	xlim: tuple[float, float] | None = None,
	ylim: tuple[float, float] | None = None,
) -> tuple[plt.Figure | None, plt.Axes]:
	"""Plot head embeddings colored by the specified column"""

	# Create AttentionPedia if not provided
	if attnpedia is None:
		attnpedia = AttentionPedia()

	# Get unique values for color assignment
	categories: list = df[color_by].unique().to_list()

	# Get colors based on the color_by column
	cmap: matplotlib.colors.Colormap
	color_map: dict
	unknown_color: str
	if color_by == "type.primary":
		# Use the colors defined in the JSON file
		color_map = attnpedia.head_type_colors()
		unknown_color = attnpedia._unknown_color
	elif color_by == "type.group":
		# For type_group, get colors from the JSON file's group definitions
		color_map = {
			group_name: group_info.get("color", attnpedia._unknown_color)
			for group_name, group_info in attnpedia.groups_data.get(
				"groups", {}
			).items()
		}
		color_map["unknown"] = attnpedia._unknown_color
		unknown_color = attnpedia._unknown_color
	else:
		# For other columns, use a default colormap
		cmap = plt.cm.get_cmap("tab10", len(categories))
		color_map = {
			cat: matplotlib.colors.rgb2hex(cmap(i)) for i, cat in enumerate(categories)
		}
		unknown_color = attnpedia._unknown_color

	# Extract embedding dimensions
	x_col: str = f"{prefix}.dim.{dims[0]}"
	y_col: str = f"{prefix}.dim.{dims[1]}"

	# Check if columns exist
	if x_col not in df.columns or y_col not in df.columns:
		raise ValueError(
			f"Embedding columns '{x_col}' or '{y_col}' not found in DataFrame"
		)

	# Create figure and axes if not provided
	fig: plt.Figure | None = None
	ax_provided: bool = ax is not None
	if not ax_provided:
		fig, ax = plt.subplots(figsize=figsize)

	# Create scatter plot
	for cat in categories:
		cat_df: pl.DataFrame = df.filter(pl.col(color_by) == cat)
		scatter_kwargs: dict = dict(
			alpha=alphas[0] if cat != "unknown" else alphas[1],
			markersize=sizes[0]
			if cat != "unknown"
			else sizes[1],  # Use markersize instead of size
			markeredgecolor="none",  # Remove marker edges
			marker="o",
			linestyle="",
		)
		if cat == "unknown":
			scatter_kwargs["color"] = unknown_color
		else:
			scatter_kwargs["color"] = color_map[cat]
		ax.plot(
			cat_df[x_col],
			cat_df[y_col],
			label=cat,
			**scatter_kwargs,
		)

	# Extract embedding metadata from the prefix
	# prefix: str = f"embed.{method}.d2.b{n_neighbors}"
	parts: list[str] = prefix.split(".")
	method: str = parts[1]
	n_neighbors: int = int(parts[3][1:])

	ax.set_xlabel(f"Dimension {dims[0]}")
	ax.set_ylabel(f"Dimension {dims[1]}")

	# Simpler title if inside a grid
	if title is None:
		if ax_provided:
			ax.set_title(f"{method}, n_neighbors={n_neighbors}")
		else:
			title = (
				f"Head Embeddings via {method}\n"
				f"colored by '{color_by}' ({len(categories)} categories), n_neighbors={n_neighbors}"
			)
			ax.set_title(title)
	else:
		ax.set_title(title)

	# Add legend (potentially outside plot for many categories)
	if not ax_provided:  # Only add legend to individual plots
		if len(categories) > 10:
			ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
		else:
			ax.legend()

	# Set x and y limits if provided
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)

	if fig is not None:
		plt.tight_layout()

	return fig, ax


def plot_head_embeddings_multi(
	df: pl.DataFrame,
	attnpedia: AttentionPedia | None = None,
	color_by: str = "type.primary",
	dims: tuple[int, int] = (0, 1),
	alphas: tuple[float, float] = (0.7, 0.3),
	sizes: tuple[int, int] = (60, 20),
	methods: list[str] | None = None,
	n_neighbors_list: list[int] | None = None,
	figsize: tuple[int, int] = (20, 16),
) -> tuple[plt.Figure, np.ndarray]:
	"""Plot a grid of head embeddings for different methods and parameters

	# Parameters:
	 - `df : pl.DataFrame`
	    DataFrame containing head embeddings
	 - `attnpedia : AttentionPedia | None`
	    AttentionPedia object for color lookups
	 - `color_by : str`
	    Column to use for coloring points (default: "type.primary")
	 - `dims : tuple[int, int]`
	    Dimensions to plot (default: (0, 1))
	 - `alphas : tuple[float, float]`
	    Alpha values for (known, unknown) categories (default: (0.7, 0.3))
	 - `sizes : tuple[int, int]`
	    Point sizes for (known, unknown) categories (default: (60, 20))
	 - `methods : list[str] | None`
	    Methods to include in grid (default: all methods in DataFrame)
	 - `n_components : int | None`
	    Component count to include (default: first found in DataFrame)
	 - `n_neighbors_list : list[int] | None`
	    Neighbor counts to include in grid (default: all in DataFrame)
	 - `figsize : tuple[int, int]`
	    Figure size (default: (20, 16))

	# Returns:
	 - `tuple[plt.Figure, np.ndarray]`
	    Figure and array of Axes objects
	"""
	# Create AttentionPedia if not provided
	if attnpedia is None:
		attnpedia = AttentionPedia()

	# Get unique values for color assignment (for shared legend)
	categories: list = df[color_by].unique().to_list()

	# Get colors based on the color_by column
	color_map: dict
	if color_by == "type.primary":
		# Use the colors defined in the JSON file
		color_map = attnpedia.head_type_colors()
	elif color_by == "type.group":
		# For type_group, get colors from the JSON file's group definitions
		color_map = {
			group_name: group_info.get("color", attnpedia._unknown_color)
			for group_name, group_info in attnpedia.groups_data.get(
				"groups", {}
			).items()
		}
		color_map["unknown"] = attnpedia._unknown_color
	else:
		# For other columns, use a default colormap
		cmap = plt.cm.get_cmap("tab10", len(categories))
		color_map = {
			cat: matplotlib.colors.rgb2hex(cmap(i)) for i, cat in enumerate(categories)
		}

	# Use attnpedia's unknown color
	unknown_color: str = attnpedia._unknown_color

	# Find all available methods and n_neighbors values
	methods_found: set[str] = set()
	n_neighbors_found: set[int] = set()
	match_model_str: str = None

	# Simple pattern matching to find all embedding columns and extract their metadata
	embed_cols: list[str] = [col for col in df.columns if col.startswith("embed.")]

	for col in embed_cols:
		parts: list[str] = col.split(".")

		# Extract information
		# f"embed.{method}.d{n_components_val}.b{n_neighbors}"
		try:
			methods_found.add(parts[1])
			n_neighbors_found.add(int(parts[3][1:]))
		except (ValueError, IndexError):
			continue

	# Use found values or specified values
	if not methods_found:
		raise ValueError("No embedding columns found in DataFrame")

	methods_list: list[str] = sorted(methods_found) if methods is None else methods
	n_neighbors_vals: list[int] = (
		sorted(n_neighbors_found) if n_neighbors_list is None else n_neighbors_list
	)

	if match_model_str is None:
		match_model_str = "ALL"

	# Create the grid of plots
	n_rows: int = len(n_neighbors_vals)
	n_cols: int = len(methods_list)

	# Create figure and grid of axes
	fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

	# Plot each embedding
	for i, n_neighbors in enumerate(n_neighbors_vals):
		for j, method in enumerate(methods_list):
			# Construct prefix for this cell
			prefix: str = f"embed.{method}.d2.b{n_neighbors}"

			# Plot on this axis
			_, _ = plot_head_embeddings(
				df=df,
				prefix=prefix,
				attnpedia=attnpedia,
				color_by=color_by,
				dims=dims,
				alphas=alphas,
				sizes=sizes,
				ax=axes[i, j],
			)

	# Create shared legend
	legend_handles: list = []
	legend_labels: list = []

	for cat in categories:
		if cat == "unknown":
			handle = plt.Line2D(
				[0],
				[0],
				marker="o",
				color="w",
				markerfacecolor=unknown_color,
				markersize=10,
				markeredgewidth=0,
				alpha=alphas[1],
			)
		else:
			handle = plt.Line2D(
				[0],
				[0],
				marker="o",
				color="w",
				markerfacecolor=color_map.get(cat, unknown_color),
				markersize=16,
				markeredgewidth=0,
				alpha=alphas[0],
			)
		legend_handles.append(handle)
		legend_labels.append(cat)

	# Add the shared legend at the bottom
	fig.legend(
		handles=legend_handles,
		labels=legend_labels,
		loc="lower center",
		bbox_to_anchor=(0.5, -0.02),
		ncol=min(7, len(categories)),
	)

	# Add overall title
	fig.suptitle(
		(
			"Head Embeddings Comparison"
			# + (
			# 	"(all models)"
			# 	if match_model_str == "ALL"
			# 	else f"(model: '{match_model_str}'"
			# )
			+ f"\ncolored by '{color_by}' ({len(categories)} categories)"
		),
		fontsize=20,
		y=1,
	)

	plt.tight_layout()
	# Adjust for the shared legend
	# plt.subplots_adjust(bottom=0.1 + 0.02 * min(5, len(categories) // 6))

	return fig, axes
