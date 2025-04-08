import numpy as np
import polars as pl

# plotting
import matplotlib.pyplot as plt

# scipy
from sklearn.decomposition import PCA

# muutils
from muutils.dbg import dbg_tensor


def plot_correlation_matrix(
	df: pl.DataFrame, feature_cols: list[str], sort_by: str = "sum"
) -> None:
	"""Plot a correlation matrix of features and identify features with NaN correlations.

	# Parameters:
	 - `df : pl.DataFrame`
	    Input dataframe containing features
	 - `feature_cols : list[str]`
	    List of feature column names to include in correlation matrix
	 - `sort_by : str`
	    Sorting method: "sum" (total correlation), "var" (variance),
	    "none" (original order)
	    (defaults to `"sum"`)
	"""
	# Calculate correlation matrix
	corr_df: pl.DataFrame = df[feature_cols].corr()
	corr_mat: np.ndarray = corr_df.to_numpy()
	dbg_tensor(corr_mat)  # Keep the original debugging call

	# Sort features if requested
	if sort_by != "none":
		# Replace NaN with 0 for sorting purposes
		corr_mat_no_nan = np.nan_to_num(corr_mat, nan=0)

		if sort_by == "sum":
			# Sort by absolute sum of correlations (total influence)
			sort_metrics = np.abs(corr_mat_no_nan).sum(axis=1)
		elif sort_by == "var":
			# Sort by variance of correlations
			sort_metrics = np.nanvar(corr_mat, axis=1)
		else:
			raise ValueError(f"Unknown sorting method: {sort_by}")

		# Get sorting indices (descending order)
		sort_idx = np.argsort(-sort_metrics)

		# Reorder matrix and feature names
		corr_mat = corr_mat[sort_idx][:, sort_idx]
		feature_cols = [feature_cols[i] for i in sort_idx]

	# Plot with larger figure size
	fig: plt.Figure = plt.figure(figsize=(25, 20))
	ax: plt.Axes = fig.add_subplot(111)
	cax: plt.AxesImage = ax.imshow(
		corr_mat, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1
	)

	# Add colorbar
	cbar = plt.colorbar(cax)
	cbar.set_label("Correlation")

	# Add feature labels
	ax.set_xticks(np.arange(len(feature_cols)))
	ax.set_yticks(np.arange(len(feature_cols)))
	ax.set_xticklabels(feature_cols, rotation=90)
	ax.set_yticklabels(feature_cols)

	# Adjust tick parameters for better readability
	plt.setp(ax.get_xticklabels(), fontsize=8)
	plt.setp(ax.get_yticklabels(), fontsize=8)

	# Add grid lines
	ax.set_xticks(np.arange(-0.5, len(feature_cols), 1), minor=True)
	ax.set_yticks(np.arange(-0.5, len(feature_cols), 1), minor=True)
	ax.grid(which="minor", color="w", linestyle="-", linewidth=1)

	# Set title
	sort_method = {
		"sum": "Total Correlation",
		"var": "Correlation Variance",
		"none": "Original Order",
	}
	ax.set_title(
		f"Feature Correlation Matrix (Sorted by {sort_method.get(sort_by, sort_by)})"
	)

	plt.tight_layout()
	plt.show()


def apply_pca(
	data: pl.DataFrame,
	n_components: int,
	feature_cols: list[str],
	plot_variance: bool = True,
) -> tuple[np.ndarray, PCA]:
	"""Compute PCA and optionally plot explained variance.

	# Parameters:
	 - `data : pl.DataFrame`
	    Input dataframe
	 - `n_components : int`
	    Number of PCA components to compute
	 - `feature_cols : list[str]`
	    Feature columns to use for PCA
	 - `plot_variance : bool`
	    Whether to plot explained variance (defaults to `True`)

	# Returns:
	 - `tuple[np.ndarray, PCA]`
	    Tuple containing:
	    - Transformed data array
	    - Fitted PCA object
	"""
	# Fit PCA
	pca: PCA = PCA(n_components=n_components, random_state=0)
	reduced: np.ndarray = pca.fit_transform(data[feature_cols].to_numpy())

	# Plot explained variance if requested
	if plot_variance:
		# Calculate explained variance
		explained_variance = pca.explained_variance_ratio_
		cumulative_variance = np.cumsum(explained_variance)

		# Create figure with two subplots
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

		# Plot individual explained variance
		ax1.bar(range(len(explained_variance)), explained_variance)
		ax1.set_xlabel("Principal Component")
		ax1.set_ylabel("Explained Variance Ratio")
		ax1.set_title("Individual Explained Variance")
		ax1.set_xticks(range(min(10, len(explained_variance))))

		# Plot cumulative explained variance
		ax2.plot(range(len(cumulative_variance)), cumulative_variance, "o-")
		ax2.axhline(y=0.8, color="r", linestyle="--", label="80% Threshold")
		ax2.axhline(y=0.9, color="g", linestyle="--", label="90% Threshold")
		ax2.set_xlabel("Number of Components")
		ax2.set_ylabel("Cumulative Explained Variance")
		ax2.set_title("Cumulative Explained Variance")
		ax2.set_xticks(range(0, min(20, len(cumulative_variance)), 2))
		ax2.legend()

		plt.tight_layout()
		plt.show()

		# Print summary
		print(f"Number of components: {n_components}")
		print(f"Total explained variance: {cumulative_variance[-1]:.4f}")

		# Find number of components for 80% and 90% variance
		comp_80 = np.argmax(cumulative_variance >= 0.8) + 1
		comp_90 = np.argmax(cumulative_variance >= 0.9) + 1
		print(f"Components needed for 80% variance: {comp_80}")
		print(f"Components needed for 90% variance: {comp_90}")

	return reduced, pca


def plot_embedding(
	embedding: np.ndarray,
	labels: pl.Series,
	dims: tuple[int, int] = (0, 1),
	title: str = "2D PCA Embedding",
	alpha: float = 0.9,
	marker_size: int = 1,
) -> None:
	"""Scatter plot of 2D embedding with points colored by label.

	# Parameters:
	 - `embedding : np.ndarray`
	    2D embedding array with shape (n_samples, 2)
	 - `labels : pl.Series`
	    Labels for each point
	 - `dims : tuple[int, int]`
	    Dimensions to plot (defaults to (0, 1))
	 - `title : str`
	    Plot title (defaults to "2D PCA Embedding")
	"""
	fig: plt.Figure = plt.figure(figsize=(10, 8))
	ax: plt.Axes = fig.add_subplot(111)

	# Convert labels to numpy array
	label_values = labels.to_numpy()

	# Get unique labels for coloring
	unique_labels = np.unique(label_values)

	# Create colormap with enough colors
	cmap = plt.cm.get_cmap("tab10" if len(unique_labels) <= 20 else "Set3")

	# Store handles for legend
	handles = []

	# Plot each label group with a different color
	for i, label in enumerate(unique_labels):
		mask = label_values == label
		color = cmap(i)

		# Main scatter plot (small points)
		ax.scatter(
			embedding[mask, dims[0]],
			embedding[mask, dims[1]],
			c=[color],
			alpha=alpha,
			s=marker_size,
			edgecolors="none",
		)

		# Create a separate point for the legend (not displayed in the plot)
		handle = plt.Line2D(
			[0],
			[0],
			marker="o",
			color="w",
			markerfacecolor=color,
			markersize=10,  # Big marker size for legend
			label=str(label),
		)
		handles.append(handle)

	ax.set_title(title)
	ax.set_aspect("equal")
	ax.set_xlabel(f"Dim {dims[0]}")
	ax.set_ylabel(f"Dim {dims[1]}")

	# Create legend with large dots
	legend = plt.legend(
		handles=handles,
		loc="upper left",
		bbox_to_anchor=(1, 1),
		title="Labels",
	)

	plt.tight_layout()
	plt.show()
