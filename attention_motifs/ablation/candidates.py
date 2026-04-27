"""Candidate head identification and selection for ablation experiments.

Provides:
- CandidateHeads: unified container for heads to evaluate, with factory
  methods that construct from clustering, pattern types, or pipeline config.
- DistanceCandidates: specialized container for heads identified via
  embedding proximity to known induction heads.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from attention_motifs.attnpedia.attnpedia import AttentionPedia, parse_cls
from attention_motifs.features.analysis import DistanceTensorResult
from attention_motifs.util.model_name import cached_sanitize_model_name


# Known induction head types from AttentionPedia
INDUCTION_TYPES = [
	"IOI:Induction",
	"SAE-survey:Induction",
]


def get_known_induction_heads(
	attnpedia: AttentionPedia | None = None,
	types: list[str] | None = None,
) -> list[str]:
	"""Get all known induction heads from AttentionPedia.

	Parameters
	----------
	attnpedia
	    AttentionPedia instance. If None, creates a new one.
	types
	    List of type prefixes to include (e.g., ["IOI:Induction"]).
	    If None, uses INDUCTION_TYPES.

	Returns
	-------
	list[str]
	    List of head identifiers (e.g., ["gpt2-small:L5:H5", "gpt2-small:L6:H9"])
	"""
	if attnpedia is None:
		attnpedia = AttentionPedia()

	if types is None:
		types = INDUCTION_TYPES

	type_to_heads = attnpedia.type_to_heads(prefixed_type=True, prefixed_head=True)

	# Collect unique heads across all specified types
	heads: set[str] = set()
	for type_name in types:
		if type_name in type_to_heads:
			heads.update(type_to_heads[type_name])

	return sorted(heads)


# ============================================================
# CandidateHeads: unified container for heads to evaluate
# ============================================================


@dataclass
class CandidateHeads:
	"""Heads to evaluate, grouped by model.

	Construct via factory class methods:
	- ``from_clustering``: cut a hierarchical clustering at a given height
	- ``from_pattern_types``: use pre-computed PatternTypes assignments
	- ``from_pipeline_config``: infer clustering path from pipeline config
	- ``from_distance_candidates``: convert from DistanceCandidates
	"""

	heads_by_model: dict[str, list[tuple[int, int]]]

	# --- Properties ---

	@property
	def models(self) -> list[str]:
		"""Sorted list of model names."""
		return sorted(self.heads_by_model.keys())

	@property
	def n_heads(self) -> int:
		"""Total number of heads across all models."""
		return sum(len(heads) for heads in self.heads_by_model.values())

	def summary(self) -> str:
		"""Human-readable summary for CLI output."""
		lines: list[str] = []
		for model_name in sorted(self.heads_by_model):
			heads: list[tuple[int, int]] = self.heads_by_model[model_name]
			head_strs: list[str] = [f"L{layer}:H{head}" for layer, head in heads]
			lines.append(f"  {model_name}: {len(heads)} heads — {', '.join(head_strs)}")
		lines.append(
			f"  Total: {self.n_heads} heads across {len(self.heads_by_model)} models"
		)
		return "\n".join(lines)

	def filter_models(self, models: list[str]) -> CandidateHeads:
		"""Return new CandidateHeads restricted to given models."""
		models_set: set[str] = {cached_sanitize_model_name(m) for m in models}
		filtered: dict[str, list[tuple[int, int]]] = {
			m: h for m, h in self.heads_by_model.items() if m in models_set
		}
		return CandidateHeads(heads_by_model=filtered)

	# --- Factory methods ---

	@classmethod
	def from_clustering(
		cls,
		clustering_path: Path | str,
		cluster_id: int,
		cut_height: float | None = None,
		n_clusters: int | None = None,
	) -> CandidateHeads:
		"""Load clustering, cut at given height/n_clusters, extract heads for cluster_id.

		Parameters
		----------
		clustering_path
		    Path to clustering directory (contains linkage.npy, clustering_meta.json).
		cluster_id
		    Cluster index to extract.
		cut_height
		    Height at which to cut the dendrogram.
		n_clusters
		    Number of clusters (alternative to cut_height).
		"""
		from attention_motifs.features.clustering import HierarchicalClusteringResult

		clustering: HierarchicalClusteringResult = HierarchicalClusteringResult.read(
			clustering_path
		)
		assignments: dict[str, int] = clustering.get_clusters(
			n_clusters=n_clusters,
			cut_height=cut_height,
		)
		return cls._from_assignments(assignments, cluster_id)

	@classmethod
	def from_pattern_types(
		cls,
		pattern_types_path: Path | str,
		cluster_id: int,
	) -> CandidateHeads:
		"""Load PatternTypes JSON, extract heads for cluster_id.

		Parameters
		----------
		pattern_types_path
		    Path to pattern_types.json file.
		cluster_id
		    Cluster index to extract (must match a type ID in the file).
		"""
		from attention_motifs.pattern_types.pattern_types import PatternTypes

		pt: PatternTypes = PatternTypes.read(pattern_types_path)
		head_ids: list[str] = pt.heads_by_type(cluster_id)

		heads_by_model: dict[str, list[tuple[int, int]]] = {}
		for head_id in head_ids:
			model_name: str
			layer: int
			head: int
			model_name, layer, head = parse_cls(head_id)
			if model_name not in heads_by_model:
				heads_by_model[model_name] = []
			heads_by_model[model_name].append((layer, head))

		return cls(heads_by_model=heads_by_model)

	@classmethod
	def from_pipeline_config(
		cls,
		cluster_id: int,
		cut_height: float | None = None,
		n_clusters: int | None = None,
		pipeline_cfg_path: Path | str = "pipeline_cfg.toml",
	) -> CandidateHeads:
		"""Infer clustering path from pipeline config, then construct.

		Loads the pipeline config and derives the clustering path via
		``cfg.data_path("clustering")``, then delegates to ``from_clustering``.

		Parameters
		----------
		cluster_id
		    Cluster index to extract.
		cut_height
		    Height at which to cut the dendrogram.
		n_clusters
		    Number of clusters (alternative to cut_height).
		pipeline_cfg_path
		    Path to pipeline TOML config (default: ``pipeline_cfg.toml``).
		"""
		from attention_motifs.pipeline.cfg import PipelineConfig

		cfg: PipelineConfig = PipelineConfig.read(Path(pipeline_cfg_path))
		clustering_path: Path = cfg.data_path("clustering")
		return cls.from_clustering(
			clustering_path,
			cluster_id=cluster_id,
			cut_height=cut_height,
			n_clusters=n_clusters,
		)

	@classmethod
	def from_distance_candidates(
		cls,
		distance_candidates: DistanceCandidates,
		n_per_model: int = 10,
	) -> CandidateHeads:
		"""Convert DistanceCandidates to CandidateHeads (drops scores).

		Parameters
		----------
		distance_candidates
		    Distance-based candidate identification result.
		n_per_model
		    Number of top candidates per model to include.
		"""
		from attention_motifs.attnpedia import heads_from_strings

		heads_by_model: dict[str, list[tuple[int, int]]] = {}
		for model_name in distance_candidates.candidates_by_model:
			top: list[tuple[str, float]] = distance_candidates.get_top_candidates(
				model_name, n=n_per_model
			)
			if top:
				head_strs: list[str] = [h for h, _ in top]
				heads_by_model[model_name] = heads_from_strings(head_strs)

		return cls(heads_by_model=heads_by_model)

	@classmethod
	def all_from_models(cls, model_names: list[str]) -> CandidateHeads:
		"""Build CandidateHeads with ALL heads for the given models.

		Uses the TransformerLens model table to look up ``n_layers`` and
		``n_heads`` per model, avoiding any dependency on clustering or
		pipeline outputs.

		Parameters
		----------
		model_names
			List of model names (e.g. ``["gpt2-small", "pythia-14m"]``).
		"""
		from attention_motifs.pipeline.model_table import ModelInfo, fetch_model_table

		table: dict[str, ModelInfo] = fetch_model_table()
		heads_by_model: dict[str, list[tuple[int, int]]] = {}
		for model_name in model_names:
			sanitized: str = cached_sanitize_model_name(model_name)
			if sanitized not in table:
				raise KeyError(
					f"Model {model_name!r} (sanitized: {sanitized!r}) not found "
					f"in TransformerLens model table. Cannot determine n_layers/n_heads."
				)
			info: ModelInfo = table[sanitized]
			heads_by_model[sanitized] = [
				(layer, head)
				for layer in range(info.n_layers)
				for head in range(info.n_heads)
			]
		return cls(heads_by_model=heads_by_model)

	@classmethod
	def all_from_cls_values(cls, cls_values: list[str]) -> CandidateHeads:
		"""Build CandidateHeads with ALL heads from a cls_values list.

		Parses each ``"model:Llayer:Hhead"`` string and groups by model.
		Useful for running ablations on every head in the pipeline.

		Parameters
		----------
		cls_values
			List of head IDs (e.g. from ``HierarchicalClusteringResult.cls_values``).
		"""
		heads_by_model: dict[str, list[tuple[int, int]]] = {}
		for head_id in cls_values:
			model_name: str
			layer: int
			head: int
			model_name, layer, head = parse_cls(head_id)
			if model_name not in heads_by_model:
				heads_by_model[model_name] = []
			heads_by_model[model_name].append((layer, head))
		return cls(heads_by_model=heads_by_model)

	# --- Internal helpers ---

	@classmethod
	def _from_assignments(
		cls,
		assignments: dict[str, int],
		cluster_id: int,
	) -> CandidateHeads:
		"""Build from a head_id → cluster_id mapping, filtering to cluster_id."""
		heads_by_model: dict[str, list[tuple[int, int]]] = {}
		for head_id, cid in assignments.items():
			if cid != cluster_id:
				continue
			model_name: str
			layer: int
			head: int
			model_name, layer, head = parse_cls(head_id)
			if model_name not in heads_by_model:
				heads_by_model[model_name] = []
			heads_by_model[model_name].append((layer, head))

		return cls(heads_by_model=heads_by_model)


# ============================================================
# DistanceCandidates: distance-based candidate identification
# ============================================================


@dataclass
class DistanceCandidates:
	"""Candidate induction heads identified via embedding proximity.

	Attributes
	----------
	reference_heads
	    Known induction heads used as reference points.
	candidates_by_model
	    Dict mapping model name to list of (head_id, score) tuples,
	    sorted by score descending (higher = more likely induction head).
	all_neighbors
	    Raw neighbor data: dict mapping reference head to list of neighbors.
	k_neighbors
	    Number of neighbors considered per reference head.
	model_n_layers
	    Dict mapping model name to number of layers in that model.
	"""

	reference_heads: list[str]
	candidates_by_model: dict[str, list[tuple[str, float]]] = field(
		default_factory=dict
	)
	all_neighbors: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
	k_neighbors: int = 10
	model_n_layers: dict[str, int] = field(default_factory=dict)

	def get_top_candidates(
		self,
		model: str,
		n: int = 10,
	) -> list[tuple[str, float]]:
		"""Get top N candidate heads for a specific model.

		Parameters
		----------
		model
		    Model name (e.g., "pythia-1b").
		n
		    Number of top candidates to return.

		Returns
		-------
		list[tuple[str, float]]
		    List of (head_id, score) tuples.
		"""
		if model not in self.candidates_by_model:
			return []
		return self.candidates_by_model[model][:n]

	def get_all_candidates(self, min_score: float = 0.0) -> list[tuple[str, float]]:
		"""Get all candidates across all models with score above threshold.

		Parameters
		----------
		min_score
		    Minimum score threshold.

		Returns
		-------
		list[tuple[str, float]]
		    List of (head_id, score) tuples, sorted by score descending.
		"""
		all_candidates: list[tuple[str, float]] = []
		for candidates in self.candidates_by_model.values():
			all_candidates.extend(
				(head, score) for head, score in candidates if score >= min_score
			)
		return sorted(all_candidates, key=lambda x: x[1], reverse=True)

	def to_dataframe(self) -> pl.DataFrame:
		"""Convert candidates to a Polars DataFrame."""
		rows: list[dict] = []
		for model, candidates in self.candidates_by_model.items():
			n_layers = self.model_n_layers.get(model, 1)
			for head, score in candidates:
				_, layer, head_idx = parse_cls(head)
				layer_depth = layer / (n_layers - 1) if n_layers > 1 else 0.0
				rows.append(
					{
						"head": head,
						"model": model,
						"layer": layer,
						"head_idx": head_idx,
						"score": score,
						"layer_depth": layer_depth,
					}
				)
		return pl.DataFrame(rows).sort("score", descending=True)

	@classmethod
	def read(cls, path: Path | str) -> DistanceCandidates:
		"""Load candidate heads from a saved file."""
		import json

		path = Path(path)
		data = json.loads(path.read_text())
		return cls(
			reference_heads=data["reference_heads"],
			candidates_by_model=data["candidates_by_model"],
			all_neighbors=data.get("all_neighbors", {}),
			k_neighbors=data.get("k_neighbors", 10),
			model_n_layers=data.get("model_n_layers", {}),
		)

	def save(self, path: Path | str) -> None:
		"""Save candidate heads to a file."""
		import json

		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)

		data = {
			"reference_heads": self.reference_heads,
			"candidates_by_model": self.candidates_by_model,
			"all_neighbors": self.all_neighbors,
			"k_neighbors": self.k_neighbors,
			"model_n_layers": self.model_n_layers,
		}
		path.write_text(json.dumps(data, indent=2))


def find_candidate_induction_heads(
	distance_result: DistanceTensorResult,
	reference_heads: list[str] | None = None,
	k_neighbors: int = 20,
	exclude_reference_model: bool = True,
	score_method: str = "frequency",
) -> DistanceCandidates:
	"""Find candidate induction heads based on proximity in embedding space.

	For each reference (known) induction head, finds the K nearest neighbors
	from other models. Heads that appear frequently across multiple reference
	heads are scored higher.

	Parameters
	----------
	distance_result
	    Pre-computed distance tensor result.
	reference_heads
	    Known induction heads to use as reference. If None, loads from AttentionPedia.
	k_neighbors
	    Number of neighbors to consider per reference head.
	exclude_reference_model
	    If True, excludes heads from the same model as reference heads.
	score_method
	    How to score candidates:
	    - "frequency": Score by how often head appears in top-K (default)
	    - "inverse_distance": Score by sum of 1/distance across appearances

	Returns
	-------
	DistanceCandidates
	    Container with candidate heads organized by model.
	"""
	if reference_heads is None:
		reference_heads = get_known_induction_heads()

	# Filter to only reference heads that exist in the distance matrix
	available_heads = set(distance_result.cls_values)
	reference_heads = [h for h in reference_heads if h in available_heads]

	# Compute n_layers per model from cls_values
	model_n_layers: dict[str, int] = {}
	for head in distance_result.cls_values:
		model, layer, _ = parse_cls(head)
		model_n_layers[model] = max(model_n_layers.get(model, 0), layer + 1)

	if not reference_heads:
		raise ValueError(
			"No reference heads found in distance matrix. "
			"Make sure the distance matrix includes the reference model."
		)

	# Get reference model(s) for exclusion
	reference_models: set[str] = set()
	if exclude_reference_model:
		for head in reference_heads:
			model, _, _ = parse_cls(head)
			reference_models.add(model)

	# Collect neighbors for each reference head
	all_neighbors: dict[str, list[tuple[str, float]]] = {}
	for ref_head in reference_heads:
		# Get more neighbors than k to allow for filtering
		neighbors = distance_result.get_closest_heads(
			ref_head, n_closest=k_neighbors * 3
		)

		# Filter out reference model heads and self
		filtered_neighbors: list[tuple[str, float]] = []
		for neighbor_head, dist in neighbors:
			if neighbor_head == ref_head:
				continue
			neighbor_model, _, _ = parse_cls(neighbor_head)
			if exclude_reference_model and neighbor_model in reference_models:
				continue
			filtered_neighbors.append((neighbor_head, dist))

		all_neighbors[ref_head] = filtered_neighbors[:k_neighbors]

	# Aggregate scores across all reference heads
	head_scores: defaultdict[str, float] = defaultdict(float)
	head_distances: defaultdict[str, list[float]] = defaultdict(list)

	for ref_head, neighbors in all_neighbors.items():
		for rank, (neighbor_head, dist) in enumerate(neighbors):
			head_distances[neighbor_head].append(dist)

			if score_method == "frequency":
				# Score based on rank position (top ranks get more weight)
				head_scores[neighbor_head] += (k_neighbors - rank) / k_neighbors
			elif score_method == "inverse_distance":
				# Score based on inverse distance
				head_scores[neighbor_head] += 1.0 / (dist + 1e-6)
			else:
				raise ValueError(f"Unknown score_method: {score_method}")

	# Organize by model
	candidates_by_model: dict[str, list[tuple[str, float]]] = defaultdict(list)
	for head, score in head_scores.items():
		model, _, _ = parse_cls(head)
		candidates_by_model[model].append((head, score))

	# Sort each model's candidates by score descending
	for model in candidates_by_model:
		candidates_by_model[model].sort(key=lambda x: x[1], reverse=True)

	return DistanceCandidates(
		reference_heads=reference_heads,
		candidates_by_model=dict(candidates_by_model),
		all_neighbors=all_neighbors,
		k_neighbors=k_neighbors,
		model_n_layers=model_n_layers,
	)


def get_control_heads(
	distance_result: DistanceTensorResult,
	candidate_heads: DistanceCandidates,
	model: str,
	n_controls: int = 10,
	method: str = "far",
) -> list[str]:
	"""Get control (non-induction) heads for baseline comparison.

	Parameters
	----------
	distance_result
	    Pre-computed distance tensor result.
	candidate_heads
	    Candidate heads result (to exclude from controls).
	model
	    Model to get control heads from.
	n_controls
	    Number of control heads to return.
	method
	    How to select controls:
	    - "far": Heads furthest from known induction heads
	    - "random": Random heads (not in candidates)

	Returns
	-------
	list[str]
	    List of control head identifiers.
	"""
	import random

	# Get all heads for this model
	model_heads = [h for h in distance_result.cls_values if h.startswith(f"{model}:")]

	# Exclude candidate heads
	candidate_set = set(h for h, _ in candidate_heads.get_top_candidates(model, n=100))
	available_heads = [h for h in model_heads if h not in candidate_set]

	if method == "random":
		random.shuffle(available_heads)
		return available_heads[:n_controls]

	elif method == "far":
		# Compute mean distance to all reference heads
		head_mean_distances: list[tuple[str, float]] = []
		for head in available_heads:
			distances = []
			for ref_head in candidate_heads.reference_heads:
				try:
					ref_idx = distance_result.cls_values.index(ref_head)
					head_idx = distance_result.cls_values.index(head)
					dist = distance_result.mean_dists[ref_idx, head_idx]
					distances.append(dist)
				except ValueError:
					continue
			if distances:
				head_mean_distances.append((head, sum(distances) / len(distances)))

		# Sort by distance descending (furthest first)
		head_mean_distances.sort(key=lambda x: x[1], reverse=True)
		return [h for h, _ in head_mean_distances[:n_controls]]

	else:
		raise ValueError(f"Unknown method: {method}")
