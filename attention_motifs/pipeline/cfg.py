from copy import deepcopy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from attention_motifs.consts import DEFAULT_COMPRESS_LEVEL
from attention_motifs.util.model_name import cached_sanitize_model_name

ClusteringMethod = Literal["hierarchical", "hdbscan", "leiden"]
LinkageMethod = Literal["ward", "average", "complete", "single"]

PIPELINE_CFG_EXAMPLES: str = """
# use `pipeline_cfg.toml`
python {script_path}
# use `pipeline_cfg.toml` and override some parameters
python {script_path} --prompts-n-samples 1000 --models gpt2-small,pythia-14m
# use `some/path/my_cfg.toml`
python {script_path} some/path/my_cfg.toml
# use `some/path/my_cfg.toml` and override some parameters
python {script_path} some/path/my_cfg.toml --prompts-n-samples 1000 --models gpt2-small,pythia-14m
""".strip()


DataFilename = Literal[
	"raw",
	"norms",
	"scaled",
	"pca",
	"pca_npy",
	"pca_web",
	"head_dists_zanj",
	"head_dists_raw",
	"head_embed",
	"head_embed_clustered",
	"clustering",
	"clustering_hdbscan",
	"clustering_leiden",
	"cluster_trends",
	"importance",
]
FigureFilename = Literal[
	"pca", "cov_full", "cov_reduced", "pca_all", "head_dists", "head_embed"
]

PlotKwargKey = Literal["pca_all_dpi", "n_dims"]

EmbeddingMethod = Literal["isomap", "umap", "tsne", "pca"]

DATA_FNAMES: dict[DataFilename, str] = {
	"raw": "raw.jsonl",
	"norms": "norms.jsonl",
	"scaled": "scaled.jsonl",
	"importance": "importance.jsonl",
	"pca": "pca.jsonl",
	"pca_npy": "pca.npy",
	"pca_web": "pca_web.csv",
	"head_dists_zanj": "head_dists.zanj",
	"head_dists_raw": "head_dists_raw",
	"head_embed": "head_embed.jsonl",
	"head_embed_clustered": "head_embed_clustered.jsonl",
	"clustering": "clustering",
	"clustering_hdbscan": "clustering_hdbscan",
	"clustering_leiden": "clustering_leiden",
	"cluster_trends": "cluster_trends.json",
}

FIGURE_FNAMES: dict[FigureFilename, str] = {
	"pca": "pca.pdf",
	"cov_full": "covariance-full.pdf",
	"cov_reduced": "covariance-reduced.pdf",
	"pca_all": "pca-all.png",
	"head_dists": "head-dists-heatmap.pdf",
	"head_embed": "head-embed.pdf",
}


DEFAULT_VIS_CONFIGS: dict[str, dict[str, Any]] = dict(
	attentionpedia=dict(
		path="attnpedia",
		cfg_path="config.json",
		cfg={
			"head_viewing": "gpt2-small:L5:H5",
			"table": {"n_nearby": 2, "n_share_class": 2},
			"n_prompts": 5,
			"pattern_size": 120,
			"attnpedia_url": "ap.json",
			"headDistsmeta_url": "../../features/head_dists_raw/dists_meta.json",
			"prompts_url": "../../patterns/gpt2-small/prompts.jsonl",
			"headDistsnpy_url": "../../features/head_dists_raw/distances.npy",
			"patterns_path": "../../patterns/",
		},
	),
	embed_pattern=dict(
		path="embeds/patterns/",
		cfg_path="config.json",
		cfg={
			"dataFile": "../../../features/pca_web.csv",
			"numericalPrefix": "pc.",
			"defaultColorColumn": "activation.model",
			"defaultSelectionColumn": "activation.model",
			"hoverColumns": ["activation.cls", "activation.prompt", "activation.n_ctx"],
			"selectedPoints": {
				"size": 5,
				"sizeMin": 0.1,
				"sizeMax": 20,
				"sizeStep": 0.1,
				"opacityMin": 0.0,
				"opacityStep": 0.01,
			},
			"nonSelectedPoints": {
				"size": 3,
				"sizeMin": 0.1,
				"sizeStep": 0.1,
				"opacityMin": 0.00,
			},
			"movement": {
				"speed": 25,
				"speedMin": 1,
				"rollSpeed": 0.02,
				"mouseSensitivity": 0.002,
				"sprintMultiplier": 3,
			},
			"rightClick": {
				"mode": "url",
				"url": {
					"template": "../../../patterns/single.html?prompt={activation.prompt}&head={activation.model}.L{activation.layer}.H{activation.head}"
				},
			},
			"middleClick": {
				"enabled": True,
				"title": "{activation.cls}",
				"content": '<img src="../../../patterns/{activation.model}/prompts/{activation.prompt}/L{activation.layer}/H{activation.head}/attn.png" style="width: 300px; height: 300px; image-rendering: pixelated;" draggable="false" />',
			},
		},
	),
	embed_head=dict(
		path="embeds/heads/",
		cfg_path="config.json",
		cfg={
			"dataFile": "../../../features/head_embed.jsonl",
			"numericalPrefix": "embed.",
			"defaultColorColumn": "type.group",
			"defaultSelectionColumn": "type.group",
			"hoverColumns": ["cls", "type.primary", "type.group"],
			"clustering_methods_url": "../../../features/clustering_methods.json",
			"clustering_meta_url": "../../../features/clustering/clustering_meta.json",
			"clustering_linkage_url": "../../../features/clustering/linkage.json",
			"cluster_labels_url": "../../../features/clustering/cluster_labels.json",
			"clustering_hdbscan_meta_url": "../../../features/clustering_hdbscan/clustering_meta.json",
			"clustering_hdbscan_partitions_url": "../../../features/clustering_hdbscan/partitions.json",
			"clustering_hdbscan_labels_url": "../../../features/clustering_hdbscan/cluster_labels.json",
			"clustering_leiden_meta_url": "../../../features/clustering_leiden/clustering_meta.json",
			"clustering_leiden_partitions_url": "../../../features/clustering_leiden/partitions.json",
			"clustering_leiden_labels_url": "../../../features/clustering_leiden/cluster_labels.json",
			"selectedPoints": {
				"size": 20,
				"sizeMin": 0.1,
				"sizeMax": 50,
				"sizeStep": 0.1,
				"opacityMin": 0.0,
				"opacityStep": 0.01,
			},
			"nonSelectedPoints": {
				"size": 10,
				"sizeMin": 0.1,
				"sizeMax": 50,
				"sizeStep": 0.1,
				"opacityMin": 0.00,
			},
			"movement": {
				"speed": 25,
				"speedMin": 1,
				"rollSpeed": 0.02,
				"mouseSensitivity": 0.002,
				"sprintMultiplier": 3,
			},
			"rightClick": {
				"mode": "url",
				"url": {"template": "../../attnpedia/index.html?head_viewing={cls}"},
			},
			"middleClick": {
				"enabled": True,
				"title": "{cls}",
				"content": '<pre style="margin:0;white-space:pre-wrap;">{cls}</pre>',
			},
			"customPanels": [
				{
					"id": "clustering",
					"title": "Clustering",
					"key": "Digit1",
					"shortcutText": "1 \u2013 clustering",
					"visible": False,
					"position": {"top": "20px", "right": "20px"},
					"html": (
						'<div style="margin-bottom:8px;">'
						'<label><input type="checkbox" id="clusterEnabled" checked> Enable cluster coloring</label>'
						"</div>"
						'<div style="margin-bottom:6px;">'
						'<label>Method: <select id="clusterMethod" style="color:#0f0;background:#222;border:1px solid #555;"></select></label>'
						"</div>"
						'<div id="clusterCutHeightRow" style="margin-bottom:6px;">'
						'<label>Cut Height: <span id="clusterCutHeightValue" style="color:#0f0">5.00</span></label>'
						'<input type="range" id="clusterCutHeight" min="0" max="10" step="0.01" value="5" style="width:100%">'
						"</div>"
						'<div id="clusterParamRow" style="margin-bottom:6px;display:none;">'
						'<label><span id="clusterParamLabel">Parameter:</span> '
						'<select id="clusterParamSelect" style="color:#0f0;background:#222;border:1px solid #555;"></select></label>'
						"</div>"
						'<div style="margin-bottom:6px;">'
						'<label>Min Cluster Size: <span id="clusterMinSizeValue" style="color:#0f0">0</span></label>'
						'<input type="range" id="clusterMinSize" min="0" max="50" step="1" value="0" style="width:100%">'
						"</div>"
						'<div id="clusterStats" style="font-size:11px;color:#888;margin-top:8px;"></div>'
					),
				}
			],
		},
	),
)


def deep_merge_dicts(
	dict1: dict,
	dict2: dict,
) -> dict:
	"""Recursively merge two dictionaries into a new dict, with dict2 overwriting dict1"""
	output: dict = deepcopy(dict1)
	for key, value in dict2.items():
		if isinstance(value, dict) and key in output and isinstance(output[key], dict):
			output[key] = deep_merge_dicts(output[key], value)
		else:
			output[key] = value
	return output


def _ser_path(path: Path) -> str | None:
	"""Serialize a Path object to a string"""
	return path.as_posix() if path is not None else None


def _deser_path(path_str: str | None) -> Path | None:
	"""Deserialize a string to a Path object"""
	return Path(path_str) if path_str is not None else None


@dataclass(kw_only=True)
class PipelineConfig:
	"""Configuration for the attention motifs pipeline"""

	# input paths
	prompts_file: Path
	patterns_dir: Path

	# prompts processing
	prompts_n_samples: int
	prompts_min_chars: int
	prompts_max_chars: int

	# data processing
	models: list[str]
	pca_n_components: int = 16

	# s3: web CSV sampling (subset of prompts for pca_web.csv)
	web_pca_n_prompts: int | None = None  # None = write all prompts to web CSV
	web_pca_seed: int = 42

	# head embedding configuration
	embedding_methods: list[EmbeddingMethod] = field(
		default_factory=lambda: ["isomap", "umap", "tsne", "pca"]
	)
	embedding_n_components_list: list[int] = field(default_factory=lambda: [2, 3])
	embedding_n_neighbors_list: list[int] = field(
		default_factory=lambda: [2, 4, 8, 16, 32, 64]
	)

	# clustering configuration (from [clustering] TOML section)
	clustering_methods: list[ClusteringMethod] = field(
		default_factory=lambda: ["hierarchical", "hdbscan", "leiden"]
	)
	# hierarchical (agglomerative, scipy)
	clustering_hierarchical_linkage_method: LinkageMethod = "average"
	clustering_hierarchical_n_clusters_list: list[int] = field(
		default_factory=lambda: [5, 10, 20, 50]
	)
	# HDBSCAN (density-based, sklearn)
	clustering_hdbscan_min_cluster_sizes: list[int] = field(
		default_factory=lambda: [3, 5, 10, 20]
	)
	# Leiden (graph community detection, igraph)
	clustering_leiden_resolutions: list[float] = field(
		default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0]
	)
	clustering_leiden_n_neighbors: int = 10

	# computing
	n_proc: int
	s2_chunksize: int = 4
	s4_n_proc: int | None = None
	force_overwrite: bool = False
	device: str = "cpu"
	smart_mode: bool = False
	estimate_memory_only: bool = False

	# multi-model parallelism (s1)
	parallel_models: bool = False
	devices: list[str] = field(default_factory=lambda: ["cuda:0"])
	vram_safety_factor: float = 10.0
	cuda_context_bytes: int = 500_000_000
	batch_size: int = 32
	compress_level: int = DEFAULT_COMPRESS_LEVEL

	# s1b: render patterns
	render_patterns_enabled: bool = True
	render_n_samples: int | None = None  # None = render all prompts_n_samples
	render_seed: int = 42

	# output paths
	features_dir: Path
	data_fnames: dict[DataFilename, str] = field(
		default_factory=lambda: DATA_FNAMES,
	)
	vis_dir: Path = Path("data/vis")
	vis_configs: dict[str, dict[str, Any]] = field(
		default_factory=lambda: DEFAULT_VIS_CONFIGS,
	)

	# ablation configuration (from [ablation] TOML section)
	# empty dict = skip s7_ablation step
	ablation: dict[str, Any] = field(default_factory=dict)

	# plotting/logging
	figures_dir: Path | None = None
	figures_fnames: dict[FigureFilename, str] = field(
		default_factory=lambda: FIGURE_FNAMES,
	)
	plot_kwargs: dict[PlotKwargKey, int] = field(
		default_factory=lambda: {
			# default DPI for PCA all figure
			"pca_all_dpi": 500,
			# default number of dimensions for PCA and other embeddings
			"n_dims": 5,  # type: ignore[dict-item]
		},
	)
	verbose: int = 1

	@property
	def do_figures(self) -> bool:
		"""Check if figures are enabled"""
		return self.figures_dir is not None

	def data_path(self, fname: DataFilename) -> Path:
		return self.features_dir / self.data_fnames[fname]

	def figure_path(self, fname: FigureFilename) -> Path:
		if self.figures_dir is None:
			raise ValueError(
				"figures_dir is not set, this means figures should be disabled"
			)
		return self.figures_dir / self.figures_fnames[fname]

	def compute_hash(self) -> str:
		"""Compute a stable SHA-256 hash of the configuration.

		The hash is computed from a canonicalized JSON representation
		of config fields that affect output content. Runtime settings
		like n_proc, device, verbose are excluded.
		"""
		import hashlib
		import json

		# Fields that affect output content (exclude runtime settings)
		config_dict: dict = dict(
			prompts_file=str(self.prompts_file),
			patterns_dir=str(self.patterns_dir),
			prompts_n_samples=self.prompts_n_samples,
			prompts_min_chars=self.prompts_min_chars,
			prompts_max_chars=self.prompts_max_chars,
			models=sorted(self.models),
			pca_n_components=self.pca_n_components,
			web_pca_n_prompts=self.web_pca_n_prompts,
			web_pca_seed=self.web_pca_seed,
			embedding_methods=sorted(self.embedding_methods),
			embedding_n_components_list=sorted(self.embedding_n_components_list),
			embedding_n_neighbors_list=sorted(self.embedding_n_neighbors_list),
			clustering_methods=sorted(self.clustering_methods),
			clustering_hierarchical_linkage_method=self.clustering_hierarchical_linkage_method,
			clustering_hierarchical_n_clusters_list=sorted(
				self.clustering_hierarchical_n_clusters_list
			),
			clustering_hdbscan_min_cluster_sizes=sorted(
				self.clustering_hdbscan_min_cluster_sizes
			),
			clustering_leiden_resolutions=sorted(self.clustering_leiden_resolutions),
			clustering_leiden_n_neighbors=self.clustering_leiden_n_neighbors,
			render_patterns_enabled=self.render_patterns_enabled,
			render_n_samples=self.render_n_samples,
			render_seed=self.render_seed,
			data_fnames=self.data_fnames,
			plot_kwargs=self.plot_kwargs,
			features_dir=str(self.features_dir),
			vis_dir=str(self.vis_dir),
			figures_dir=str(self.figures_dir) if self.figures_dir else None,
		)

		json_str: str = json.dumps(config_dict, sort_keys=True)
		hash_digest: str = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
		return hash_digest

	def validate_cfg(self) -> None:
		# TODO: check models actually exist in TransformerLens?
		assert all(isinstance(model, str) for model in self.models), (
			"All models must be strings"
		)
		assert isinstance(self.prompts_n_samples, int) and self.prompts_n_samples > 0, (
			"prompts_n_samples must be a positive integer"
		)
		# assert self.prompts_file.is_file(), (
		# 	f"prompts_file {self.prompts_file} does not exist"
		# )
		# assert not self.patterns_dir.is_file(), (
		# 	f"patterns_dir {self.patterns_dir} must be a directory"
		# )
		# assert not self.features_dir.is_file(), (
		# 	f"features_dir {self.features_dir} must be a directory"
		# )
		assert isinstance(self.n_proc, int) and self.n_proc > 0, (
			"n_proc must be a positive integer"
		)
		assert isinstance(self.force_overwrite, bool), (
			"force_overwrite must be a boolean"
		)
		assert isinstance(self.device, str), (
			"device must be a string representing the torch device (e.g., 'cpu', 'cuda')"
		)
		assert self.prompts_min_chars >= 0, (
			"prompts_min_chars must be a non-negative integer"
		)
		assert self.prompts_max_chars >= self.prompts_min_chars, (
			"prompts_max_chars must be greater than or equal to prompts_min_chars"
		)

		assert isinstance(self.devices, list) and len(self.devices) > 0, (
			"devices must be a non-empty list of device strings"
		)
		assert (
			isinstance(self.vram_safety_factor, (int, float))
			and self.vram_safety_factor > 0
		), "vram_safety_factor must be a positive number"
		assert (
			isinstance(self.cuda_context_bytes, int) and self.cuda_context_bytes >= 0
		), "cuda_context_bytes must be a non-negative integer"

		# s3 web PCA sampling validation
		if self.web_pca_n_prompts is not None:
			assert (
				isinstance(self.web_pca_n_prompts, int) and self.web_pca_n_prompts > 0
			), "web_pca_n_prompts must be a positive integer or None"
		assert isinstance(self.web_pca_seed, int), "web_pca_seed must be an integer"

		# s1b render patterns validation
		if self.render_n_samples is not None:
			assert (
				isinstance(self.render_n_samples, int) and self.render_n_samples > 0
			), "render_n_samples must be a positive integer or None"
		assert isinstance(self.render_seed, int), "render_seed must be an integer"

		# Basic validation for embedding parameters
		valid_methods = {"isomap", "umap", "tsne", "pca"}
		assert all(method in valid_methods for method in self.embedding_methods), (
			f"embedding_methods must be subset of {valid_methods}"
		)

		# Clustering validation
		valid_clustering_methods: set[str] = {"hierarchical", "hdbscan", "leiden"}
		assert len(self.clustering_methods) > 0, "clustering_methods must not be empty"
		assert all(m in valid_clustering_methods for m in self.clustering_methods), (
			f"clustering_methods must be subset of {valid_clustering_methods}"
		)
		valid_linkage: set[str] = {"ward", "average", "complete", "single"}
		assert self.clustering_hierarchical_linkage_method in valid_linkage, (
			f"clustering_hierarchical_linkage_method must be one of {valid_linkage}"
		)
		assert (
			isinstance(self.clustering_hierarchical_n_clusters_list, list)
			and len(self.clustering_hierarchical_n_clusters_list) > 0
			and all(k >= 2 for k in self.clustering_hierarchical_n_clusters_list)
		), (
			"clustering_hierarchical_n_clusters_list must be a non-empty list of integers >= 2"
		)
		assert all(s >= 2 for s in self.clustering_hdbscan_min_cluster_sizes), (
			"clustering_hdbscan_min_cluster_sizes values must be >= 2"
		)
		assert all(r > 0 for r in self.clustering_leiden_resolutions), (
			"clustering_leiden_resolutions values must be > 0"
		)
		assert self.clustering_leiden_n_neighbors >= 1, (
			"clustering_leiden_n_neighbors must be >= 1"
		)

	def as_str(self) -> str:
		"""Return a string representation of the configuration"""
		return "\n".join(
			[
				"PipelineConfig(",
				"\n".join(
					[
						f"  {field.name}={getattr(self, field.name)!r},"
						for field in self.__dataclass_fields__.values()
						# if field.name not in ("data_fnames", "figures_fnames")
					]
				),
				")",
			]
		)

	def __str__(self) -> str:
		"""Return a string representation of the configuration"""
		return self.as_str()

	def __repr__(self) -> str:
		"""Return a string representation of the configuration"""
		return self.as_str()

	@staticmethod
	def _load_clustering_config(data: dict) -> dict[str, Any]:
		"""Extract clustering config from a TOML data dict.

		Reads from [clustering] section if present, with backward compatibility
		for old flat top-level keys (clustering_linkage_method, etc.).
		"""
		c: dict[str, Any] = data.get("clustering", {})
		# Backward compat: fall back to old flat top-level keys
		return dict(
			clustering_methods=c.get(
				"methods",
				data.get("clustering_methods", ["hierarchical", "hdbscan", "leiden"]),
			),
			clustering_hierarchical_linkage_method=c.get(
				"hierarchical_linkage_method",
				data.get("clustering_linkage_method", "average"),
			),
			clustering_hierarchical_n_clusters_list=c.get(
				"hierarchical_n_clusters_list",
				data.get("clustering_n_clusters_list", [5, 10, 20, 50]),
			),
			clustering_hdbscan_min_cluster_sizes=c.get(
				"hdbscan_min_cluster_sizes",
				data.get("clustering_hdbscan_min_cluster_sizes", [3, 5, 10, 20]),
			),
			clustering_leiden_resolutions=c.get(
				"leiden_resolutions",
				data.get("clustering_leiden_resolutions", [0.1, 0.25, 0.5, 1.0, 2.0]),
			),
			clustering_leiden_n_neighbors=c.get(
				"leiden_n_neighbors",
				data.get("clustering_leiden_n_neighbors", 10),
			),
		)

	@classmethod
	def load(cls, data: dict) -> "PipelineConfig":
		"""Load configuration from a dictionary"""
		config: "PipelineConfig" = cls(
			prompts_n_samples=data["prompts_n_samples"],
			n_proc=data["n_proc"],
			prompts_file=Path(data["prompts_file"]),
			patterns_dir=Path(data["patterns_dir"]),
			features_dir=Path(data["features_dir"]),
			vis_dir=Path(data.get("vis_dir", "data/vis")),
			vis_configs=deep_merge_dicts(
				DEFAULT_VIS_CONFIGS,
				data.get("vis_configs", DEFAULT_VIS_CONFIGS),
			),
			models=data["models"],
			pca_n_components=data.get(
				"pca_n_components", 16
			),  # default to 16 if not specified
			web_pca_n_prompts=data.get("web_pca_n_prompts", None),
			web_pca_seed=data.get("web_pca_seed", 42),
			embedding_methods=data.get(
				"embedding_methods", ["isomap", "umap", "tsne", "pca"]
			),
			embedding_n_components_list=data.get("embedding_n_components_list", [2, 3]),
			embedding_n_neighbors_list=data.get(
				"embedding_n_neighbors_list", [2, 4, 8, 16, 32, 64]
			),
			**cls._load_clustering_config(data),
			prompts_min_chars=data["prompts_min_chars"],
			prompts_max_chars=data["prompts_max_chars"],
			device=data.get("device", "cpu"),  # default to 'cpu' if not specified
			force_overwrite=data.get("force_overwrite", False),  # default to False
			parallel_models=data.get("parallel_models", False),
			devices=data.get("devices", [data.get("device", "cpu")]),
			vram_safety_factor=data.get("vram_safety_factor", 10.0),
			cuda_context_bytes=data.get("cuda_context_bytes", 500_000_000),
			batch_size=data.get("batch_size", 32),
			compress_level=data.get("compress_level", DEFAULT_COMPRESS_LEVEL),
			s2_chunksize=data.get("s2_chunksize", 4),
			s4_n_proc=data.get("s4_n_proc", None),
			render_patterns_enabled=data.get("render_patterns_enabled", True),
			render_n_samples=data.get("render_n_samples", None),
			render_seed=data.get("render_seed", 42),
			figures_dir=(
				Path(data["figures_dir"]) if "figures_dir" in data else None
				# default to None if not specified
			),
			verbose=data.get("verbose", 1),  # default to 1 if not specified
			data_fnames={**DATA_FNAMES, **data.get("data_fnames", DATA_FNAMES)},
			figures_fnames={
				**FIGURE_FNAMES,
				**data.get("figures_fnames", FIGURE_FNAMES),
			},
			plot_kwargs=data.get("plot_kwargs", {}),
			smart_mode=data.get("smart_mode", False),
			ablation=data.get("ablation", {}),
		)
		config.models = [cached_sanitize_model_name(m) for m in config.models]
		config.validate_cfg()
		return config

	@classmethod
	def read(cls, path: Path) -> "PipelineConfig":
		"""Load configuration from a TOML file."""
		with path.open("rb") as f:
			data: dict = tomllib.load(f)
		return cls.load(data)

	@classmethod
	def from_cli(cls, argv: list[str]) -> "PipelineConfig":
		"""returns a PipelineConfig from CLI

		The first (optional) positional argument is a path to a TOML file.
		If omitted, the file `pipeline_cfg.toml` in the current working
		directory is assumed.  Any keyword flags override values loaded
		from the TOML.

		# Usage:
		```
		# use `pipeline_cfg.toml`
		python scrips/run_pipeline.py
		# use `pipeline_cfg.toml` and override some parameters
		python scrips/run_pipeline.py --prompts-n-samples 1000 --models gpt2-small,pythia-14m
		# use `some/path/my_cfg.toml`
		python scrips/run_pipeline.py some/path/my_cfg.toml
		# use `some/path/my_cfg.toml` and override some parameters
		python scrips/run_pipeline.py some/path/my_cfg.toml --prompts-n-samples 1000 --models gpt2-small,pythia-14m
		```
		"""
		import argparse

		parser: argparse.ArgumentParser = argparse.ArgumentParser(
			description="Attention-motifs pipeline configuration"
		)
		# optional positional path to the config file
		parser.add_argument(
			"config_path",
			nargs="?",
			default="pipeline_cfg.toml",
			type=Path,
			help="Path to a TOML config file (default: pipeline_cfg.toml)",
		)

		# overrideable fields
		parser.add_argument(
			"--models", type=str, help="Comma-separated list of model names"
		)
		parser.add_argument(
			"--prompts-n-samples", type=int, help="Number of samples per prompt"
		)
		parser.add_argument("--n_proc", type=int, help="Number of parallel processes")
		parser.add_argument(
			"--prompts-file", type=Path, help="Path to the prompts file"
		)
		parser.add_argument(
			"--patterns-dir", type=Path, help="Directory for learned patterns"
		)
		parser.add_argument(
			"--features-dir", type=Path, help="Directory for extracted features"
		)
		parser.add_argument(
			"--vis-dir",
			type=Path,
			default=None,
			help="Directory for visualizations (default: 'vis')",
		)
		parser.add_argument(
			"--device",
			type=str,
			default=None,
			help="torch device to use",
		)
		parser.add_argument(
			"--force_overwrite",
			action="store_true",
			help="Force overwrite existing files",
		)
		parser.add_argument(
			"--figures-dir",
			type=Path,
			default=None,
			help="Directory for plots (default: None, no plots)",
		)
		parser.add_argument(
			"--verbose",
			type=int,
			default=1,
			help="Verbosity level (default: 1). Higher values mean more output",
		)
		parser.add_argument(
			"--smart",
			action="store_true",
			help="Enable smart mode: skip previously completed steps if config unchanged",
		)
		parser.add_argument(
			"--estimate-memory-usage",
			action="store_true",
			help="Print estimated s4 memory usage and exit (no pipeline runs)",
		)
		parser.add_argument(
			"--parallel-models",
			action="store_true",
			help="Enable VRAM-aware parallel model scheduling for s1",
		)
		parser.add_argument(
			"--devices",
			type=str,
			default=None,
			help="Comma-separated list of CUDA devices (e.g., 'cuda:0,cuda:1')",
		)
		parser.add_argument(
			"--vram-safety-factor",
			type=float,
			default=None,
			help="Multiplied with n_params to estimate VRAM in bytes (default: 10.0). "
			"Must account for dtype size, activation caches, and allocator overhead. "
			"Batch-size memory is not modelled — adjust this if needed.",
		)
		parser.add_argument(
			"--cuda-context-bytes",
			type=int,
			default=None,
			help="Fixed CUDA context overhead in bytes for VRAM estimation (default: 500000000)",
		)
		parser.add_argument(
			"--batch-size",
			type=int,
			default=None,
			help="Batch size for activation generation (default: 32)",
		)
		parser.add_argument(
			"--compress-level",
			type=int,
			default=None,
			help="Compression level for .npz saves: 0=none, 1=fast, 6=default (default: 6)",
		)
		parser.add_argument(
			"--s2-chunksize",
			type=int,
			default=None,
			help="Chunksize for multiprocessing pool.imap in s2 feature computation (default: 4)",
		)
		parser.add_argument(
			"--s4-n-proc",
			type=int,
			default=None,
			help="Number of processes for s4 head distance computation (default: n_proc)",
		)

		args: argparse.Namespace = parser.parse_args(argv)

		# 1. Load the base configuration from the TOML file
		config: PipelineConfig = cls.read(args.config_path)

		# 2. Apply any command-line overrides
		if args.models is not None:
			config.models = [m.strip() for m in args.models.split(",") if m.strip()]
		if args.prompts_n_samples is not None:
			config.prompts_n_samples = args.prompts_n_samples
		if args.n_proc is not None:
			config.n_proc = args.n_proc
		if args.prompts_file is not None:
			config.prompts_file = args.prompts_file
		if args.patterns_dir is not None:
			config.patterns_dir = args.patterns_dir
		if args.features_dir is not None:
			config.features_dir = args.features_dir
		if args.vis_dir is not None:
			config.vis_dir = args.vis_dir
		if args.device is not None:
			config.device = args.device
		if args.force_overwrite is not None:
			config.force_overwrite = args.force_overwrite
		if args.figures_dir is not None:
			config.figures_dir = args.figures_dir
		if args.smart:
			config.smart_mode = True
		if args.estimate_memory_usage:
			config.estimate_memory_only = True
		if args.parallel_models:
			config.parallel_models = True
		if args.devices is not None:
			config.devices = [d.strip() for d in args.devices.split(",") if d.strip()]
		if args.vram_safety_factor is not None:
			config.vram_safety_factor = args.vram_safety_factor
		if args.cuda_context_bytes is not None:
			config.cuda_context_bytes = args.cuda_context_bytes
		if args.batch_size is not None:
			config.batch_size = args.batch_size
		if args.compress_level is not None:
			config.compress_level = args.compress_level
		if args.s2_chunksize is not None:
			config.s2_chunksize = args.s2_chunksize
		if args.s4_n_proc is not None:
			config.s4_n_proc = args.s4_n_proc

		# 3. Sanitize model names (CLI may have provided raw aliases)
		config.models = [cached_sanitize_model_name(m) for m in config.models]

		# 4. Final sanity check
		config.validate_cfg()
		return config


def pipeline_step_major(msg: str) -> None:
	"""Print a message for a pipeline step"""
	# print(f"\033[94m==================== {msg} ====================\033[m")
	import shutil

	term_width: int = shutil.get_terminal_size((80, 20)).columns
	print(f"\033[94m{'=' * term_width}\033[m", flush=True)
	print(f"\033[94m{msg.center(term_width)}\033[m", flush=True)
	print(f"\033[94m{'=' * term_width}\033[m", flush=True)


def pipeline_model_progress(idx: int, total: int, name: str) -> None:
	"""Print model progress in cyan for visibility"""
	print(f"\033[96mprocessing model {idx + 1} / {total}: {name}\033[m", flush=True)
