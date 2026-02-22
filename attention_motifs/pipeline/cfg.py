from copy import deepcopy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
	"head_dists_zanj",
	"head_dists_raw",
	"head_embed",
	"clustering",
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
	"head_dists_zanj": "head_dists.zanj",
	"head_dists_raw": "head_dists_raw",
	"head_embed": "head_embed.jsonl",
	"clustering": "clustering.jsonl",
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
			"dataFile": "../../../features/pca.jsonl",
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

	# head embedding configuration
	embedding_methods: list[EmbeddingMethod] = field(
		default_factory=lambda: ["isomap", "umap", "tsne", "pca"]
	)
	embedding_n_components_list: list[int] = field(default_factory=lambda: [2, 3])
	embedding_n_neighbors_list: list[int] = field(
		default_factory=lambda: [2, 4, 8, 16, 32, 64]
	)

	# computing
	n_proc: int
	force_overwrite: bool = False
	device: str = "cpu"
	smart_mode: bool = False

	# output paths
	features_dir: Path
	data_fnames: dict[DataFilename, str] = field(
		default_factory=lambda: DATA_FNAMES,
	)
	vis_dir: Path = Path("data/vis")
	vis_configs: dict[str, dict[str, Any]] = field(
		default_factory=lambda: DEFAULT_VIS_CONFIGS,
	)

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
			embedding_methods=sorted(self.embedding_methods),
			embedding_n_components_list=sorted(self.embedding_n_components_list),
			embedding_n_neighbors_list=sorted(self.embedding_n_neighbors_list),
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

		# Basic validation for embedding parameters
		valid_methods = {"isomap", "umap", "tsne", "pca"}
		assert all(method in valid_methods for method in self.embedding_methods), (
			f"embedding_methods must be subset of {valid_methods}"
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
			embedding_methods=data.get(
				"embedding_methods", ["isomap", "umap", "tsne", "pca"]
			),
			embedding_n_components_list=data.get("embedding_n_components_list", [2, 3]),
			embedding_n_neighbors_list=data.get(
				"embedding_n_neighbors_list", [2, 4, 8, 16, 32, 64]
			),
			prompts_min_chars=data["prompts_min_chars"],
			prompts_max_chars=data["prompts_max_chars"],
			device=data.get("device", "cpu"),  # default to 'cpu' if not specified
			force_overwrite=data.get("force_overwrite", False),  # default to False
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
		)
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

		# 3. Final sanity check
		config.validate_cfg()
		return config


def pipeline_step_major(msg: str) -> None:
	"""Print a message for a pipeline step"""
	# print(f"\033[94m==================== {msg} ====================\033[m")
	import shutil

	term_width: int = shutil.get_terminal_size((80, 20)).columns
	print(f"\033[94m{'=' * term_width}\033[m")
	print(f"\033[94m{msg.center(term_width)}\033[m")
	print(f"\033[94m{'=' * term_width}\033[m")


def pipeline_model_progress(idx: int, total: int, name: str) -> None:
	"""Print model progress in cyan for visibility"""
	print(f"\033[96mprocessing model {idx + 1} / {total}: {name}\033[m")
