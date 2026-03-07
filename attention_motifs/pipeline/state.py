"""Pipeline state tracking for smart mode."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
import json
import warnings

StepName = Literal[
	"s1_activations",
	"s1b_render_patterns",
	"s1c_write_idxs",
	"s2_features",
	"s3_feat_proc",
	"s3b_feat_fig",
	"s4_head_dist",
	"s4b_write_frontend",
	"s4c_clustering",
	"s5_head_embed",
	"s5b_head_embed_plots",
	"s6_clustered_embed",
	"s6b_cluster_trends",
	"s6c_write_cluster_frontend",
	"s7_ablation",
]


@dataclass
class StepRecord:
	"""Record of a completed pipeline step."""

	name: StepName
	completed_at: str  # ISO format timestamp
	duration_seconds: float

	def serialize(self) -> dict[str, str | float]:
		"""Convert to JSON-compatible dict."""
		return dict(
			name=self.name,
			completed_at=self.completed_at,
			duration_seconds=self.duration_seconds,
		)

	@classmethod
	def load(cls, data: dict[str, str | float]) -> "StepRecord":
		"""Create instance from dict."""
		return cls(
			name=data["name"],  # type: ignore[arg-type]
			completed_at=str(data["completed_at"]),
			duration_seconds=float(data["duration_seconds"]),
		)


@dataclass
class PipelineState:
	"""Tracks pipeline execution state for smart mode."""

	config_hash: str
	completed_steps: dict[StepName, StepRecord] = field(default_factory=dict)
	last_run: str | None = None  # ISO format timestamp

	def serialize(self) -> dict:
		"""Convert to JSON-compatible dict."""
		return dict(
			config_hash=self.config_hash,
			completed_steps={
				name: record.serialize()
				for name, record in self.completed_steps.items()
			},
			last_run=self.last_run,
		)

	@classmethod
	def load(cls, data: dict) -> "PipelineState":
		"""Create instance from dict."""
		return cls(
			config_hash=data["config_hash"],
			completed_steps={
				name: StepRecord.load(record)
				for name, record in data.get("completed_steps", {}).items()
			},
			last_run=data.get("last_run"),
		)

	def save(self, path: Path | str) -> None:
		"""Write state to JSON file."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, "w") as f:
			json.dump(self.serialize(), f, indent=2)

	@classmethod
	def read(cls, path: Path | str) -> "PipelineState":
		"""Load state from JSON file."""
		path = Path(path)
		try:
			with open(path, "r") as f:
				data: dict = json.load(f)
			return cls.load(data)
		except (json.JSONDecodeError, KeyError) as e:
			warnings.warn(
				f"Failed to load pipeline state from {path}: {e}. "
				"Starting with fresh state."
			)
			return cls(config_hash="")

	def is_step_complete(self, step_name: StepName) -> bool:
		"""Check if a step has been completed."""
		return step_name in self.completed_steps

	def mark_step_complete(
		self,
		step_name: StepName,
		duration_seconds: float,
	) -> None:
		"""Mark a step as completed."""
		self.completed_steps[step_name] = StepRecord(
			name=step_name,
			completed_at=datetime.now().isoformat(),
			duration_seconds=duration_seconds,
		)
		self.last_run = datetime.now().isoformat()

	def invalidate_all(self) -> None:
		"""Clear all completed steps (called when config changes)."""
		self.completed_steps = {}
