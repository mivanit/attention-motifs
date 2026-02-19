import json
import sys
import matplotlib.colors as mcolors
from typing import Callable, TypedDict
import warnings
from pathlib import Path
import importlib.resources
from collections import defaultdict

import polars as pl

import attention_motifs

ATTNPEDIA_PATH: Path = (
	Path(str(importlib.resources.files(attention_motifs)))
	/ "attnpedia"
	/ "attn-pedia.json"
)

ATTNPEDIA_GROUPS_PATH: Path = (
	Path(str(importlib.resources.files(attention_motifs)))
	/ "attnpedia"
	/ "attn-pedia-groups.json"
)

if not ATTNPEDIA_PATH.is_file():
	warnings.warn(f"attpedia json does not exist: {ATTNPEDIA_PATH = }.")

if not ATTNPEDIA_GROUPS_PATH.is_file():
	warnings.warn(f"attpedia groups json does not exist: {ATTNPEDIA_GROUPS_PATH = }.")


def parse_head(head_id: str) -> tuple[int, int]:
	"""Parse head ID string to (layer, head) tuple.

	Parameters
	----------
	head_id
		String in format 'L{layer}:H{head}' or '{model}:L{layer}:H{head}'.

	Returns
	-------
	tuple[int, int]
		(layer, head) indices.
	"""
	parts = head_id.split(":")
	if len(parts) == 2:
		layer_part, head_part = parts
	elif len(parts) == 3:
		_, layer_part, head_part = parts
	else:
		raise ValueError(f"Invalid head string format: {head_id}")

	layer = int(layer_part.removeprefix("L"))
	head = int(head_part.removeprefix("H"))
	return layer, head


def parse_cls(cls_: str) -> tuple[str, int, int]:
	"""Parse full head ID string to (model, layer, head) tuple.

	Parameters
	----------
	cls_
		String in format '{model}:L{layer}:H{head}'.

	Returns
	-------
	tuple[str, int, int]
		(model, layer, head).
	"""
	model_part, layer_part, head_part = cls_.split(":")
	layer = int(layer_part.removeprefix("L"))
	head = int(head_part.removeprefix("H"))
	return model_part, layer, head


def heads_from_strings(head_strs: list[str]) -> list[tuple[int, int]]:
	"""Convert list of head strings to (layer, head) tuples.

	Parameters
	----------
	head_strs
		List of strings like ['L5:H5', 'L6:H9'] or ['gpt2-small:L5:H5'].

	Returns
	-------
	list[tuple[int, int]]
		List of (layer, head) tuples.
	"""
	return [parse_head(s) for s in head_strs]


class HeadClassSchema(TypedDict):
	"""Schema for a head classification entry.

	Attributes:
		type: Classification type name (e.g. "induction", "previous_token")
		heads: List of head IDs in format "L{layer}:H{head}" (e.g. "L0:H0", "L5:H7")
	"""

	type: str
	heads: list[str]


class PaperGroupSchema(TypedDict):
	"""Schema for a paper/research group entry in the AttentionPedia JSON.

	Attributes:
		prefix: Short identifier for the paper (e.g. "olsson2022")
		url: URL to the paper or resource
		notes: Additional notes about the paper's findings
		model: Model name the classifications apply to (e.g. "gpt2-small")
		classes: List of head classification entries from this paper
	"""

	prefix: str
	url: str
	notes: str
	model: str
	classes: list[HeadClassSchema]


AttentionPediaSchema = list[PaperGroupSchema]


class AttentionPedia:
	def __init__(
		self, path: Path = ATTNPEDIA_PATH, groups_path: Path = ATTNPEDIA_GROUPS_PATH
	):
		self.data_raw: AttentionPediaSchema = json.loads(path.read_text())

		# Load group definitions and colors
		try:
			self.groups_data: dict = json.loads(groups_path.read_text())
			self._unknown_color: str = self.groups_data.get("unknown_color", "#777777")
		except (FileNotFoundError, json.JSONDecodeError):
			warnings.warn(f"Could not load attention groups from {groups_path}.")
			self.groups_data = {"groups": {}}
			self._unknown_color = "#777777"

		# Initialize cache for derived data
		self._head_type_colors: dict[str, str] | None = None
		self._head_type_groups: dict[str, str] | None = None
		self._generate_head_type_maps()

	def dataframe(self) -> pl.DataFrame:
		df_raw: list[dict] = list()
		for group in self.data_raw:
			model_name: str = group["model"]

			for head_cls in group["classes"]:
				for h in head_cls["heads"]:
					layer_idx, head_idx = parse_head(h)

					df_raw.append(
						{
							"paper.prefix": group["prefix"],
							"paper.url": group["url"],
							"paper.notes": group["notes"],
							"paper.name": model_name,
							"type": head_cls["type"],
							"prefixed_type": f"{group['prefix']}:{head_cls['type']}",
							"layer_idx": layer_idx,
							"head_idx": head_idx,
							"head_id": h,
							"head_id_full": f"{model_name}:{h}",
						}
					)

		return pl.DataFrame(df_raw)

	def head_type_tuples(
		self,
		prefixed_type: bool = True,
		prefixed_head: bool = True,
		drop_papers: list[str] | None = None,
	) -> list[tuple[str, str]]:
		"""Returns a dictionary mapping head IDs to types."""
		df_selected: pl.DataFrame = self.dataframe()
		# drop where "paper.prefix" is in drop_papers
		if drop_papers:
			df_selected = df_selected.filter(
				~df_selected["paper.prefix"].is_in(drop_papers)
			)

		return [
			(
				row["head_id_full" if prefixed_head else "head_id"],
				row["prefixed_type" if prefixed_type else "type"],
			)
			for row in df_selected.iter_rows(named=True)
		]

	def head_to_types(
		self,
		prefixed_type: bool = True,
		prefixed_head: bool = True,
		drop_papers: list[str] | None = None,
	) -> dict[str, list[str]]:
		"""Returns a dictionary mapping head IDs to types."""
		head_type_tuples: list[tuple[str, str]] = self.head_type_tuples(
			prefixed_type=prefixed_type,
			prefixed_head=prefixed_head,
			drop_papers=drop_papers,
		)
		head_to_types: defaultdict[str, list[str]] = defaultdict(list)
		for head, type_ in head_type_tuples:
			head_to_types[head].append(type_)
		return dict(head_to_types)

	def type_to_heads(
		self,
		prefixed_type: bool = True,
		prefixed_head: bool = True,
		drop_papers: list[str] | None = None,
	) -> dict[str, list[str]]:
		"""Returns a dictionary mapping types to head IDs."""
		head_type_tuples: list[tuple[str, str]] = self.head_type_tuples(
			prefixed_type=prefixed_type,
			prefixed_head=prefixed_head,
			drop_papers=drop_papers,
		)
		type_to_heads: defaultdict[str, list[str]] = defaultdict(list)
		for head, type_ in head_type_tuples:
			type_to_heads[type_].append(head)
		return dict(type_to_heads)

	def head_to_type(
		self,
		prefixed_type: bool = True,
		prefixed_head: bool = True,
		drop_papers: list[str] | None = None,
		type_select: Callable[[list[str]], str] = lambda x: x[0],
	) -> dict[str, str]:
		"""Returns a dictionary mapping head IDs to a single type

		uses `type_select` to select the type if there are multiple types,
		which by default selects the 0th type.
		"""
		return {
			k: type_select(v)
			for k, v in self.head_to_types(
				prefixed_type=prefixed_type,
				prefixed_head=prefixed_head,
				drop_papers=drop_papers,
			).items()
		}

	def head_type_colors(self, refresh: bool = False) -> dict[str, str]:
		"""Get a mapping of attention head types to color codes.

		Uses consistent coloring where similar head types have similar colors.

		# Parameters:
		 - `refresh : bool`
		    Force regeneration of the color mapping (default: False)

		# Returns:
		 - `dict[str, str]`
		    Dictionary mapping attention head types to hex color codes
		"""
		if self._head_type_colors is None or refresh:
			self._generate_head_type_maps()

		return self._head_type_colors

	def head_type_groups(self, refresh: bool = False) -> dict[str, str]:
		"""Get a mapping of attention head types to their group names.

		# Parameters:
		 - `refresh : bool`
		    Force regeneration of the group mapping (default: False)

		# Returns:
		 - `dict[str, str]`
		    Dictionary mapping attention head types to group names
		"""
		if self._head_type_groups is None or refresh:
			self._generate_head_type_maps()

		return self._head_type_groups

	def _generate_head_type_maps(self) -> None:
		"""Generate color and group mappings for attention head types from the groups JSON."""
		group_data: dict = self.groups_data.get("groups", {})

		# Initialize empty dictionaries
		color_dict: dict[str, str] = {}
		group_dict: dict[str, str] = {}

		# Process each group
		for group_name, group_info in group_data.items():
			types: list[str] = group_info.get("types", [])
			base_color: str = group_info.get("color", self._unknown_color)
			base_rgb: tuple = mcolors.to_rgb(base_color)

			# Create color variations for items in this group
			if len(types) == 1:
				# If only one item, use the base color
				color_dict[types[0]] = base_color
			else:
				# Create variations by adjusting brightness/saturation
				for i, type_name in enumerate(types):
					# Get variation factor (more items = more variation)
					factor: float = 0.2 * (i / (len(types) - 1 or 1) - 0.5)

					# Create variation (lighten/darken)
					new_rgb: tuple = tuple(
						min(max(c * (1 + factor), 0), 1) for c in base_rgb
					)
					color_dict[type_name] = mcolors.rgb2hex(new_rgb)

			# Assign group to each type
			for type_name in types:
				group_dict[type_name] = group_name

		# Add 'unknown' with a distinct color
		color_dict["unknown"] = self._unknown_color
		group_dict["unknown"] = "unknown"

		# Store the results
		self._head_type_colors = color_dict
		self._head_type_groups = group_dict

	def type_metadata(self) -> dict[str, dict]:
		"""Get metadata about each attention head type.

		# Returns:
		- `dict[str, dict]`
			Dictionary mapping type names to their metadata (url, notes, model, n_heads, heads)
		"""
		metadata: dict = dict()
		for paper in self.data_raw:
			for head_class in paper["classes"]:
				head_class_name: str = f"{paper['prefix']}:{head_class['type']}"
				metadata[head_class_name] = dict(
					url=paper["url"],
					notes=paper["notes"],
					model=paper["model"],
					n_heads=len(head_class["heads"]),
					heads=[f"{paper['model']}:{h}" for h in head_class["heads"]],
				)

		return metadata


def main() -> None:
	"""Main CLI interface for AttentionPedia."""
	import argparse

	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="AttentionPedia CLI"
	)
	parser.add_argument(
		"mode",
		choices=["head-to-types", "type-to-heads", "type-metadata", "all"],
		help="What to output",
	)
	parser.add_argument(
		"--out", type=Path, help="Output file path (default: print to console)"
	)

	args: argparse.Namespace = parser.parse_args()

	pedia: AttentionPedia = AttentionPedia()
	result: dict | list[dict]

	lines: bool = False
	match args.mode:
		case "all":
			result = {
				"head_to_types": pedia.head_to_types(),
				"type_to_heads": pedia.type_to_heads(),
				"type_metadata": pedia.type_metadata(),
			}
		case "dataframe":
			result = pedia.dataframe().to_dict(orient="records")
			lines = True
		case "df":
			result = pedia.dataframe().to_dict(orient="records")
			lines = True
		case "head-to-types":
			result = pedia.head_to_types()
		case "type-to-heads":
			result = pedia.type_to_heads()
		case "type-metadata":
			result = pedia.type_metadata()
		case _:
			raise ValueError(f"Unknown mode: {args.mode}")

	# Output result
	output: str
	if lines:
		output = "\n".join(json.dumps(item) for item in result)
	else:
		output = json.dumps(result, indent="\t")

	if args.out:
		args.out.write_text(output)
		print(f"# written to {args.out}", file=sys.stderr)
	else:
		print(output)


if __name__ == "__main__":
	main()
