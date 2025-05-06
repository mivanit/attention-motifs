import json
from typing import Callable, Literal
import warnings
from pathlib import Path
import importlib.resources
from collections import defaultdict

import numpy as np
import polars as pl

import attention_motifs

ATTNPEDIA_PATH: Path = (
	Path(importlib.resources.files(attention_motifs))
	/ "attnpedia"
	/ "attn-pedia.json"
)

if not ATTNPEDIA_PATH.is_file():
	warnings.warn(
		f"attpedia json does not exist: {ATTNPEDIA_PATH = }."
	)


AttentionPediaSchema = list[
	dict[
		# all values are strings, except for the "classes" key
		# which is a list of dictionaries
		Literal["prefix", "url", "notes", "model", "classes"],
		str | list[dict[
			Literal["type", "heads"],
			# "type" maps to a string, "heads" maps to a list of strings
			# where each string is of the form "L{layer}:H{head}"
			# e.g. "L0:H0", "L5:H7", etc.
			str | list[str],
		]],
	]
]


class AttentionPedia:
	def __init__(self, path: Path = ATTNPEDIA_PATH):
		self.data_raw: AttentionPediaSchema = json.loads(path.read_text())

	def dataframe(self) -> pl.DataFrame:
		df_raw: list[dict] = list()
		for group in self.data_raw:
			model_name: str = group["model"]

			for head_cls in group["classes"]:
				for h in head_cls["heads"]:
					h_split: tuple[str, str] = tuple(h.split(":"))
					layer_idx: int = int(h_split[0].removeprefix("L"))
					head_idx: int = int(h_split[1].removeprefix("H"))

					df_raw.append({
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
					})

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
				row["prefixed_type" if prefixed_type else "type"]
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