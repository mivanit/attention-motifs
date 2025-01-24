import hashlib
import base64
from pathlib import Path
from typing import NamedTuple


import torch
from jaxtyping import Float, Int

# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
	JSONitem,
)

AttentionPattern = Float[torch.Tensor, "n_ctx n_ctx"]
AttentionPatternBatch = Float[torch.Tensor, "batch n_ctx n_ctx"]
TokenSequence = Int[torch.Tensor, "n_ctx"]
TokenSequenceBatch = Int[torch.Tensor, "batch n_ctx"]


def b64encode(data: bytes) -> str:
	return base64.b64encode(data, altchars=b"_-").decode("utf-8")


def compute_text_hashes(text: str) -> tuple[int, str]:
	hash_digest: bytes = hashlib.sha256(text.encode("utf-8")).digest()
	# get an integer hash
	hash_int: int = int.from_bytes(hash_digest, byteorder="big")
	# base64 encode it
	hash_str: str = b64encode(hash_digest)

	return hash_int, hash_str


@serializable_dataclass
class AttentionPatternMetadata(SerializableDataclass):
	model_name: str
	idx_layer: int
	idx_head: int
	prompt_hash: str
	n_ctx: int


	def tuple(self) -> tuple[str, int, int, str, int]:
		return (
			self.model_name,
			self.idx_layer,
			self.idx_head,
			self.prompt_hash,
			self.n_ctx,
		)
	
	def hash_int(self) -> int:
		return compute_text_hashes(self.tuple())[0]
	
	def hash_str(self) -> str:
		return compute_text_hashes(self.tuple())[1]
	
	def __hash__(self) -> int:
		return self.hash_int()




PROMPT_SPECIAL_KEYS: set[str] = {"text", "hash_int", "hash_str"}


@serializable_dataclass
class Prompt(SerializableDataclass):
	"""A prompt is a dictionary with a text key and an optional hash key."""

	text: str
	hash_int: int
	hash_str: str
	meta: dict[str, JSONitem]

	@classmethod
	def from_dict(cls, data: dict[str, JSONitem]) -> "Prompt":
		assert "text" in data

		# compute hashes if not present
		if ("hash_int" not in data) or ("hash_str" not in data):
			# if either is missing, recompute the hashes
			hash_int: int
			hash_str: str
			hash_int, hash_str = compute_text_hashes(data["text"])

			# and assert they match, if present
			assert data.get("hash_int", hash_int) == hash_int
			assert data.get("hash_str", hash_str) == hash_str

			# write to the data dict
			data["hash_int"] = hash_int
			data["hash_str"] = hash_str

		# return class
		return cls(
			text=data["text"],
			hash_int=data["hash_int"],
			hash_str=data["hash_str"],
			# anything else is in the metadata dict
			meta={k: v for k, v in data.items() if k not in PROMPT_SPECIAL_KEYS},
		)

	def __getitem__(self, key: str) -> JSONitem:
		match key:
			case "hash":
				return self.hash
			case "text":
				return self.text
			case _:
				return self.meta[key]

	def __hash__(self) -> int:
		return self.hash_int

@serializable_dataclass
class PromptDatasetConfig(SerializableDataclass):
	source_path: Path = serializable_field(
		serialization_fn=lambda p: p.as_posix(),
		deserialize_fn=lambda data: Path(data),
	)
	source_info: dict[str, JSONitem]
	chars_len_min: int|None = serializable_field(default=None)


@serializable_dataclass
class PromptDataset(SerializableDataclass):
	"""holds a dataset of prompts
	
	# Parameters:
	 - `prompts: list[Prompt]`
	 	a list of `Prompt` objects
	 - `hash_map: dict[str, int]`
	 	a mapping from prompt hash to index in the `prompts` list.
		we use a the hash as a b64 encoded string as the key instead of an int for two reasons:
		 - int -> b64 string is easier than the reverse
		 - json dict keys must be strings, not ints, and so we avoid an unnecessary conversion

	we avoid storing the prompts in a dict to preserve order, and also to eventually dump them into a jsonl file
	"""	
	prompts: list[Prompt] = serializable_field(
		serialization_fn=lambda p_lst: [p.serialize() for p in p_lst],
		deserialize_fn=lambda data: [Prompt.load(p) for p in data],
	)
	hash_map: dict[str, int]

	def __len__(self) -> int:
		return len(self.prompts)

	def index_get(self, idx: int) -> Prompt:
		return self.prompts[idx]
	
	def hash_str_get(self, hash_str: str) -> Prompt:
		return self.prompts[self.hash_map[hash_str]]
	
	def hash_int_get(self, hash_int: int) -> Prompt:
		# convert to string
		hash_str: str = b64encode(hash_int)
		return self.prompts[self.hash_map[hash_str]]
	
	def hash_get(self, hash: int | str) -> Prompt:
		if isinstance(hash, int):
			return self.hash_int_get(hash)
		elif isinstance(hash, str):
			return self.hash_str_get(hash)
		else:
			raise TypeError(f"hash must be int or str, not {type(hash) = }, {hash = }")
		

	

	def from_raw_prompts




@serializable_dataclass
class AttentionPatternDataset(SerializableDataclass):
	n_ctx: int
	n_patterns: int
	patterns: AttentionPatternBatch
	metadata: list[AttentionPatternMetadata]

	def __len__(self) -> int:
		return self.n_patterns

	def __getitem__(
		self, idx: int
	) -> tuple[Float[torch.Tensor, "n_ctx n_ctx"], AttentionPatternMetadata]:
		return self.patterns[idx], self.metadata[idx]