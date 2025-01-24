import hashlib
import base64
import json
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
	
	@classmethod
	def from_text(cls, text: str) -> "Prompt":
		hash_int, hash_str = compute_text_hashes(text)
		return cls(
			text=text,
			hash_int=hash_int,
			hash_str=hash_str,
			meta={},
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
	"""holds the config for a prompt dataset"""
	name: str
	source_path: Path = serializable_field(
		serialization_fn=lambda p: p.as_posix(),
		deserialize_fn=lambda data: Path(data),
	)
	source_info: dict[str, JSONitem]

	@classmethod
	def from_source_path(cls, source_path: Path) -> "PromptDatasetConfig":
		return cls(
			name=source_path.stem,
			source_path=source_path,
			source_info={
				"source_path": source_path.as_posix(),
			},
		)


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
	config: PromptDatasetConfig
	prompts: list[Prompt] = serializable_field(
		serialization_fn=lambda p_lst: [p.serialize() for p in p_lst],
		deserialize_fn=lambda data: [Prompt.load(p) for p in data],
	)
	hash_map: dict[str, int]

	@classmethod
	def from_config(cls, config: PromptDatasetConfig) -> "PromptDataset":
		"""create a dataset from a config by loading the prompts from the source path"""
		if not config.source_path.exists():
			raise FileNotFoundError(f"Prompt dataset source path does not exist: {config.source_path = }")
		# load the prompts from the source path
		prompts: list[Prompt] = []
		with open(config.source_path, "r") as f:
			for line in f:
				prompts.append(Prompt.from_dict(json.loads(line)))

		return cls.from_prompts(config=config, prompts=prompts)

	@classmethod
	def from_prompts(cls, config: PromptDatasetConfig, prompts: list[Prompt]) -> "PromptDataset":
		"""create a dataset from a config and prompts by building the hash map"""		
		hash_map: dict[str, int] = {p.hash_str: i for i, p in enumerate(prompts)}
		return cls(config=config, prompts=prompts, hash_map=hash_map)

	def __len__(self) -> int:
		return len(self.prompts)

	def index_get(self, idx: int) -> Prompt:
		"get a prompt by it's index in the prompts list"
		return self.prompts[idx]
	
	def hash_str_get(self, hash_str: str) -> Prompt:
		"get a prompt by what the text hashes to (base64 encoded string)"
		return self.prompts[self.hash_map[hash_str]]
	
	def hash_int_get(self, hash_int: int) -> Prompt:
		"get a prompt by what the text hashes to (raw integer)"
		# convert to string
		hash_str: str = b64encode(hash_int)
		return self.prompts[self.hash_map[hash_str]]
	
	def hash_get(self, hash: int | str) -> Prompt:
		"get a prompt by what the text hashes to"
		if isinstance(hash, int):
			return self.hash_int_get(hash)
		elif isinstance(hash, str):
			return self.hash_str_get(hash)
		else:
			raise TypeError(f"hash must be int or str, not {type(hash) = }, {hash = }")



@serializable_dataclass
class AttentionPatternDataset(SerializableDataclass):
	n_ctx: int
	n_patterns: int
	patterns: AttentionPatternBatch
	metadata: list[AttentionPatternMetadata]

	def __len__(self) -> int:
		return self.n_patterns
	

	def shuffle(self) -> None:
		"shuffle the dataset in-place"
		perm: Int[torch.Tensor, "n_patterns"] = torch.randperm(self.n_patterns)
		self.patterns = self.patterns[perm]
		self.metadata = [self.metadata[i] for i in perm]

	def __getitem__(
		self,
		idx: int|slice,
	) -> tuple[AttentionPattern, AttentionPatternMetadata]:		
		return self.patterns[idx], self.metadata[idx]
