# custom utils

import json
from pathlib import Path


# custom utils
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
	JSONitem,
)

from attention_motifs.consts import (
	PromptHashStr,
	b64encode,
	compute_text_hashes,
	str_batches,
)

PROMPT_SPECIAL_KEYS: set[str] = {"text", "hash_int", "hash_str"}


@serializable_dataclass
class Prompt(SerializableDataclass):
	"""A prompt is a dictionary with a text key and an optional hash key."""

	text: str
	hash_int: int
	hash_str: PromptHashStr
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


DEFAULT_CHAR_LEN_MIN: int|None = 64
DEFAULT_CHAR_LEN_MAX: int|None = 1024

@serializable_dataclass
class PromptDatasetConfig(SerializableDataclass):
	"""holds the config for a prompt dataset"""

	name: str
	source_path: Path = serializable_field(
		serialization_fn=lambda p: p.as_posix(),
		deserialize_fn=lambda data: Path(data),
	)
	source_info: dict[str, JSONitem]
	char_len_min: int|None = serializable_field(default=DEFAULT_CHAR_LEN_MIN)
	char_len_max: int|None = serializable_field(default=DEFAULT_CHAR_LEN_MAX)

	@classmethod
	def from_source_path(
		cls,
		source_path: Path,
		char_len_min: int|None = DEFAULT_CHAR_LEN_MIN,
		char_len_max: int|None = DEFAULT_CHAR_LEN_MAX,
	) -> "PromptDatasetConfig":
		source_path = Path(source_path)
		return cls(
			name=source_path.stem,
			source_path=source_path,
			source_info={
				"source_path": source_path.as_posix(),
			},
			char_len_min=char_len_min,
			char_len_max=char_len_max,
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
	hash_map: dict[PromptHashStr, int]

	@classmethod
	def from_config(cls, config: PromptDatasetConfig) -> "PromptDataset":
		"""create a dataset from a config by loading the prompts from the source path"""
		if not config.source_path.exists():
			raise FileNotFoundError(
				f"Prompt dataset source path does not exist: {config.source_path = }"
			)
		# load the prompts from the source path
		prompts: list[Prompt] = []
		with open(config.source_path, "r") as f:
			for line_idx, line in enumerate(f):
				# add fname metadata
				d_raw: dict = json.loads(line)
				d_raw["source_fname"] = config.source_path.as_posix()
				d_raw["source_line"] = line_idx

				# trim too-short samples
				if config.char_len_min is not None:
					if len(d_raw["text"]) < config.char_len_min:
						continue

				# split up too-long samples
				if config.char_len_max is not None:
					# grab the original text
					d_text: str = d_raw["text"]
					text_slices: list[str] = list(str_batches(
						text=d_text,
						batch_size=config.char_len_max,
						allow_last_incomplete=True,
					))
					# cut last if it's too short
					if len(text_slices[-1]) < config.char_len_min:
						text_slices.pop()

					# add em all
					for i, text_slice in enumerate(text_slices):
						prompts.append(Prompt.from_dict({**d_raw, "text": text_slice, "text_idx": i}))
				else:
					# add the prompt if no length constraints
					prompts.append(Prompt.from_dict(d_raw))

		return cls.from_prompts(config=config, prompts=prompts)

	@classmethod
	def from_prompts(
		cls, config: PromptDatasetConfig, prompts: list[Prompt]
	) -> "PromptDataset":
		"""create a dataset from a config and prompts by building the hash map"""
		hash_map: dict[PromptHashStr, int] = {
			p.hash_str: i for i, p in enumerate(prompts)
		}
		return cls(config=config, prompts=prompts, hash_map=hash_map)

	def summary(self) -> JSONitem:
		example_hash_str: PromptHashStr = self.prompts[0].hash_str
		return dict(
			config=self.config.serialize(),
			prompt_count=len(self),
			example={
				"prompts[0]": self.prompts[0].serialize(),
				f"hash_map[{example_hash_str}]": self.hash_map[example_hash_str],
			}
		)

	def __len__(self) -> int:
		return len(self.prompts)

	def index_get(self, idx: int) -> Prompt:
		"get a prompt by it's index in the prompts list"
		return self.prompts[idx]

	def hash_str_get(self, hash_str: PromptHashStr) -> Prompt:
		"get a prompt by what the text hashes to (base64 encoded string)"
		return self.prompts[self.hash_map[hash_str]]

	def hash_int_get(self, hash_int: int) -> Prompt:
		"get a prompt by what the text hashes to (raw integer)"
		# convert to string
		hash_str: str = b64encode(hash_int)
		return self.prompts[self.hash_map[hash_str]]

	def hash_get(self, hash: int | PromptHashStr) -> Prompt:
		"get a prompt by what the text hashes to"
		if isinstance(hash, int):
			return self.hash_int_get(hash)
		elif isinstance(hash, str):
			return self.hash_str_get(hash)
		else:
			raise TypeError(f"hash must be int or str, not {type(hash) = }, {hash = }")

	def __iter__(self):
		return iter(self.prompts)
