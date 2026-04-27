# Attention Motifs

**IMPORTANT NOTE:** do NOT make edits to anything in `data/`, as everything in that dir is ephemeral. make edits to the source of the frontend in `attention_motifs/`

# Coding Conventions

## Type Hinting

ALWAYS TYPE HINT EVERYTHING. Every function, every variable declaration. Use jaxtyping for hinting arrays whenever practical.

**Variable declarations must always have type annotations**, even when the type is obvious:
```python
# WRONG -- never do this
path = Path("output")
data = json.load(f)
result = {}
n = len(items)

# RIGHT -- always annotate
path: Path = Path("output")
data: dict = json.load(f)
result: dict[str, int] = {}
n: int = len(items)
```

**Function signatures** -- always annotate all parameters and return type:
```python
def add(x: int, y: int) -> int:
	return x + y

def process_data(data: list[dict[str, int]]) -> dict[str, int]:
	intermediate_result: dict[str, int] = {}
	for item in data:
		intermediate_result[item['key']] = item['value'] * 2
	return intermediate_result
```

**Modern union syntax** -- use `X | Y` and `X | None`, not `Union` or `Optional`:
```python
# WRONG
from typing import Union, Optional
def foo(x: Optional[str] = None) -> Union[int, str]: ...

# RIGHT
def foo(x: str | None = None) -> int | str: ...
```

## Jaxtyping for Arrays

Use `jaxtyping` (`Float`, `Int`, etc.) to annotate array shapes for both numpy and torch:
```python
import numpy as np
import torch
from jaxtyping import Float, Int

# torch tensors
pattern: Float[torch.Tensor, "n_ctx n_ctx"] = model.get_pattern()
tokens: Int[torch.Tensor, "batch n_ctx"] = tokenizer(text)
head_output: Float[torch.Tensor, "batch pos d_head"] = z_output

# numpy arrays
embeddings: Float[np.ndarray, "n_heads n_features"] = pca.transform(raw)
labels: Int[np.ndarray, " n_heads"] = cluster_result.labels

# IMPORTANT: put a space before the dim name if there is only one dim (ruff F722)
counts: Int[np.ndarray, " batch"] = (arr > 0).sum(axis=1)
#                        ^ space here
```

Define **type aliases** for commonly used shapes:
```python
AttentionPattern = Float[torch.Tensor, "n_ctx n_ctx"]
AttentionPatternBatch = Float[torch.Tensor, "batch n_ctx n_ctx"]
TokenSequence = Int[torch.Tensor, "n_ctx"]
```

## Makefile Edits

The Makefile is from a template. When modifying template recipes (anything above the custom `am-*` section at the bottom), wrap changed lines with bare `~~` divider comments -- no text on the divider lines:

```makefile
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PYTEST_OPTIONS ?= --durations=20
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Edits within the custom `am-*` section at the bottom do not need dividers.

## Serialization Patterns

For dataclasses and similar containers, use these method naming conventions:

- **File I/O**: `.save(path)` / `.read(path)` - Write to / read from file
- **Data conversion**: `.serialize()` / `.load(data)` - Convert to / from JSON-compatible dict

Example:
```python
# File operations
candidates = CandidateHeads.read(Path("candidates.json"))
candidates.save(Path("candidates.json"))

# Dict operations (for embedding in other structures)
data = candidates.serialize()  # -> dict
candidates = CandidateHeads.load(data)  # dict -> instance
```

When a class needs automatic serialization (especially with numpy arrays) via `zanj`, use `@serializable_dataclass` + `SerializableDataclass` from `muutils`:

```python
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)

@serializable_dataclass
class MyResult(SerializableDataclass):
	cls_values: list[str]
	distances: Float[np.ndarray, "h h"]
	is_reduced: bool = serializable_field(default=False)
```

**How it works:**

- `@serializable_dataclass` replaces `@dataclass` (calls it internally -- don't use both)
- Auto-generates `.serialize()`, `.load()`
- Auto-registers with ZANJ (`register_handler=True` by default) so ZANJ can reconstruct the class

**`serializable_field()`:**

- Use `serializable_field()` when you need:
  - Defaults on a serializable field: `serializable_field(default=False)`
  - Custom serialization: `serializable_field(serialization_fn=lambda x: x.tolist())`
  - Custom deserialization: `serializable_field(deserialize_fn=lambda x: np.array(x))`
  - `loading_fn` is a legacy alternative to `deserialize_fn` that receives the **entire dict** instead of just the field value -- prefer `deserialize_fn`

**Decorator params:**

- `methods_no_override=["serialize", "load"]` -- skip auto-generating these methods so you can define custom ones
- `properties_to_serialize=["my_prop"]` -- include `@property` values in serialized output
- `frozen=True`, `kw_only=True` -- passed through to `@dataclass`

**ZANJ save/read pattern:**

```python
from zanj import ZANJ

@serializable_dataclass()
class MyClass(SerializableDataclass):
	# anything which we can normally do `json.dump()` on works out of the box
	count: int
	name: str
	data: dict[str, int]
	vocab: list[str]
	# as do numpy/torch arrays, dataframes, and nested dataclasses:
	array: Float[np.ndarray, "n d"]
	dataframe: pd.DataFrame
	other_serializable_dataclass: OtherClass # automatically converted, if `OtherClass` is also a `SerializableDataclass`

	# custom classes require custom serialization functions:
	device: torch.device = serializable_field(
		serialization_fn=lambda x: str(x),
		deserialize_fn=lambda x: torch.device(x),
	)
	optimizer: type[torch.optim.Optimizer] = serializable_field(
		serialization_fn=lambda x: x.__name__,
		deserialize_fn=lambda x: getattr(torch.optim, x),
	)
	some_kind_of_object: Any = serializable_field(
		serialization_fn=lambda x: custom_serialize(x),
		deserialize_fn=lambda x: custom_deserialize(x),
	)

	# .load() and .serialize() are auto-generated

	def save(self, path: Path | str, zanj: ZANJ | None = None) -> None:
		if zanj is None:
			zanj = ZANJ()
		zanj.save(self.serialize(), path)

	@classmethod
	def read(cls, path: Path | str, zanj: ZANJ | None = None) -> "MyClass":
		if zanj is None:
			zanj = ZANJ()
		return zanj.read(path)
```

## Dataclass Conventions

- **`kw_only=True`** for config-style dataclasses with many fields:
  ```python
  @dataclass(kw_only=True)
  class PipelineConfig:
      prompts_file: Path
      n_samples: int
      device: str = "cpu"
  ```

## Literal Types

Use `Literal[...]` type aliases for constrained string values (not string Enums):
```python
EmbeddingMethod = Literal["isomap", "umap", "tsne", "pca"]
DataFilename = Literal["raw", "norms", "scaled", "pca", "pca_npy"]

# Then use in signatures and dataclass fields:
embedding_methods: list[EmbeddingMethod] = field(...)
```

# Project Structure

| Directory | Purpose | Editable? |
|-----------|---------|-----------|
| `attention_motifs/` | Source code, pipeline, frontend source | ✅ Yes |
| `data/` | Generated output (patterns, features, figures) | ❌ No |
| `notebooks/` | Jupyter notebooks for running pipeline | ✅ Yes |
| `tests/` | Test suite | ✅ Yes |

## Data Generation

**How `data/` is generated:**

- `make am-pipeline` → runs steps s0-s5b
- Config: `pipeline_cfg.toml`

**Pipeline steps:**

- s1: Extract attention patterns → `data/patterns/`
- s2-s3: Compute features → `data/features/`
- s4b: Copy frontend → `data/vis/`
- s5: Generate embeddings → `data/figures/`

## Frontend Workflow

```
Source: attention_motifs/frontend/**/src/**  ← EDIT HERE
   ↓ make am-frontend-bundle
Bundle: attention_motifs/frontend/**/index.html  ← DON'T EDIT (generated)
   ↓ pipeline step s4b
Output: data/vis/*/index.html  ← DON'T EDIT (generated)
```

**Frontend editability:**

| Path | Editable? |
|------|-----------|
| `attention_motifs/frontend/**/src/**` | ✅ Yes (source) |
| `attention_motifs/frontend/**/index.html` | ❌ No (bundled output) |
| `attention_motifs/frontend/**/build/` | ❌ No (build artifacts) |

## Common Commands

**General:**

| Command | Description |
|---------|-------------|
| `make test` | Run test suite |
| `make format` | Format code (ruff/prettier) |

**Attention-Motifs (`am-*`):**

| Command | Description |
|---------|-------------|
| `make am-setup` | Run all setup steps (download models, data, install playwright) |
| `make am-pipeline` | Run full pipeline (uses `$(PIPELINE_CFG_PATH)`, default: `pipeline_cfg.toml`) |
| `make am-pipeline-test` | Run pipeline with test config (`tests/pipeline_cfg_test.toml`) |
| `make am-frontend-bundle` | Format + build ap.json + bundle all frontend HTML |
| `make am-frontend-deploy` | Bundle frontend + run s1c + s4b (copy to data/) |
| `make am-deploy` | Build and deploy frontend to data/ |
| `make am-serve` | Serve `data/` on localhost |
| `make am-serve-patterns` | Serve pattern lens on localhost |
| `make am-clean` | Delete ALL generated files in data/ (careful!) |

**Standalone scripts:**

| Command | Description |
|---------|-------------|
| `uv run python attention_motifs/pipeline/s5_head_embed.py --add-metadata [path]` | Fast-add `model_family`/`model_size`/`layer_depth` columns to existing `head_embed.jsonl` without re-running full s5 |
| `uv run python attention_motifs/pipeline/s3_feat_proc.py --add-metadata [dir]` | Fast-add `activation.model_family`/`activation.model_size` columns to existing pattern embedding files (raw, scaled, pca) in `dir` (default: `data/features/`) |

**Typical workflows:**

- Edit frontend source → `make am-frontend-bundle` → `make am-frontend-deploy`
- Run full analysis → `make am-pipeline PIPELINE_CFG_PATH=my_config.toml`
- View results locally → `make am-serve` or `make am-serve-patterns`
