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

When a class needs automatic serialization via `zanj`, use `SerializableDataclass` from `muutils`:
```python
from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)

@serializable_dataclass
class MyResult(SerializableDataclass):
	embeddings: Float[np.ndarray, "n_heads n_dims"]
	method: str
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

## Project Structure

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
| `make am-pipeline` | Run full pipeline (uses `$(PIPELINE_CFG_PATH)`, default: `pipeline_cfg.toml`) |
| `make am-pipeline-test` | Run pipeline with test config (`tests/pipeline_cfg_test.toml`) |
| `make am-frontend-bundle` | Format + build ap.json + bundle all frontend HTML |
| `make am-frontend-format` | Format frontend with prettier |
| `make am-rebuild-interfaces` | Bundle frontend + run s1c + s4b (copy to data/) |
| `make am-server-embed` | Serve `data/features/` on localhost (head embeddings) |
| `make am-server-patternlens` | Serve pattern lens on localhost |
| `make am-clean` | Delete ALL generated files in data/ (careful!) |

**Typical workflows:**

- Edit frontend source → `make am-frontend-bundle` → `make am-rebuild-interfaces`
- Run full analysis → `make am-pipeline PIPELINE_CFG_PATH=my_config.toml`
- View results locally → `make am-server-embed` or `make am-server-patternlens`
