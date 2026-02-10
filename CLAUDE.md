# Attention Motifs

**IMPORTANT NOTE:** do NOT make edits to anything in `data/`, as everything in that dir is ephemeral. make edits to the source of the frontend in `attention_motifs/`

# Coding Conventions

## Type Hinting

ALWAYS TYPE HINT EVERYTHING. Every function, every variable declaration. Use jaxtyping for hinting arrays whenever practical.
```python
# always type hint functions
def add(x: int, y: int) -> int:
	return x + y

# be as explicit as possible
def process_data(data: list[dict[str, int]]) -> dict[str, int]:
	# always type hint variables too
	intermediate_result: dict[str, int] = {}
	for item in data:
		# process item and update intermediate_result
		intermediate_result[item['key']] = item['value'] * 2
	return intermediate_result

def save_data(path: Path, data: dict[str, int]) -> None:
	# type hint even when it's obvious!
	new_path: Path = path.with_suffix('.json')

	...

import numpy as np
from jaxtyping import Float, Int


def process_array(
	arr: Float[np.ndarray, "batch features"],
) -> Float[np.ndarray, "batch features"]:
	intermediate: Float[np.ndarray, "batch features"] = arr * 2
	# put a space before the dim name if there is only one dim -- its a ruff thing
	batch_nonzero_features: Int[np.ndarray, " batch"] = (intermediate > 0).astype(int)

	...
```



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
