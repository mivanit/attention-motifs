# Attention Motifs

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
