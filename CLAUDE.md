# Attention Motifs

## Coding Conventions

### Serialization Patterns

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
