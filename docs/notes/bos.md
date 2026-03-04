# BOS Token Handling

Reference for how beginning-of-sequence (BOS) tokens flow through the pipeline, from TransformerLens internals through pattern extraction, ablation experiments, feature computation, and the frontend viewer.

## 1. TransformerLens BOS Behavior

This is the root of all BOS complexity. TransformerLens has two independent BOS mechanisms that interact:

**`default_prepend_bos`** (per-model config): whether the model *should* have a BOS token. Most models default to `True` (GPT-2, Pythia, Gemma, Llama, Mistral, TinyStories). Only Qwen and Falcon explicitly set `False`.

**`tokenizer_prepends_bos`** (runtime-detected): whether the HuggingFace tokenizer *actually adds* a BOS token when encoding. Detected empirically via `len(tokenizer.encode("")) > 0` after the tokenizer is initialized with `add_bos_token=True`.

These two flags interact in `to_tokens()` to ensure exactly 0 or 1 BOS:

| `default_prepend_bos` | `tokenizer_prepends_bos` | Action |
|---|---|---|
| `True` | `True` | Tokenizer already adds BOS → do nothing |
| `True` | `False` | Manually prepend BOS string before tokenizing |
| `False` | `True` | Strip the auto-added BOS after tokenizing |
| `False` | `False` | Do nothing |

### The critical rule: `prepend_bos` is ignored for tensor inputs

`HookedTransformer.input_to_embed` (line 378-386): when the input is already a tensor, the `prepend_bos` argument has **no effect on tokenization**. The tensor is used as-is. The `prepend_bos` flag is only consulted later to build the attention mask (for left-padding / KV-cache scenarios).

This means: **when passing pre-tokenized tensors to the model, you are fully responsible for ensuring the BOS token is present (or absent) as appropriate.**

### `get_tokenizer_with_bos()`

TransformerLens reinitializes the tokenizer with `add_bos_token=True` via `get_tokenizer_with_bos()`. However, some tokenizers ignore this flag entirely (e.g., GPT-2's BPE tokenizer). This is why `tokenizer_prepends_bos` is detected empirically rather than trusted from config.

Consequence: `tokenizer.encode(text)` may or may not include BOS, even after `get_tokenizer_with_bos()`. Do not assume either way — always check or prepend explicitly.


## 2. Pattern Extraction (`pattern_lens/activations.py`)

Two different tokenization methods are used for different purposes:

- **`model.to_tokens(text)`** → sequence length **includes** BOS (when `default_prepend_bos=True`). The attention pattern tensor shape is `(seq_len, seq_len)` where `seq_len` includes BOS.
- **`tokenizer.tokenize(text)`** → token list **excludes** BOS. Stored in `prompt.json` metadata as the human-readable token strings.

These differ by 1 for BOS models. The code documents this explicitly (lines 407-410) and uses the correct method for each purpose: `to_tokens()` for determining attention pattern dimensions, `tokenize()` for metadata.

The batched forward pass (`model.run_with_cache(texts, ...)`) passes raw strings, so BOS is auto-prepended by `to_tokens()` inside TransformerLens per the model's `default_prepend_bos`.


## 3. Ablation Data Generation (`ablation/data.py`)

### `generate_repeated_sequences`

Explicit `prepend_bos=True` default. Constructs sequences as `[BOS][A1...An][A1...An]...` by manually concatenating a BOS tensor at position 0. The `RepeatedSequence` dataclass tracks:

- `has_bos: bool` — whether `tokens[0]` is BOS
- `repetition_starts: list[int]` — shifted by +1 when BOS is present

BOS ID is resolved via `bos_token_id` parameter, falling back to `tokenizer.bos_token_id`, then to `0`.

### `generate_long_context_prompts`

Explicit `prepend_bos=True` default (added in commit `4e37f64`). Tokenizes with `tokenizer.encode()` then manually prepends BOS before truncation to `target_length`. The BOS is included in the target length budget.

### Pitfall: `tokenizer.encode()` behavior varies

GPT-2's tokenizer typically does NOT add BOS even with `add_bos_token=True` (it ignores this flag). Llama's tokenizer DOES add BOS. Both data generation functions explicitly prepend BOS rather than relying on the tokenizer, ensuring consistent behavior across models.

**Known caveat**: if the tokenizer *does* add BOS (e.g. Llama), `generate_long_context_prompts` will produce a double-BOS. The `icl_score` function that consumes these tensors has a guard against double-BOS (checking `tokens[0, 0] != bos_id`), but `generate_long_context_prompts` itself does not. This is acceptable for the current models (GPT-2 family) but would need a guard for Llama.


## 4. Ablation Metrics (`ablation/metrics.py`)

### The `prepend_bos=not has_bos` invariant

All `RepeatedSequence`-based metric functions follow this pattern:

```python
has_bos: bool = sequences[0].has_bos if sequences else False
logits = model(tokens, prepend_bos=not has_bos)
```

- If sequences already have BOS → tell model NOT to prepend → dimensions align directly
- If sequences lack BOS → tell model TO prepend → model adds BOS at position 0

Note: for tensor inputs, `prepend_bos` doesn't actually affect tokenization (§1), but it does affect the attention mask. The convention is maintained for correctness and clarity.

### Position shift

When the model adds BOS to sequences that lack it, all attention pattern positions shift by 1 relative to the original token indices:

```python
pos_shift: int = 0 if has_bos else 1
```

This offset is applied when indexing into attention patterns in `_prefix_score_impl` and `ov_copying_score`.

### Loss alignment

When BOS is model-inserted (not in input), logits need different slicing:
- `has_bos=True`: `logits[:, :-1]` predicts `tokens[:, 1:]` — standard next-token prediction
- `has_bos=False`: `logits[:, 1:-1]` predicts `tokens[:, 1:]` — skip the BOS logit position

### `icl_score` tensor path

String inputs go through `model.to_tokens()` which handles BOS automatically. Tensor inputs need manual handling:

```python
if model.cfg.default_prepend_bos:
    bos_id = getattr(model.tokenizer, "bos_token_id", None) or 0
    if tokens.shape[1] == 0 or tokens[0, 0].item() != bos_id:
        # prepend BOS
```

The `tokens[0, 0] != bos_id` check prevents double-BOS when the tensor already starts with BOS.


## 5. Pattern-Preserving Ablation (`experiment.py`)

```python
sequences = generate_repeated_sequences(..., prepend_bos=True)   # has BOS
tokens_batch = sequences_to_batch(sequences)                      # tensor with BOS at [0]
ablator.set_clean_patterns(tokens_batch, prepend_bos=False)       # don't add another
```

This is belt-and-suspenders: `prepend_bos=False` is passed explicitly, AND `prepend_bos` is ignored for tensor inputs anyway. The intent is clear: the tokens already have BOS, don't touch them.


## 6. Feature Computation (`features/features.py`)

The `first_tok` feature group (`A[:, 0]`) captures how much each query position attends to position 0 of the sequence.

- For models with `default_prepend_bos=True` (GPT-2, Pythia, TinyStories, Gemma): position 0 is the BOS token. High `first_tok` values indicate BOS-attending behavior.
- For models without BOS prepending (Qwen, Falcon): position 0 is the first content token. The feature measures something different.

Documented in code comment at line 279-280.


## 7. Frontend / Viewer (`pattern_lens`)

The pattern lens single-head viewer uses `CONFIG.data.tokenBoundary.start = ["<BOS>"]` to prepend a BOS label to the token display, aligning the label count with the attention pattern dimensions (which include BOS for BOS models).

This is currently **not model-aware** — the `<BOS>` label is always shown regardless of `default_prepend_bos`. For Qwen/Falcon, this would cause a spurious BOS label and off-by-one alignment between tokens and attention cells.

This is tracked as an upstream fix in `pattern_lens`. The `s1c_write_idxs.py` pipeline step writes viewer config but does not include BOS-related keys; the viewer falls back to its hardcoded default.


## 8. Pitfalls & Rules of Thumb

| Pitfall | Where it bites | How the code handles it |
|---|---|---|
| Assuming `prepend_bos` works for tensors | Anywhere pre-tokenized tensors are passed to `model()` | Explicit BOS prepend in data generation; `icl_score` has a guard |
| `tokenizer.encode()` BOS behavior varies by tokenizer class | `generate_long_context_prompts`, any manual tokenization | Always prepend BOS explicitly rather than relying on tokenizer |
| Attention pattern dimensions include BOS but metadata tokens don't | Pattern extraction, frontend display | `to_tokens()` for dimensions, `tokenize()` for metadata; differ by 1 |
| Position indexing off-by-one when BOS is model-inserted vs in-input | All attention pattern analysis (`prefix_matching_score`, `ov_copying_score`) | `pos_shift = 0 if has_bos else 1` |
| Loss alignment differs when BOS is model-inserted | `induction_loss`, all loss-based metrics | Different slicing paths for `has_bos=True` vs `False` |
| Double-BOS from tokenizer + manual prepend | `generate_long_context_prompts` + Llama tokenizer | `icl_score` guards with `tokens[0,0] != bos_id`; data gen does not (acceptable for current GPT-2-family models) |
| `model_cfg.json` stores BOS config but frontend doesn't use it | Pattern lens viewer | Upstream fix pending |

### The one rule

**When passing tensors to TransformerLens, you are responsible for BOS.** The model will not add or remove it. Check `model.cfg.default_prepend_bos` and act accordingly.
