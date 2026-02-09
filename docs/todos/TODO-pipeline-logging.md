# TODO: Pipeline Logging Improvements

## Issues to Fix

### 1. Prompt path formatting shows `PosixPath()` wrapper
**Current output:**
```
loading prompts from prompts_path = PosixPath('data/text/pile_demo.jsonl')
```
**Desired output:**
```
loading prompts from data/text/pile_demo.jsonl
```

### 2. "prompts loaded" message not indented
**Current output:**
```
128 prompts loaded
```
**Desired output:**
```
  128 prompts loaded
```

### 3. Deprecation warning from transformer_lens
**Current output:**
```
`torch_dtype` is deprecated! Use `dtype` instead!
```
**Cause:** transformer_lens passes `torch_dtype` to HuggingFace's `AutoModelForCausalLM.from_pretrained()`, but HuggingFace has deprecated this parameter in favor of `dtype`.

**Location:** `transformer_lens/loading_from_pretrained.py` lines 2301, 2309, 2322, 2329, 2339, 2346

**Fix:** Suppress the warning until transformer_lens updates their code.

### 4. File paths not relative to cwd
**Current output:**
```
✔️  (0.00s) writing index.html
```
**Desired output:**
```
✔️  (0.00s) writing data/patterns/index.html
```

---

## Implementation

All changes are in the **pattern-lens** repository:
- **Repo:** `/home/miv/projects/attn/pattern-lens/`
- **File:** `pattern_lens/activations.py`

### Change 1: Fix prompt path formatting

**File:** `pattern_lens/activations.py`
**Line:** 461

```python
# BEFORE (line 460-462)
with SpinnerContext(
    message=f"loading prompts from {prompts_path = }",
    **SPINNER_KWARGS,
):

# AFTER
with SpinnerContext(
    message=f"loading prompts from {Path(prompts_path).as_posix()}",
    **SPINNER_KWARGS,
):
```

**Explanation:** The `{prompts_path = }` syntax is Python's self-documenting f-string which includes the variable name and repr(). Using `.as_posix()` gives a clean string path.

---

### Change 2: Indent "prompts loaded" message

**File:** `pattern_lens/activations.py`
**Line:** 478

```python
# BEFORE
print(f"{len(prompts)} prompts loaded")

# AFTER
print(f"  {len(prompts)} prompts loaded")
```

---

### Change 3: Suppress torch_dtype deprecation

**File:** `pattern_lens/activations.py`
**Location:** After imports (around line 47)

```python
# Add after existing imports
import warnings

# Suppress HuggingFace deprecation warning for torch_dtype
# (upstream issue in transformer_lens - they use torch_dtype instead of dtype)
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
```

---

### Change 4: Show relative path for index.html

**File:** `pattern_lens/activations.py`
**Line:** 481

```python
# BEFORE
with SpinnerContext(message="writing index.html", **SPINNER_KWARGS):
    if not no_index_html:
        write_html_index(save_path_p)

# AFTER
def _rel_path(p: Path) -> str:
    """Return path relative to cwd if possible, otherwise absolute."""
    try:
        return p.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return p.as_posix()

index_path = save_path_p / "index.html"
with SpinnerContext(message=f"writing {_rel_path(index_path)}", **SPINNER_KWARGS):
    if not no_index_html:
        write_html_index(save_path_p)
```

**Alternative (simpler, if paths are always under cwd):**
```python
with SpinnerContext(message=f"writing {(save_path_p / 'index.html').as_posix()}", **SPINNER_KWARGS):
    if not no_index_html:
        write_html_index(save_path_p)
```

---

## Other paths to check

Review these other logging statements in `pattern_lens/activations.py` for consistency:

| Line | Current Message | Should Update? |
|------|-----------------|----------------|
| 431 | `f"using device: {device_}"` | No (not a path) |
| 433 | `"loading model"` | No |
| 450 | `f"saving model info to {model_path.as_posix()}"` | Yes - make relative |
| 510 | `"updating jsonl metadata for models and prompts"` | No |

---

## After Implementation

1. **In pattern-lens repo:**
   ```bash
   cd /home/miv/projects/attn/pattern-lens
   git add -A && git commit -m "Improve pipeline logging output formatting"
   git push
   ```

2. **In attention-motifs repo:**
   ```bash
   cd /home/miv/projects/attn/attention-motifs
   uv sync  # Pull updated pattern-lens
   ```

3. **Verify:**
   ```bash
   python -m attention_motifs.pipeline.s1_activations
   ```

   Expected output:
   ```
   ✔️  (0.00s) loading prompts from data/text/pile_demo.jsonl
     128 prompts loaded
   ✔️  (0.00s) writing data/patterns/index.html
   ```
   (No `PosixPath()` wrapper, indented count, relative paths, no deprecation warning)
