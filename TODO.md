# TODO

## attentionpedia optimizations

- use new array.js feature to load only part of an array

## Investigate NaN in feature computation pipeline

### Problem observed
When running step 2 (compute attention features) on larger models (pythia-410m), the pipeline crashes with:
```
ValueError: autodetected range of [nan, nan] is not finite
```

Also see RuntimeWarnings about "Precision loss occurred in moment calculation due to catastrophic cancellation" for skewness/kurtosis calculations.

### What's happening

1. **Log transform produces extreme values**: In `compute_scalar_features`, attention matrices with very small or zero values produce `-inf` after `np.log(A + 1e-9)`

2. **np.nan_to_num default behavior**: The original code `np.nan_to_num(..., nan=-10)` only specified replacement for NaN. The `-inf` values were replaced with the system's most negative representable float (~-1.8e308)

3. **Overflow in cosine similarity**: These extreme values cause overflow when computing `cosine_similarity_matrix(A_log)`:
   - Dot products become `inf`
   - `inf / inf = NaN`

4. **NaN propagation**: The NaN values propagate through:
   - `gram_features` → histogram with `density=True` produces NaN when input is NaN
   - `vec_features` → `np.histogram` fails when trying to auto-detect bin range from all-NaN array

### Current fix (bandaid)
- Added `neginf=-20, posinf=0` to clamp extreme values in log transform
- Added defensive NaN guards in `gram_features` and `vec_features`

### Needs investigation
- Why do some attention matrices have values that produce these edge cases?
- Are the affected attention heads meaningful or degenerate (e.g., untrained, broken)?
- Should we skip/flag these heads instead of silently replacing NaN with defaults?
- The skewness/kurtosis precision warnings suggest nearly-constant attention patterns - what do these represent?
- Consider whether the current NaN replacement values (0.5 for gram input, 0.0 for histogram) are appropriate or if they bias the feature statistics

## Fix OV copying score to use non-repeated sequences

### Problem

The `ov_copying_score` metric (in `attention_motifs/ablation/metrics.py`) is intended to implement the "Copying" evaluator from Olsson et al. (2022), which specifies generating "a sequence of 25 random tokens" (singular, non-repeated). The current implementation reuses the same repeated sequences (`[BOS][A₁...A₂₅][A₁...A₂₅]...`) used by all other metrics.

On repeated sequences, induction heads attend strongly to offset+1 positions, so the OV copying score conflates QK-circuit behavior (where the head attends) with OV-circuit behavior (what it writes). This makes the score measure "OV copying during induction" rather than "general OV-circuit copying tendency," which is what the paper intends.

### Impact

- Not comparable to Olsson et al. (2022) values
- Redundant with `copying_score`, which directly measures induction-specific logit increase more cleanly
- Currently not used in the paper for this reason

### Fix

Generate a separate batch of non-repeated random sequences (25 tokens each, same vocab exclusion rules) and pass them to `ov_copying_score`. This requires a new experiment run, which is computationally expensive.

## Minor fixes

### transition_tensor.py:146 — off-by-one in residuals loop
`range(n_idxs)` should be `range(1, n_idxs)`. When `i_idx == 0`, `tt_resampled[-1]` grabs the last element (highest power) via negative indexing instead of identity. Only affects the numpy plotting path, not the pipeline (which uses the torch version).

### s3b_feat_fig.py:93 — `handles` possibly unbound
`handles` is set inside a loop over embedding methods. If the list were empty, `handles` would be unbound at the `plt.legend()` call. Currently masked by `pyright: ignore[reportPossiblyUnboundVariable]`. Should initialize `handles = []` before the loop.

## Code quality

### head_analysis.py:206 — `_embed_meta` private attribute hack
`df._embed_meta = dict(...)` stashes metadata on a polars DataFrame via a private attribute. Works but fragile across polars versions. Could return a tuple or wrapper dataclass instead.

### cfg.py:432 — validate model names at config time
`validate_cfg()` checks models are strings but doesn't verify they exist in TransformerLens. Invalid names fail much later in the pipeline.
