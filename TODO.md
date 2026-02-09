# TODO

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
