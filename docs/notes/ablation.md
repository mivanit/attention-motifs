# Ablation Study Notes

## Overview

**Goal**: Test whether heads that are nearby known GPT-2 induction heads in embedding space also behave like induction heads when ablated.

**Core metric**: Loss increase on repeated sequences `[A B C][A B C]...`

Ablating an induction head should increase loss on these sequences because the model can no longer use the induction mechanism to predict repeated tokens.

## Methodology

### Ablation Types
- **ZERO**: Set head output to 0
- **MEAN**: Set head output to its mean activation (computed on calibration data)

### Test Setup
1. Generate repeated token sequences: `[random tokens] × N repetitions`
2. Compute baseline loss (no ablation)
3. Ablate each head and measure loss
4. `loss_increase = ablated_loss - baseline_loss`

Positive loss increase = head was helping with the task.

### Parameters (notebook 06)
- `n_sequences = 50`
- `seq_length = 25` (base pattern length)
- `n_repetitions = 4`
- Baseline loss on gpt2-small: ~0.25

## Sanity Check Results (gpt2-small)

### Known Induction Heads (AttentionPedia)
```
L5:H1, L5:H5, L5:H8, L5:H9, L6:H9, L7:H2, L7:H10
```

### ZERO Ablation Results

| Head | Loss Increase | Notes |
|------|--------------|-------|
| L7:H2 | 4.44 | High impact |
| L7:H10 | 3.02 | High impact |
| L6:H9 | 2.69 | High impact |
| L5:H9 | 1.53 | Medium |
| L5:H8 | 0.49 | Low |
| L5:H1 | 0.36 | Low |
| L5:H5 | 0.25 | Low |

### Control Heads (arbitrary non-induction)

| Head | Loss Increase |
|------|--------------|
| L0:H0 | 2.49 |
| L1:H1 | 2.57 |
| L2:H2 | 2.62 |
| L11:H11 | 2.65 |

## Key Observations

1. **Not all "known" induction heads show high loss increase**
   - L5:H1, L5:H5, L5:H8 show 0.25-0.49 increase
   - These are LOWER than random early-layer heads (~2.5)

2. **L6:H9, L7:H2, L7:H10 are the heavy-lifters**
   - Show 2.69-4.44 loss increase
   - Clearly more important for the induction task

3. **Early-layer heads cause high loss when ablated**
   - L0-L2 heads show ~2.5 loss increase
   - Not because they're induction heads
   - Because they're foundational to all downstream processing

4. **Control group is flawed**
   - Early-layer heads are a bad control
   - Need controls from similar layers (L5-L7) that aren't induction heads

## Interpretation

The L5 induction heads may be:
- Redundant with L6/L7 heads
- Less critical for the specific task
- Doing partial/weaker induction

The methodology needs refinement:
- Test ALL heads to see the full distribution
- Use layer-matched controls
- Look for unknown high-impact heads

## Notebook Updates

Updated `notebooks/06-ablation_study.ipynb`:
- Removed `prefix_matching_score` (not needed for now, had bugs)
- Added full-head scan (all 144 heads in gpt2-small)
- Added histogram showing distribution of loss increase
- Added sorted table of all heads by loss increase
- Added summary statistics comparing known vs unknown heads

## Next Steps

1. Run the full-head scan to see the complete distribution
2. Check if known induction heads are actually outliers
3. Identify any "unknown" high-impact heads
4. Use better controls (same-layer non-induction heads)
