# Appendix: Ablation Methodology

## Overview

To validate that attention heads identified as induction-like via embedding proximity also exhibit induction behavior under causal intervention, we conduct single-head ablation experiments following the methodology of Olsson et al. (2022). Each candidate head is individually ablated and its effect on model performance is measured across several metrics designed to capture different aspects of induction behavior. Below we describe the test data, ablation methods, evaluation metrics, and the experiment pipeline, noting where our implementation diverges from the original paper and why.

## Test Data Generation

Following Olsson et al. (2022, § Methods, "Head activation evaluators"), we construct test sequences by sampling 25 random tokens from the model's vocabulary and repeating the resulting pattern 4 times, prepended with a beginning-of-sequence (BOS) token:

$$\texttt{[BOS]}\;\underbrace{A_1\,A_2\,\ldots\,A_{25}}_{\text{rep 0}}\;\underbrace{A_1\,A_2\,\ldots\,A_{25}}_{\text{rep 1}}\;\underbrace{A_1\,A_2\,\ldots\,A_{25}}_{\text{rep 2}}\;\underbrace{A_1\,A_2\,\ldots\,A_{25}}_{\text{rep 3}}$$

Token sampling follows the paper's specification: we exclude special tokens (BOS, EOS, PAD, UNK), the 100 lowest-ID tokens (which in BPE tokenizers correspond to the most common tokens), and the 1000 highest-ID tokens (rare or special tokens). The paper phrases this as "excluding the most common and the least common tokens."

**Divergence: number of sequences.** The paper averages each metric over 10 examples. We use 50--100 sequences per evaluation to reduce variance. The metric definitions are unchanged; only the sample size differs.

## Ablation Methods

We implement three ablation methods. The first and third follow Olsson et al. (2022, § Methods, "Per-token losses (attention head ablations)"); the second is an additional baseline.

### Zero ablation ("full ablation")

The paper's "full ablation": replace the ablated head's result vector with a zero vector. All downstream computations are affected, including Q, K, and V calculations in later layers and all subsequent attention patterns.

In our implementation, a TransformerLens hook on `blocks.{L}.attn.hook_z` sets the target head's output slice to zero during the forward pass.

### Mean ablation

Replace the ablated head's output with its mean activation computed over calibration data. This is a less aggressive intervention than zeroing: the head's *average* contribution to the residual stream is preserved, and only its input-dependent variation is removed. This method is not described in the paper and is included as an additional baseline for comparison.

Calibration activations are computed over the same test sequences used for evaluation.

### Pattern-preserving ablation

The paper describes this as a two-step procedure:

> *Step 1:* Run the model for the first time, saving all attention patterns and discarding the output logits.
> *Step 2:* Run model a second time. Replace the ablated attention head's result vector with a zero vector, and force all attention patterns to be the version recorded in the first run.

The effect is that only downstream V-path computations are affected by the ablation; Q and K calculations, and therefore all attention patterns, remain as if the head were present. This isolates the head's contribution through the value pathway from its contribution to later layers' attention distributions.

Our implementation matches this procedure: `cache_clean_patterns()` performs the first forward pass and caches all attention patterns, then `ablate_heads()` registers both z-ablation hooks (zeroing the target head) and pattern-freeze hooks (substituting cached clean patterns) for all layers.

## Evaluation Metrics

### Loss on repeated sequences (primary metric)

The core test for induction behavior. We compute cross-entropy loss on the repeated token sequences, restricting the computation to *induction positions* — positions in the 2nd through 4th repetitions where the model could use the previous occurrence of the pattern to predict the next token. The primary metric is:

$$\texttt{loss\_increase} = \mathcal{L}_{\text{ablated}} - \mathcal{L}_{\text{baseline}}$$

A positive loss increase indicates the ablated head was contributing to the model's ability to predict tokens in the repeated pattern.

**Induction position mask.** Positions are included if they fall within repetitions 1--3 (0-indexed) and are not the first token of a repetition. For a sequence $[\text{BOS}][A\,B\,C\,D\,E][A\,B\,C\,D\,E][A\,B\,C\,D\,E][A\,B\,C\,D\,E]$, position 6 (the first A of rep 1) is excluded because the $E \to A$ transition has never appeared before (in rep 0, A was preceded by BOS). Positions 11 and 16 (the first A of reps 2 and 3) are also excluded, even though $E \to A$ has been seen by that point — we intentionally focus on within-pattern induction ($[A][B]\ldots[A] \to [B]$) and avoid the confounding cross-repetition boundary effect. This exclusion affects approximately 14% of valid induction positions and is applied consistently to both baseline and ablated measurements, so it cancels in the loss increase.

### Prefix matching score (offset +1)

For each token in the 2nd+ repetition, we measure the attention weight allocated to the token *after* the matching token in the previous repetition. On the pattern $[A\,B\,C][A\,B\,C]$, at position B in rep 1 (2nd repetition), we check the attention weight to C in rep 0 — the token the induction head should be copying.

The score is the mean of these attention weights across all valid induction positions and sequences.

**Divergence from Olsson et al. (2022).** The paper defines prefix matching as attention to "the tokens that preceded the same token in earlier repeats" — offset $-1$ from the earlier occurrence. We use offset $+1$ instead, for the following reason.

The induction mechanism operates via K-composition with a previous-token head. The previous-token head writes "A preceded me" into position B's residual stream. When the induction head at the second occurrence of A forms its query, the key at B (in the previous repetition) matches because it encodes "the token before me was A." The induction head therefore attends to B — offset $+1$ from A's first occurrence — and copies B through its OV circuit to predict it.

This is what TransformerLens's `get_induction_head_detection_pattern()` computes: it applies `torch.roll(duplicate_pattern, shifts=1, dims=1)`, producing the pattern where query position $q$ attends to key position $k$ such that $\text{tokens}[k+1] = \text{tokens}[q]$. The paper's own description is internally inconsistent: the informal definition (§ Methods) says induction heads "attend to previous instances of the present token and increase the logit of the token that followed" (offset 0 + copying), while the evaluator definition says offset $-1$. Neither matches the offset $+1$ of the actual K-composition mechanism.

We report both metrics: offset $+1$ as `prefix_score_decrease` (our primary prefix matching metric) and offset $-1$ as `prefix_score_decrease_legacy`.

### Preceding token score (offset −1)

This is the literal metric from Olsson et al. (2022): "the average of all attention pattern entries attending from a given token back to the tokens that preceded the same token in earlier repeats." It measures attention to offset $-1$ and is tracked alongside the offset $+1$ metric for completeness. Both are computed for every head in both baseline and ablated conditions.

### In-context learning (ICL) score

Following the paper: "the average loss at the 500th token in context minus the average loss at the 50th token." A negative score indicates the model's loss decreases over the course of a long sequence, implying it learns from context. Computed on long natural-text prompts (512+ tokens). When BOS is expected by the model, it is prepended to tensor inputs to match the convention used by string-based evaluation.

### Copying score (induction-specific)

Measures the logit increase for the correct next token at induction positions, isolating the head's contribution via the direct path $z \cdot W_O \cdot W_U$ (head output $\to$ residual stream $\to$ unembedding). At each induction position, we check how much the head's output raises the logit of the token that should be predicted next.

This is complementary to prefix matching: prefix matching measures *where* the head attends (QK circuit), while the copying score measures *what information it writes* (OV circuit).

### OV copying score (paper-style)

The paper's "Copying" evaluator (§ Methods, "Head activation evaluators"):

> Compute this head's contribution to the residual stream, then convert that using the unembeddings (i.e. along the "direct path") to impacts on each logit. Logits are transformed by subtracting the mean of the logits and passing through a ReLU, allowing the evaluator to focus on where logits are raised. Compute the ratio of the amount it raises the logits of the token being attended to, to that of all tokens in this sample. This value ranges from 0 (only raises other tokens) to 0.5 (only raises the present token), so we scale it into the range of −1 to 1.

Our implementation:

1. Compute the head's logit contribution: $\ell = z \cdot W_O \cdot W_U$
2. Subtract the per-position mean: $\ell' = \ell - \bar{\ell}$
3. Apply ReLU: $\ell^+ = \text{ReLU}(\ell')$
4. For each query position, compute the attention-weighted ratio:
$$r = \frac{\sum_j \alpha_{q,j} \cdot \ell^+_{q,\,\text{tokens}[j]}}{\sum_{t \in S} \ell^+_{q,t}}$$
where $\alpha_{q,j}$ is the attention weight from position $q$ to position $j$, $\text{tokens}[j]$ is the token at position $j$, and $S$ is the set of unique token types present in the sequence.
5. Scale to $[-1, 1]$: $\text{score} = 2r - 1$

The denominator sums over the unique token types present in each individual sequence (approximately 25 types plus BOS), matching the paper's "all tokens in this sample."

**Divergence from paper.** The paper specifies generating "a sequence of 25 random tokens" (singular, non-repeated) for the copying evaluator, while the prefix matching evaluator uses the repeated sequence. Our implementation uses the same repeated sequences for all metrics. Because induction heads attend strongly to offset-$+1$ positions on repeated sequences, this makes the OV copying score somewhat induction-specific rather than a pure OV-circuit measure. The separate induction-specific `copying_score` (§ above) provides an explicitly induction-contextualized alternative.

## Experiment Pipeline

For each candidate head:

1. **Baseline.** Run the unmodified model on the test sequences and compute all metrics.
2. **Ablation.** For each ablation method, ablate the head and recompute all metrics under the ablation.
3. **Record.** Store the difference (ablated $-$ baseline for loss; baseline $-$ ablated for scores) to quantify the head's causal contribution.

Ablation is performed one head at a time (single-head ablation), matching the paper's approach. All metrics share the same pre-generated test sequences for consistency within an experiment. For pattern-preserving ablation, clean attention patterns are cached once before the per-head loop and reused.

## Summary of Divergences from Olsson et al. (2022)

| Aspect | Paper | Our implementation | Rationale |
|---|---|---|---|
| Prefix matching offset | $-1$ (attend to token *before* match) | $+1$ (primary), $-1$ (secondary) | $+1$ matches the actual K-composition mechanism and TransformerLens's implementation; $-1$ tracked as a secondary metric |
| OV copying denominator | "all tokens in this sample" | Per-sequence unique token types | Our implementation matches the paper; this corrects an earlier bug that summed over the full vocabulary |
| OV copying input | Single non-repeated sequence | Repeated sequences | Reuses the shared test data; noted as slightly more induction-specific |
| Repetition boundary exclusion | Not discussed | First token of each repetition excluded from induction mask | Avoids confounding boundary effects; consistent between conditions |
| Number of samples | 10 | 50--100 | Reduces variance without changing metric definitions |
| Mean ablation | Not described | Included as additional method | Provides a less aggressive baseline for comparison |
