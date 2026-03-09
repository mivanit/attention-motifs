# Attention Pattern Features

## Overview

For each attention head, we extract a set of scalar features from its attention pattern matrix $A \in \mathbb{R}^{n \times n}$, where $n$ is the context length. In autoregressive (decoder-only) models, $A$ is lower-triangular with rows summing to 1: entry $A_{i,j}$ gives the attention weight from source position $i$ to target position $j$, with $A_{i,j} = 0$ for $j > i$.

Features are computed in three stages:

1. **Vector extraction**: pull out meaningful 1D slices of $A$
2. **Gram matrix construction**: build matrices capturing pairwise similarity structure
3. **Scalar summarization**: reduce vectors and matrices to scalar statistics

---

## Vector Extractions

Two vectors are extracted directly from $A$ and summarized with the full set of base scalar features (see below).

### Diagonal: self-attention

$$d = \operatorname{diag}(A), \quad d_i = A_{i,i}$$

The diagonal captures how much each position attends to itself. High diagonal values indicate identity-like or "copying" behavior; low values indicate the head distributes attention elsewhere.

### First-token column

$$f = A_{:,\,0}$$

The first column of $A$ gives each position's attention to token 0. For models that prepend a BOS token (GPT-2, Pythia, TinyStories, Gemma), this measures BOS attention. For models without BOS prepending (e.g. Llama), this is attention to the first content token.

---

## Base Scalar Features

Given a 1D vector $x \in \mathbb{R}^n$, we compute two groups of scalar statistics.

### Distribution statistics

| Feature | Definition | Notes |
|---------|-----------|-------|
| Mean | $\bar{x} = \frac{1}{n}\sum_i x_i$ | |
| Median | $\operatorname{median}(x)$ | |
| Variance | $s^2 = \frac{1}{n-1}\sum_i (x_i - \bar{x})^2$ | Unbiased (Bessel's correction) |
| Skewness | $\frac{1}{n}\sum_i \left(\frac{x_i - \bar{x}}{s}\right)^3$ | Pearson skewness |
| Kurtosis | $\frac{1}{n}\sum_i \left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$ | Excess (Fisher) kurtosis |
| Entropy | $H = -\sum_k p_k \log_2 p_k$ | Shannon entropy over a 10-bin histogram of $x$ |
| RMS | $\sqrt{\frac{1}{n}\sum_i x_i^2}$ | Root mean square |
| Energy | $\sum_i x_i^2$ | Total squared magnitude |
| L2 norm | $\frac{1}{n}\lVert x \rVert_2$ | Normalized Euclidean norm |

Kurtosis, energy, and L2 norm are included only for the vector extractions (diagonal and first-token), not for the gram histogram features.

### Time-series statistics

These treat the position index as a time axis, capturing how the vector evolves across positions.

| Feature | Definition | Notes |
|---------|-----------|-------|
| Lag-1 autocorrelation | $r_1 = \operatorname{corr}(x_{1:n-1},\; x_{2:n})$ | Pearson correlation between consecutive elements |
| PSD total power | $\sum_k S(f_k)$ | Total power from Welch's power spectral density estimate |
| Linear regression slope | $m$ from $x_i \approx m \cdot i + b$ | Rate of change across positions |
| Linear regression intercept | $b$ from $x_i \approx m \cdot i + b$ | Baseline level |
| Linear regression $R^2$ | $r^2$ | Goodness of linear fit |

---

## Gram Matrix Features

Six matrices are computed from $A$, each capturing a different aspect of the pattern's structure. These fall into two families.

### Raw gram matrices (dot product)

These are standard Gram matrices measuring pairwise inner products between rows or columns of $A$.

**Row gram** $G_\text{row} = A A^\top$

Entry $(G_\text{row})_{i,j} = \langle A_{i,:},\, A_{j,:} \rangle$: the dot product between the attention distributions of positions $i$ and $j$. High values indicate that two source positions distribute their attention similarly.

**Column gram** $G_\text{col} = A^\top A$

Entry $(G_\text{col})_{i,j} = \langle A_{:,i},\, A_{:,j} \rangle$: the dot product between the attention received by target positions $i$ and $j$. High values indicate that two target positions are attended to by a similar set of source positions.

**Skew gram** $G_\text{skew} = \tilde{A}^\top \tilde{A}$

where $\tilde{A} = \operatorname{skew}(A)$ (see below). This captures relative-position structure by aligning all rows so that self-attention occupies the same column, making positional patterns comparable across rows of different lengths.

### Log-space cosine similarity matrices

These apply cosine similarity to log-transformed attention weights, emphasizing the *shape* of the distribution rather than its magnitude.

First, compute $A_\text{log} = \log(A + \epsilon)$ (with $\epsilon = 10^{-9}$ and NaN/inf guarding). Then:

**Log row cosine** $C_\text{row}$

$$C_{i,j} = \frac{\langle (A_\text{log})_{i,:},\; (A_\text{log})_{j,:} \rangle}{\lVert (A_\text{log})_{i,:} \rVert \cdot \lVert (A_\text{log})_{j,:} \rVert}$$

Cosine similarity between log-space row vectors. The log transform compresses the dynamic range of attention weights, so this captures structural similarity even when magnitudes differ greatly (e.g., a sharp peak vs. a broad peak at the same relative positions).

**Log column cosine** $C_\text{col}$

Same as above but computed on columns of $A_\text{log}$.

**Log skew cosine** $C_\text{skew}$

Cosine similarity on the rows of $\operatorname{skew}(A_\text{log})$, combining log-space normalization with relative-position alignment.

---

## The Skew Transform

The skew transform $\operatorname{skew}: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n}$ right-aligns the rows of a lower-triangular matrix so that the diagonal ends up in the rightmost column:

$$\operatorname{skew}(A)_{i,\, j + (n - i - 1)} = A_{i,j} \quad \text{for } j \le i$$

For example, with $n=4$:

$$A = \begin{pmatrix} a & 0 & 0 & 0 \\ b & c & 0 & 0 \\ d & e & f & 0 \\ g & h & i & j \end{pmatrix} \;\;\longrightarrow\;\; \operatorname{skew}(A) = \begin{pmatrix} 0 & 0 & 0 & a \\ 0 & 0 & b & c \\ 0 & d & e & f \\ g & h & i & j \end{pmatrix}$$

This aligns rows by *relative* position: column $n-1$ always contains the self-attention weight, column $n-2$ the weight on the immediately preceding token, and so on. Without this transform, the diagonal entries of $A$ lie in different columns for each row, making row-wise comparisons conflate positional structure with sequence position.

---

## Gram-to-Scalar Pipeline

Each of the six gram/similarity matrices $G \in \mathbb{R}^{n \times n}$ is reduced to a set of scalars as follows:

1. **Flatten**: collect all $n^2$ entries of $G$ into a vector
2. **Histogram**: bin the values into 32 uniform bins over $[0, 1]$ (normalized density)
3. **Summarize**: apply the base scalar features (reduced set — without energy, kurtosis, or L2 norm) to the 32-dimensional histogram vector

The histogram step captures the overall *distribution* of pairwise similarities. For instance, a head where all rows attend identically will produce a histogram concentrated near 1.0 (high mean, low entropy), while a head with diverse attention patterns will spread mass across the range (lower mean, higher entropy).

---

## Feature Naming

Features follow the naming pattern `feat.<context>.<statistic>`, where `<context>` identifies the source matrix or vector, and `<statistic>` is one of the base scalar features.

| Context prefix | Source |
|---------------|--------|
| `diag` | Diagonal of $A$ |
| `first_tok` | First column of $A$ |
| `gram.row` | Row gram $AA^\top$ |
| `gram.col` | Column gram $A^\top A$ |
| `gram.skew` | Skew gram $\tilde{A}^\top \tilde{A}$ |
| `log.gram.row` | Log-space row cosine similarity |
| `log.gram.col` | Log-space column cosine similarity |
| `log.gram.skew` | Log-space skew cosine similarity |

For gram contexts, the statistic is additionally prefixed with `hist` (since it is computed on the histogram), giving names like `feat.gram.row.hist.entropy` or `feat.log.gram.skew.hist.linreg.slope`.
