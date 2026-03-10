# Attention Pattern Features

## Overview

For each attention head, we extract a set of scalar features from its attention pattern matrix $A \in \mathbb{R}^{n \times n}$, where $n$ is the context length. In autoregressive (decoder-only) models, $A$ is lower-triangular with rows summing to 1: entry $A_{i,j}$ gives the attention weight from source position $i$ to target position $j$, with $A_{i,j} = 0$ for $j > i$.

Features are computed in three stages:

1. **Vector extraction**: pull out meaningful 1D slices of $A$
2. **Gram matrix construction**: build matrices capturing pairwise similarity structure
3. **Scalar summarization**: reduce vectors and matrices to scalar statistics

---

## Vector Extractions

Eight vectors are extracted from $A$ and summarized with the full set of base scalar features (see below).

### Diagonal: self-attention

$$d = \operatorname{diag}(A), \quad d_i = A_{i,i}$$

The diagonal captures how much each position attends to itself. High diagonal values indicate identity-like or "copying" behavior; low values indicate the head distributes attention elsewhere.

### First-token column

$$f = A_{:,\,0}$$

The first column of $A$ gives each position's attention to token 0. For models that prepend a BOS token (GPT-2, Pythia, TinyStories, Gemma), this measures BOS attention. For models without BOS prepending (e.g. Llama), this is attention to the first content token.

### Last-token column

$$\ell = A_{:,\,n-1}$$

The last column of $A$ gives each position's attention to the final token in the context.

### Previous-token (subdiagonal)

$$p_i = A_{i,\,i-1} \quad \text{for } i \ge 1$$

The first subdiagonal of $A$, giving each position's attention to the immediately preceding token. This is the signature of "previous-token" or bigram heads.

### Row entropy

$$H_i = -\sum_j A_{i,j} \ln A_{i,j}$$

The Shannon entropy (natural log) of each row's attention distribution. Low entropy indicates the head concentrates attention on a few positions; high entropy indicates a diffuse, uniform-like distribution.

### Attention distance

$$\delta_i = \sum_j A_{i,j} \, |i - j|$$

The expected absolute distance between source and target positions, weighted by attention. Small values indicate local attention (attending to nearby tokens); large values indicate long-range attention.

### Row max

$$m_i = \max_j A_{i,j}$$

The maximum attention weight in each row, measuring peakedness. A row max near 1 means the head attends almost entirely to a single position; values near $1/i$ indicate near-uniform attention.

### Column sum

$$c_j = \sum_i A_{i,j}$$

The total attention received by each target position, summed over all source positions. Positions with high column sums are broadly attended to across the sequence (e.g., BOS tokens, separator tokens).

---

## Standalone Scalar Features

### Band energy

$$E_\text{band} = \frac{\sum_{i,j:\,|i-j| \le k} A_{i,j}}{\sum_{i,j} A_{i,j}}, \quad k = \max\!\left(1,\, \lfloor n/4 \rfloor\right)$$

The fraction of total attention mass concentrated within $k$ diagonals of the main diagonal. Values near 1 indicate the head attends primarily to nearby positions; lower values indicate significant long-range attention.

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

Kurtosis, energy, and L2 norm are included only for the vector extractions, not for the gram histogram features.

### Time-series statistics

These treat the position index as a time axis, capturing how the vector evolves across positions.

| Feature | Definition | Notes |
|---------|-----------|-------|
| Lag-1 autocorrelation | $r_1 = \operatorname{corr}(x_{1:n-1},\; x_{2:n})$ | Pearson correlation between consecutive elements |
| PSD total power | $\sum_k P(f_k)$ | Total power from Welch's power spectral density estimate |
| Linear regression slope | $m$ from $x_i \approx m \cdot i + b$ | Rate of change across positions |
| Linear regression intercept | $b$ from $x_i \approx m \cdot i + b$ | Baseline level |
| Linear regression $R^2$ | $r^2$ | Goodness of linear fit |

---

## Gram Matrix Features

Eight matrices are computed from $A$, each capturing a different aspect of the pattern's structure. These fall into two families.

### Raw gram matrices (dot product)

These are standard Gram matrices measuring pairwise inner products between rows or columns of $A$.

**Row gram** $G_\text{row} = A A^\top$

Entry $(G_\text{row})_{i,j} = \langle A_{i,:},\, A_{j,:} \rangle$: the dot product between the attention distributions of positions $i$ and $j$. High values indicate that two source positions distribute their attention similarly.

**Column gram** $G_\text{col} = A^\top A$

Entry $(G_\text{col})_{i,j} = \langle A_{:,i},\, A_{:,j} \rangle$: the dot product between the attention received by target positions $i$ and $j$. High values indicate that two target positions are attended to by a similar set of source positions.

**Skew row gram** $G_\text{skew,row} = S(A) S(A)^\top$

where $S$ is the skew transform (see below). Entry $(G_\text{skew,row})_{i,j} = \langle S(A)_{i,:},\, S(A)_{j,:} \rangle$: the dot product between the relative-position attention profiles of positions $i$ and $j$. High values indicate that two source positions distribute attention similarly as a function of relative position.

**Skew column gram** $G_\text{skew,col} = S(A)^\top S(A)$

Entry $(G_\text{skew,col})_{i,j} = \langle S(A)_{:,i},\, S(A)_{:,j} \rangle$: the dot product between columns of the skewed matrix, capturing which relative offsets receive similar amounts of attention.

### Log-space cosine similarity matrices

These apply cosine similarity to log-transformed attention weights, emphasizing the *shape* of the distribution rather than its magnitude.

First, compute $A_\text{log} = \log(A + \epsilon)$ (with $\epsilon = 10^{-9}$ and NaN/inf guarding). Then:

**Log row cosine** $C_\text{row}$

$$C_{i,j} = \frac{\langle (A_\text{log})_{i,:},\; (A_\text{log})_{j,:} \rangle}{\lVert (A_\text{log})_{i,:} \rVert \cdot \lVert (A_\text{log})_{j,:} \rVert}$$

Cosine similarity between log-space row vectors. The log transform compresses the dynamic range of attention weights, so this captures structural similarity even when magnitudes differ greatly (e.g., a sharp peak vs. a broad peak at the same relative positions).

**Log column cosine** $C_\text{col}$

Same as above but computed on columns of $A_\text{log}$.

**Log skew row cosine** $C_\text{skew,row}$

Cosine similarity on the rows of $S(A_\text{log})$, combining log-space normalization with relative-position alignment.

**Log skew column cosine** $C_\text{skew,col}$

Same as above but computed on columns of $S(A_\text{log})$.

---

## The Skew Transform

The skew transform $S: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n}$ right-aligns the rows of a lower-triangular matrix so that the diagonal ends up in the rightmost column:

$$S(A)\big[i,\; j + (n - i - 1)\big] := A[i,j] \quad \text{for } j \le i$$

For example, with $n=4$:

$$A = \begin{pmatrix} a & 0 & 0 & 0 \\ b & c & 0 & 0 \\ d & e & f & 0 \\ g & h & i & j \end{pmatrix} \;\;\longrightarrow\;\; S(A) = \begin{pmatrix} 0 & 0 & 0 & a \\ 0 & 0 & b & c \\ 0 & d & e & f \\ g & h & i & j \end{pmatrix}$$

This aligns rows by *relative* position: column $n-1$ always contains the self-attention weight, column $n-2$ the weight on the immediately preceding token, and so on. Without this transform, the diagonal entries of $A$ lie in different columns for each row, making row-wise comparisons conflate positional structure with sequence position.

---

## Gram-to-Scalar Pipeline

Each of the eight gram/similarity matrices $G \in \mathbb{R}^{n \times n}$ is reduced to a set of scalars as follows:

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
| `last_tok` | Last column of $A$ |
| `prev_tok` | Subdiagonal of $A$ |
| `row_entropy` | Per-row entropy $H_i$ |
| `attn_distance` | Per-row expected attention distance $\delta_i$ |
| `row_max` | Per-row maximum attention $m_i$ |
| `col_sum` | Per-column attention sum $c_j$ |
| `band_energy` | Fraction of attention in diagonal band (standalone scalar) |
| `gram.row` | Row gram $AA^\top$ |
| `gram.col` | Column gram $A^\top A$ |
| `gram.skew.row` | Skew row gram $S(A) S(A)^\top$ |
| `gram.skew.col` | Skew column gram $S(A)^\top S(A)$ |
| `log.gram.row` | Log-space row cosine similarity |
| `log.gram.col` | Log-space column cosine similarity |
| `log.gram.skew.row` | Log-space skew row cosine similarity |
| `log.gram.skew.col` | Log-space skew column cosine similarity |

For gram contexts, the statistic is additionally prefixed with `hist` (since it is computed on the histogram), giving names like `feat.gram.row.hist.entropy` or `feat.log.gram.skew.row.hist.linreg.slope`.
