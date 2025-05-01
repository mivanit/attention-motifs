import numpy as np
from jaxtyping import Float
from numba import njit

@njit(cache=True)
def cosine_similarity_matrix(
    X: Float[np.ndarray, "n n"],
    col: bool = False,
    eps: float = 1e-10,
) -> Float[np.ndarray, "n n"]:

    if col:
        # treat columns as vectors: use X.T @ X  (both operands contiguous)
        dot = X.T @ X
        norms = np.sqrt((X * X).sum(0))          # column norms
    else:
        dot = X @ X.T
        norms = np.sqrt((X * X).sum(1))          # row norms

    denom = norms.reshape(-1, 1) * norms.reshape(1, -1) + eps
    return dot / denom
