import numpy as np
from jaxtyping import Float
# from numba import jit

# @jit(cache=True)
def cosine_similarity_matrix(
	X: Float[np.ndarray, "n n"], col: bool = False, eps: float = 1e-10
) -> np.ndarray:
	# If col==True, treat columns as vectors (i.e., work on the transposed matrix)
	X_: Float[np.ndarray, "n n"] = X.T if col else X

	# Compute the dot product matrix for the chosen vectors (rows of X_)
	dot_product: Float[np.ndarray, "n n"] = X_ @ X_.T

	# Compute the norm of each vector (row) in X_
	norms: Float[np.ndarray, " n"] = np.linalg.norm(X_, axis=1)

	# Compute the cosine similarity matrix, adding eps to avoid division by zero
	similarity_matrix: np.ndarray = dot_product / (
		(norms[:, None] * norms[None, :]) + eps
	)
	return similarity_matrix
