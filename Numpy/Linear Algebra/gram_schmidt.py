import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    input_vectors = np.asarray(vectors, dtype=float)
    bases = []

    for v in input_vectors:
        u = v.copy()

        for b in bases:
            projection_scalar = np.dot(u, b)
            u = u - projection_scalar * b

        norm = np.linalg.norm(u)
        if norm >= tol:
            bases.append(u / norm)

    return bases


def are_orthonormal(vectors, atol=1e-8):
    """
    Return True if the input vectors are orthonormal, otherwise False.

    Parameters
    ----------
    vectors : list[np.ndarray]
        List of 1D numpy arrays with the same length.
    atol : float
        Absolute tolerance for floating-point comparisons.

    Returns
    -------
    bool
    """
    if not vectors:
        return False

    vectors = [np.asarray(v, dtype=float).ravel() for v in vectors]

    # All vectors must have the same dimension
    dim = vectors[0].shape[0]
    if any(v.shape[0] != dim for v in vectors):
        return False

    # Stack as rows
    M = np.vstack(vectors)

    # Gram matrix: dot products between all pairs
    G = M @ M.T

    # For orthonormal vectors, this must be the identity matrix
    return np.allclose(G, np.eye(len(vectors)), atol=atol)

basis = orthonormal_basis([[2, 1, 0, 1], [1, 2, 1, 0], [3, 2, 1, 2], [1, 0, 2, 1]])
print([b.round(4) for b in basis])
print(are_orthonormal(basis))