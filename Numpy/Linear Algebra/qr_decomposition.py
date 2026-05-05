import numpy as np


def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.

    Args:
        A: An m x n matrix represented as list of lists

    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    A_ = np.asarray(A, dtype=float).copy()
    m, n = A_.shape

    Q = np.zeros_like(A_)
    R = np.zeros((n, n))

    tol = 1e-12

    for k in range(n):
        Q[:, k] = A_[:, k]
        for j in range(k):
            R[j, k] = np.dot(Q[:, j], A_[:, k]) # Modified Gram-Schmidt: R[j, k] = np.dot(Q[:, j], Q[:, k]) to prevent floating point rounding errors
            Q[:, k] -= R[j, k] * Q[:, j]

        R[k, k] = np.linalg.norm(Q[:, k])

        if R[k, k] > tol:
            Q[:, k] /= R[k, k]
        else:
            R[k, k] = 0.0
            Q[:, k] = 0.0

    return Q.tolist(), R.tolist()

Q, R = qr_decomposition([[1, 0], [0, 1]])
print([[round(x, 4) for x in row] for row in Q])
print([[round(x, 4) for x in row] for row in R])
