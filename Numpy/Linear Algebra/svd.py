import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.

    Args:
        A: A 2x2 numpy array

    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    B = A.T @ A

    theta = np.pi / 4 if B[0, 0] == B[1, 1] else (1 / 2) * np.arctan2(2 * B[0, 1], (B[0, 0] - B[1, 1]))

    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    D = V.T @ B @ V

    S = np.sort(np.sqrt(np.diag(D)))[::-1]

    U = A @ V @ np.diag(1 / S)

    return U, S, V.T

A = np.array([[1, 2], [3, 4]])
U, S, Vt = svd_2x2_singular_values(A)
print('Implement:')
print(U, S, Vt, sep='\n')

u, s, vh = np.linalg.svd(A)
print('Built-in:')
print(u, s, vh, sep='\n')