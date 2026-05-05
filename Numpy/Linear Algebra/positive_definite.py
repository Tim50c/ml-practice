import numpy as np


def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.

    Args:
        matrix: A 2D list representing a square matrix

    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    tol = 1e-10

    A = np.asarray(matrix, dtype=float)
    n = len(A)
    is_symmetric = np.allclose(A, A.T, atol=tol)

    QQ = np.eye(n)

    for _ in range(1000):
        Q, R = np.linalg.qr(A)

        A = R @ Q
        QQ = QQ @ Q

    eigenvalues = np.round(np.sort(np.diag(A)), 4)
    eigenvectors = QQ

    is_positive_definite = False

    if is_symmetric and np.all(eigenvalues > tol):
        is_positive_definite = True

    return {
        'is_positive_definite': is_positive_definite,
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
    }

result = check_positive_definite([[2, 1], [1, 2]])
print({'is_positive_definite': result['is_positive_definite'],
       'eigenvalues': result['eigenvalues'],
       'eigenvectors': result['eigenvectors']})