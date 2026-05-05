import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros_like(b, dtype=float)
    D = np.diag(A)
    R = A - np.diag(D)

    for _ in range(n):
        x = (b - R @ x) / D

    return x.tolist()


def gauss_seidel(A, b, n):
    x = np.zeros_like(b)
    DL = np.tril(A)
    inv_DL = np.linalg.inv(DL)

    for _ in range(n):
        residual = b - A @ x
        x = inv_DL @ residual + x

    return x

print(solve_jacobi(np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]]), np.array([4, 6, 7]),5))