import numpy as np

def gaussian_elimination(A, b, tol=1e-10):
    """
    Find the unique solution for Ax = b, A is a square matrix
    :param A: Input matrix of shape (n, n) (left-hand side matrix in Ax = b)
    :param b: Input vector of shape (n,) (right-hand side vector in Ax = b)
    :param tol: Tolerance for considering values as zero
    :return: x - the unique solution of Ax = b
    """
    n = len(A)

    augmented = np.column_stack((A, b))
    augmented = augmented.astype(float)

    for i in range(n):
        pivot_row = i
        max_val = np.abs(augmented[i, i])

        for j in range(i + 1, n):
            if abs(augmented[j, i]) > max_val:
                pivot_row = j
                max_val = abs(augmented[j, i])

        if pivot_row != i:
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]

        if abs(augmented[i, i]) < tol:
            return "System has no unique solution"

        pivot = augmented[i, i]
        augmented[i] = augmented[i] / pivot

        for j in range(i + 1, n):
            augmented[j] = augmented[j] - augmented[j, i] * augmented[i]

    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = augmented[i, -1]
        for j in range(i + 1, n):
            x[i] = x[i] - x[j] * augmented[i, j]

    return x


def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix using Gaussian elimination.

    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero

    Returns:
        The rank of the matrix (integer)
    """

    # Gaussian elimination
    A = A.astype(float)
    m, n = A.shape
    rank = 0
    row = 0

    for col in range(n):
        max_val = tol
        pivot_row = None

        for i in range(row, m):
            current = abs(A[i, col])
            if current > max_val:
                max_val = current
                pivot_row = i

        if pivot_row is None:
            continue
        if row != pivot_row:
            A[[row, pivot_row]] = A[[pivot_row, row]]

        A[row] = A[row] / A[row, col]

        for j in range(row + 1, m):
            A[j] = A[j] - A[j, col] * A[row]

        row += 1
        rank += 1

        if row == m:
            break

    return rank


def rref(matrix):
    matrix = matrix.astype(float)
    m, n = matrix.shape
    row = 0
    tol = 1e-10

    for col in range(n):
        max_val = abs(matrix[row, col])
        pivot_row = row

        for i in range(row + 1, m):
            current = abs(matrix[i, col])
            if current > max_val:
                max_val = current
                pivot_row = i

        if max_val < tol:
            continue

        if row != pivot_row:
            matrix[[row, pivot_row]] = matrix[[pivot_row, row]]

        matrix[row] = matrix[row] / matrix[row, col]

        for j in range(m):
            if j != row:
                matrix[j] = matrix[j] - matrix[j, col] * matrix[row]

        row += 1

        if row == m:
            break

    return matrix

matrix = np.array([ [1, 2, -1], [2, 4, -1], [-2, -4, -3]])
output = rref(matrix)
print(output)