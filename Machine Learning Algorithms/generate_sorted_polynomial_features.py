import numpy as np
from itertools import combinations_with_replacement


def polynomial_features(X, degree):
    n_rows, n_cols = X.shape

    combs = []
    for d in range(0, degree + 1):
        combs = combs + list(combinations_with_replacement(range(n_cols), d))

    X_poly = np.zeros((n_rows, len(combs)), dtype=float)

    for i, comb in enumerate(combs):
        poly = np.prod(X[:, list(comb)], axis=1)
        X_poly[:, i] = poly

    return np.sort(X_poly, axis=1)


print(polynomial_features(np.array([[1, 2], [3, 4], [5, 6]]), 3))
