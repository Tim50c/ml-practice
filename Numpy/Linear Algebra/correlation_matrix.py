import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y = X
    else:
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

    m = X.shape[0]

    Xc = X - np.mean(X, axis=0)
    Yc = Y - np.mean(Y, axis=0)

    cov = (Xc.T @ Yc) / (m - 1)

    stdX = np.std(X, axis=0, ddof=1)
    stdY = np.std(Y, axis=0, ddof=1)

    denom = np.outer(stdX, stdY)

    corr = cov / denom

    # Handle division by zero (constant columns)
    corr[denom == 0] = np.nan

    return corr

print(calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])))