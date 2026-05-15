import numpy as np


def fit_gmm_1d(
    X,
    K,
    initial_means,
    initial_variances,
    initial_weights,
    n_iterations,
    eps=1e-6,
):
    """
    Fit a 1D Gaussian Mixture Model using the EM algorithm.

    Args:
        X: 1D array-like of samples
        K: Number of Gaussian components
        initial_means: Initial component means
        initial_variances: Initial component variances
        initial_weights: Initial mixture weights
        n_iterations: Number of EM iterations
        eps: Minimum allowed variance for numerical stability

    Returns:
        Dictionary containing means, variances, and weights.
    """
    X = np.asarray(X, dtype=np.float64)
    means = np.asarray(initial_means, dtype=np.float64)
    variances = np.asarray(initial_variances, dtype=np.float64)
    weights = np.asarray(initial_weights, dtype=np.float64)

    # Ensure valid shapes
    if X.ndim != 1:
        raise ValueError("X must be 1-dimensional")

    if not (
        len(means) == len(variances) == len(weights) == K
    ):
        raise ValueError("Parameter lengths must equal K")

    # Normalize weights
    weights /= weights.sum()

    # Prevent zero variances
    variances = np.maximum(variances, eps)

    N = X.size
    X_col = X[:, None]

    for _ in range(n_iterations):
        # =========================================================
        # E-step
        # =========================================================

        diff = X_col - means

        log_prob = (
            np.log(weights)
            - 0.5 * (
                np.log(2.0 * np.pi * variances)
                + (diff ** 2) / variances
            )
        )

        # Stable log-sum-exp normalization
        log_prob_max = np.max(log_prob, axis=1, keepdims=True)

        responsibilities = np.exp(log_prob - log_prob_max)

        responsibilities /= responsibilities.sum(
            axis=1,
            keepdims=True,
        )

        # =========================================================
        # M-step
        # =========================================================

        Nk = responsibilities.sum(axis=0)

        weights = Nk / N

        means = (responsibilities.T @ X) / Nk

        diff = X_col - means

        variances = (
            np.sum(responsibilities * diff ** 2, axis=0)
            / Nk
        )

        variances = np.maximum(variances, eps)

    return {
        "means": np.round(means, 4).tolist(),
        "variances": np.round(variances, 4).tolist(),
        "weights": np.round(weights, 4).tolist(),
    }


result = fit_gmm_1d([0.0, 0.0, 5.0, 5.0, 5.0], 2, [0.0, 5.0], [0.5, 0.5], [0.4, 0.6], 8)
print(result)
