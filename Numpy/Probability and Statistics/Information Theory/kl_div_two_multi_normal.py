import numpy as np


def multivariate_kl_divergence(
    mu_p: np.ndarray,
    cov_p: np.ndarray,
    mu_q: np.ndarray,
    cov_q: np.ndarray
) -> float:
    """
    Compute the KL divergence D_KL(P || Q) between two
    multivariate Gaussian distributions:

        P ~ N(mu_p, cov_p)
        Q ~ N(mu_q, cov_q)

    Args:
        mu_p: Mean vector of P
        cov_p: Covariance matrix of P
        mu_q: Mean vector of Q
        cov_q: Covariance matrix of Q

    Returns:
        KL divergence as a float.
    """

    mu_p = np.asarray(mu_p, dtype=float)
    mu_q = np.asarray(mu_q, dtype=float)
    cov_p = np.asarray(cov_p, dtype=float)
    cov_q = np.asarray(cov_q, dtype=float)

    # Shape validation
    if mu_p.ndim != 1 or mu_q.ndim != 1:
        raise ValueError("Mean vectors must be 1D")

    if cov_p.ndim != 2 or cov_q.ndim != 2:
        raise ValueError("Covariance matrices must be 2D")

    if mu_p.shape != mu_q.shape:
        raise ValueError("Mean vectors must have the same shape")

    p = mu_p.shape[0]

    if cov_p.shape != (p, p):
        raise ValueError("cov_p has invalid shape")

    if cov_q.shape != (p, p):
        raise ValueError("cov_q has invalid shape")

    # Finite-value validation
    if (
        not np.isfinite(mu_p).all()
        or not np.isfinite(mu_q).all()
        or not np.isfinite(cov_p).all()
        or not np.isfinite(cov_q).all()
    ):
        raise ValueError("Inputs must contain only finite values")

    # Symmetry check
    if not np.allclose(cov_p, cov_p.T):
        raise ValueError("cov_p must be symmetric")

    if not np.allclose(cov_q, cov_q.T):
        raise ValueError("cov_q must be symmetric")

    # Positive definiteness check
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)

    if sign_p <= 0:
        raise ValueError("cov_p must be positive definite")

    if sign_q <= 0:
        raise ValueError("cov_q must be positive definite")

    # Use solve instead of explicit inverse for stability
    cov_q_inv_cov_p = np.linalg.solve(cov_q, cov_p)

    trace_term = np.trace(cov_q_inv_cov_p)

    mu_diff = mu_q - mu_p

    mahalanobis_term = mu_diff @ np.linalg.solve(cov_q, mu_diff)

    kl_div = 0.5 * (
        logdet_q
        - logdet_p
        - p
        + trace_term
        + mahalanobis_term
    )

    # Numerical protection
    return float(max(kl_div, 0.0))