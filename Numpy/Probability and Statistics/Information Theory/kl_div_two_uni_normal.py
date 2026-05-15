import numpy as np


def kl_divergence_normal(
    mu_p: float,
    sigma_p: float,
    mu_q: float,
    sigma_q: float
) -> float:
    """
    Compute the KL divergence D_KL(P || Q) between two
    univariate normal distributions:

        P ~ N(mu_p, sigma_p^2)
        Q ~ N(mu_q, sigma_q^2)

    Args:
        mu_p: Mean of P
        sigma_p: Standard deviation of P (> 0)
        mu_q: Mean of Q
        sigma_q: Standard deviation of Q (> 0)

    Returns:
        KL divergence D_KL(P || Q)
    """

    # Convert to float
    mu_p = float(mu_p)
    sigma_p = float(sigma_p)
    mu_q = float(mu_q)
    sigma_q = float(sigma_q)

    # Finite-value validation
    if not np.isfinite([mu_p, sigma_p, mu_q, sigma_q]).all():
        raise ValueError("All parameters must be finite numbers")

    # Standard deviation validation
    if sigma_p <= 0:
        raise ValueError("sigma_p must be positive")

    if sigma_q <= 0:
        raise ValueError("sigma_q must be positive")

    kl_div = (
        np.log(sigma_q / sigma_p)
        + (
            sigma_p**2
            + (mu_p - mu_q)**2
        ) / (2 * sigma_q**2)
        - 0.5
    )

    return float(kl_div)