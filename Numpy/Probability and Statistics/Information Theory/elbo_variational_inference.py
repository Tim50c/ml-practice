import numpy as np


def compute_elbo(
    x: list[float],
    q_mean: float,
    q_std: float,
    prior_mean: float,
    prior_std: float,
    likelihood_std: float,
    n_samples: int = 1000,
) -> float:
    """
    Estimate the ELBO for a 1D latent-variable model using
    Monte Carlo variational inference.

    Model:
        z ~ N(prior_mean, prior_std²)
        x | z ~ N(z, likelihood_std²)

    Variational distribution:
        q(z) = N(q_mean, q_std²)

    Args:
        x: Observed data points.
        q_mean: Mean of q(z).
        q_std: Standard deviation of q(z).
        prior_mean: Mean of prior p(z).
        prior_std: Standard deviation of prior p(z).
        likelihood_std: Standard deviation of p(x|z).
        n_samples: Number of Monte Carlo samples.
        rng: Optional NumPy random generator.

    Returns:
        Estimated ELBO value.
    """
    x_arr = np.asarray(x, dtype=np.float64)

    if x_arr.ndim != 1:
        raise ValueError("x must be 1-dimensional")

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if q_std <= 0 or prior_std <= 0 or likelihood_std <= 0:
        raise ValueError("Standard deviations must be positive")

    rng = np.random.default_rng()

    # =========================================================
    # Sample from q(z) using reparameterization
    # =========================================================

    z_samples  = (
        q_mean
        + q_std * rng.standard_normal(n_samples)
    )

    # =========================================================
    # E_q[log p(x | z)]
    # =========================================================

    likelihood_var = likelihood_std ** 2

    diff = x_arr[:, None] - z_samples

    log_likelihood = (
        -0.5 * np.log(2.0 * np.pi * likelihood_var)
        - 0.5 * (diff ** 2) / likelihood_var
    )

    expected_log_likelihood = log_likelihood.sum(axis=0).mean()

    # =========================================================
    # E_q[log p(z)]
    # =========================================================

    prior_var = prior_std ** 2

    log_prior = (
        -0.5 * np.log(2.0 * np.pi * prior_var)
        - 0.5 * ((z_samples - prior_mean) ** 2) / prior_var
    )

    expected_log_prior = log_prior.mean()

    # =========================================================
    # Entropy H[q(z)]
    # =========================================================

    entropy_q = 0.5 * np.log(
        2.0 * np.pi * np.e * (q_std ** 2)
    )

    elbo = (
        expected_log_likelihood
        + expected_log_prior
        + entropy_q
    )

    return float(elbo)

np.random.seed(42)
elbo = compute_elbo([1.0], 0.5, 0.7, 0.0, 1.0, 1.0, n_samples=10000)
print(round(elbo, 2))
