import math
import numpy as np

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.

    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial

    Returns:
        Probability of k successes
    """
    if not (0 <= k <= n and 0 <= p <= 1):
        return 0.0

    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))



def poisson_probability(k: int, lam: float) -> float:
    """
    Probability of observing exactly k events in a Poisson process.

    Parameters:
        k: Number of events
        lam: Expected number of events (λ)

    Returns:
        P(X = k)
    """
    if k < 0 or lam < 0:
        return 0.0

    if not isinstance(k, int):
        return 0.0

    return (
        (lam ** k)
        * math.exp(-lam)
        / math.factorial(k)
    )




def hypergeometric_pmf(N: int, K: int, n: int, k: int) -> float:
    """
    Calculate the PMF of the hypergeometric distribution.

    Args:
        N: Total population size
        K: Number of success states in population
        n: Number of draws (without replacement)
        k: Number of observed successes

    Returns:
        float: P(X = k), rounded to 4 decimal places
    """
    # Validate parameter ranges
    if not (0 <= K <= N and 0 <= n <= N or 0 <= k <= min(n, K)):
        return 0.0

    # Impossible event
    if n - k > N - K:
        return 0.0

    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)




def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """
    Calculate the probability of observing exactly k failures
    before achieving r successes in independent Bernoulli trials.

    Args:
        k: Number of failures (non-negative integer)
        r: Number of successes required (positive integer)
        p: Probability of success on each trial (0 < p <= 1)

    Returns:
        Probability P(X = k) rounded to 5 decimal places
    """
    if not (k >= 0 and r > 0 and 0 <= p <= 1):
        return 0.0

    return math.comb(k + r - 1, k) * (p ** r) * ((1 - p) ** k)




def normal_pdf(x: float, mean: float, std_dev: float) -> float:
    """
    Probability density function of a normal distribution.

    Parameters:
        x: Value at which to evaluate the PDF
        mean: Mean (μ)
        std_dev: Standard deviation (σ)

    Returns:
        f(x)
    """
    if std_dev <= 0:
        return 0.0

    variance = std_dev ** 2

    return (
        math.exp(-((x - mean) ** 2) / (2 * variance))
        / math.sqrt(2 * math.pi * variance)
    )


def exponential_distribution(x: list[float], lam: float) -> dict:
    """
    Compute exponential distribution properties.

    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution

    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    if lam <= 0.0:
        return {
            'pdf': None,
            'cdf': None,
            'mean': None,
            'variance': None,
        }

    x = np.asarray(x, dtype=float)
    mask = x >= 0
    exp_term = np.exp(-lam * x)

    pdf = np.where(mask, lam * exp_term, 0.0)
    cdf = np.where(mask, 1 - exp_term, 0.0)

    return {
        'pdf': pdf.tolist(),
        'cdf': cdf.tolist(),
        'mean': lam ** (-1),
        'variance': lam ** (-2),
    }



def chi_square_pdf(x: float, k: float) -> float:
    """
    Probability density function of a chi-square distribution.

    Parameters:
        x: Value at which to evaluate the PDF
        k: Degrees of freedom

    Returns:
        f(x)
    """
    if k <= 0 or x < 0:
        return 0.0

    half_k = k / 2

    return (
        (x ** (half_k - 1))
        * math.exp(-x / 2)
        / ((2 ** half_k) * math.gamma(half_k))
    )




def beta_distribution_stats(
    x: float,
    alpha: float,
    beta_param: float,
) -> dict:
    """
    Compute Beta distribution statistics.

    Parameters:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (α > 0)
        beta_param: Second shape parameter (β > 0)

    Returns:
        Dictionary containing:
            - pdf
            - mean
            - variance
    """
    if alpha <= 0 or beta_param <= 0:
        raise ValueError("alpha and beta_param must be positive")

    apb = alpha + beta_param

    pdf = 0.0

    if 0 <= x <= 1:
        beta_fn = (
            math.gamma(alpha)
            * math.gamma(beta_param)
            / math.gamma(apb)
        )

        pdf = (
            (x ** (alpha - 1))
            * ((1 - x) ** (beta_param - 1))
            / beta_fn
        )

    return {
        "pdf": pdf,
        "mean": alpha / apb,
        "variance": (
            alpha * beta_param
            / (apb ** 2 * (apb + 1))
        ),
    }
