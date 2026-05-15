import numpy as np
from scipy.stats import t as student_t
import math

# Two-sample independent t-test (Welch's t-test) to determine if two samples have significantly different means.
def two_sample_t_test(
    sample1,
    sample2,
    alpha=0.05,
):
    """
    Perform Welch's two-sample t-test.

    Args:
        sample1: First sample
        sample2: Second sample
        alpha: Significance level

    Returns:
        Dictionary containing:
            - t_statistic
            - p_value
            - degrees_of_freedom
            - reject_null
            - cohens_d
    """
    x1 = np.asarray(sample1, dtype=np.float64)
    x2 = np.asarray(sample2, dtype=np.float64)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("Samples must be 1-dimensional")

    n1 = x1.size
    n2 = x2.size

    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample must contain at least 2 observations")

    mean1 = x1.mean()
    mean2 = x2.mean()

    var1 = x1.var(ddof=1)
    var2 = x2.var(ddof=1)

    # Welch standard error
    var1_n = var1 / n1
    var2_n = var2 / n2

    std_err = np.sqrt(var1_n + var2_n)

    if std_err == 0:
        raise ValueError("Pooled standard error is zero")

    # Welch t-statistic
    t_stat = (mean1 - mean2) / std_err

    # Welch-Satterthwaite degrees of freedom
    numerator = (var1_n + var2_n) ** 2

    denominator = (
        (var1_n ** 2) / (n1 - 1)
        + (var2_n ** 2) / (n2 - 1)
    )

    df = numerator / denominator

    # Two-tailed p-value
    p_value = 2.0 * student_t.sf(np.abs(t_stat), df)

    # Cohen's d (pooled SD version)
    pooled_var = (
        ((n1 - 1) * var1 + (n2 - 1) * var2)
        / (n1 + n2 - 2)
    )

    pooled_std = np.sqrt(pooled_var)

    cohens_d = (
        (mean1 - mean2) / pooled_std
        if pooled_std > 0
        else 0.0
    )

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": float(df),
        "reject_null": bool(p_value < alpha),
        "cohens_d": float(cohens_d),
    }



# Calculate Statistical Power for A/B Testing
def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05,
                    two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test.

    Parameters:
    effect_size: Cohen's d (standardized effect size)
    sample_size_per_group: Number of observations per group
    alpha: Significance level (default 0.05)
    two_tailed: Whether the test is two-tailed (default True)

    Returns:
    Statistical power as a float rounded to 4 decimal places
    """
    if sample_size_per_group <= 0:
        raise ValueError("sample_size_per_group must be positive")

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    # Approximate critical z-values
    # format: confidence -> (one_tail, two_tail)
    Z_TABLE = {
        0.80: (0.8416, 1.2816),
        0.90: (1.2816, 1.6449),
        0.95: (1.6449, 1.9600),
        0.98: (2.0537, 2.3263),
        0.99: (2.3263, 2.5758),
    }

    confidence = round(1 - alpha, 2)

    if confidence not in Z_TABLE:
        raise ValueError(
            f"Unsupported alpha={alpha}. "
            f"Supported confidence levels: {list(Z_TABLE.keys())}"
        )

    one_tail_z, two_tail_z = Z_TABLE[confidence]
    z_critical = two_tail_z if two_tailed else one_tail_z

    # Standard normal CDF using erf approximation
    def phi(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # Non-centrality parameter
    ncp = effect_size * math.sqrt(sample_size_per_group / 2)

    # Power calculation
    if two_tailed:
        power = (
                1
                - phi(z_critical - ncp)
                + phi(-z_critical - ncp)
        )
    else:
        power = 1 - phi(z_critical - ncp)

    return round(power, 4)


result = two_sample_t_test([12, 14, 13, 15, 14], [8, 9, 10, 9, 11], alpha=0.05)
print(f"{result['t_statistic']:.4f},{result['p_value']:.6f},{result['reject_null']},{result['cohens_d']:.4f}")

print(calculate_power(0.2, 200, 0.05, True))
