import numpy as np


def jensen_shannon_divergence(
    P: list[float],
    Q: list[float]
) -> float:
    """
    Compute the Jensen-Shannon divergence between two
    probability distributions.

    Args:
        P: First probability distribution
        Q: Second probability distribution

    Returns:
        Jensen-Shannon divergence in nats.
    """

    p = np.asarray(P, dtype=float)
    q = np.asarray(Q, dtype=float)

    # Shape validation
    if p.shape != q.shape:
        raise ValueError("P and Q must have the same shape")

    # Finite-value validation
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("P and Q must contain only finite values")

    # Probability validation
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Probabilities must be non-negative")

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("P must sum to 1")

    if not np.isclose(np.sum(q), 1.0):
        raise ValueError("Q must sum to 1")

    # Mixture distribution
    m = 0.5 * (p + q)

    # Compute KL(P || M)
    p_mask = p > 0
    kl_pm = np.sum(
        p[p_mask] * np.log(p[p_mask] / m[p_mask])
    )

    # Compute KL(Q || M)
    q_mask = q > 0
    kl_qm = np.sum(
        q[q_mask] * np.log(q[q_mask] / m[q_mask])
    )

    jsd = 0.5 * (kl_pm + kl_qm)

    # Numerical protection
    return float(max(jsd, 0.0))