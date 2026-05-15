import numpy as np


def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.

    Args:
        P: True probability distribution
        Q: Predicted probability distribution

    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    eps = 1e-10

    P_np = np.asarray(P, dtype=float)
    Q_np = np.asarray(Q, dtype=float)

    if P_np.shape != Q_np.shape:
        raise ValueError("P and Q must have the same shape")

    if np.any(P_np < 0) or np.any(Q_np < 0):
        raise ValueError("Probabilities must be non-negative")

    if not np.isclose(np.sum(P_np), 1.0):
        raise ValueError("P must sum to 1")

    if not np.isclose(np.sum(Q_np), 1.0):
        raise ValueError("Q must sum to 1")

    mask_p = P_np > 0.0

    entropy = -np.sum(P_np[mask_p] * np.log(P_np[mask_p]))

    Q_safe = np.clip(Q_np, eps, 1.0)
    cross_entropy = -np.sum(P_np[mask_p] * np.log(Q_safe[mask_p]))

    return entropy, cross_entropy

