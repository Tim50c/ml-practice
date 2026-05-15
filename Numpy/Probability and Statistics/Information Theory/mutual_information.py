import numpy as np


def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information I(X;Y).

    Args:
        joint_prob:
            2D joint probability distribution P(X,Y)

    Returns:
        Mutual information in nats.
    """

    joint = np.asarray(joint_prob, dtype=float)

    # Shape validation
    if joint.ndim != 2:
        raise ValueError("joint_prob must be a 2D array")

    # Finite-value validation
    if not np.isfinite(joint).all():
        raise ValueError("joint_prob must contain only finite values")

    # Probability validation
    if np.any(joint < 0):
        raise ValueError("Probabilities must be non-negative")

    total = np.sum(joint)

    if not np.isclose(total, 1.0):
        raise ValueError("joint_prob must sum to 1")

    # Marginal distributions
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)

    # Product distribution P(X)P(Y)
    px_py = np.outer(px, py)

    # Avoid log(0) and 0 * log(0)
    mask = joint > 0

    mi = np.sum(
        joint[mask] * np.log(joint[mask] / px_py[mask])
    )

    # Numerical protection against tiny negative values
    return float(max(mi, 0.0))