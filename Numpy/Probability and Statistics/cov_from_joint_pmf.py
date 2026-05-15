import numpy as np


def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.

    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])

    Returns:
        Covariance of X and Y as a float
    """
    X = np.asarray(x_values, dtype=float)
    Y = np.asarray(y_values, dtype=float)

    X_pmf = np.sum(joint_pmf, axis=1)
    Y_pmf = np.sum(joint_pmf, axis=0)

    X_expected = np.sum(X * X_pmf)
    Y_expected = np.sum(Y * Y_pmf)

    XY = np.outer(X, Y)
    XY_expected = np.sum(XY * joint_pmf)

    return XY_expected - X_expected * Y_expected

x_values = [0, 1]
y_values = [0, 1]
joint_pmf = np.array([[0.4, 0.1], [0.1, 0.4]])
print(round(covariance_from_joint_pmf(x_values, y_values, joint_pmf), 4))