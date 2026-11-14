import numpy as np
from typing import Tuple


def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    best_gini = np.inf
    best_feature = None
    best_threshold = None

    n_samples, n_features = X.shape
    n_pos = np.sum(y)

    for i in range(n_features):
        x = np.sort(np.unique(X[:, i]))

        for threshold in x[0:len(x)-1]:
            mask = X[:, i] <= threshold

            y_l = y[mask]

            n_l = len(y_l)
            n_r = n_samples - n_l

            k_l = np.sum(y_l)
            k_r = n_pos - k_l

            p_l = k_l / n_l
            p_r = k_r / n_r

            g_l = 2 * p_l * (1 - p_l)
            g_r = 2 * p_r * (1 - p_r)

            gini = (n_l / n_samples) * g_l + (n_r / n_samples) * g_r

            if gini < best_gini:
                best_gini = gini
                best_feature = i
                best_threshold = threshold

    return int(best_feature), float(best_threshold)

X1 = np.array([[2.5], [3.5], [1.0], [4.0]])
y1 = np.array([0, 1, 0, 1])
f1, t1 = find_best_split(X1, y1)
print(f1, round(t1, 4))
