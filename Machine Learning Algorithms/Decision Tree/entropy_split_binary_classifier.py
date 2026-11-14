import numpy as np


def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)

    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    best_ig = -np.inf
    best_feature = None
    best_threshold = None

    n_samples, n_features = X.shape

    _, counts = np.unique(y, return_counts=True)
    p = counts / n_samples
    H = 0.0 if np.any(p == 1.0) else -1.0*np.sum(p * np.log2(p))

    for i in range(n_features):
        x = np.sort(X[:, i])
        thresholds = (x[:-1] + x[1:]) / 2

        for threshold in thresholds:
            mask = X[:, i] <= threshold

            y_l = y[mask]
            y_r = y[~mask]

            n_l = len(y_l)
            n_r = len(y_r)

            _, counts_l = np.unique(y_l, return_counts=True)
            p_l = counts_l / n_l
            H_l = 0.0 if np.any(p_l == 1.0) else -1.0*np.sum(p_l * np.log2(p_l))

            _, counts_r = np.unique(y_r, return_counts=True)
            p_r = counts_r / n_r
            H_r = 0.0 if np.any(p_r == 1.0) else -1.0*np.sum(p_r * np.log2(p_r))

            ig = H - ((n_l / n_samples) * H_l + (n_r / n_samples) * H_r)

            if ig > best_ig:
                best_ig = ig
                best_feature = i
                best_threshold = threshold

    return int(best_feature), float(best_threshold), float(best_ig)

X = np.array([[1, 3], [2, 1], [3, 4], [4, 2]])
y = np.array([0, 1, 0, 1])
result = entropy_split_selection(X, y)
print(f"({result[0]}, {result[1]}, {result[2]})")
