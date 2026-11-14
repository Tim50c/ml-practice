import numpy as np


def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    """
    Build a decision tree for regression and predict on test data.

    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        X_test: Test features, shape (m_samples, n_features)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node

    Returns:
        List of predictions for X_test, rounded to 4 decimal places
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    n_samples, n_features = X_train.shape

    def find_best_split(X, y):
        n_S, n_features_S = X.shape
        mse_S = np.var(y)
        best_reduction = 0.0
        best_feature = None
        best_threshold = None

        for j in range(n_features_S):
            indices = np.argsort(X[:, j], stable=True)
            Xj = X[indices, j].copy()
            yj = y[indices].copy()

            vals = np.unique(Xj)
            thresholds = (vals[1:] + vals[:-1]) / 2
            split_indices = np.searchsorted(Xj, thresholds, side='right')

            cumsum = np.cumsum(yj)
            cumsum2 = np.cumsum(yj ** 2)

            for s in range(len(split_indices)):
                threshold = thresholds[s]

                n_l = split_indices[s]
                n_r = n_S - n_l

                if n_l == 0 or n_r == 0:
                    continue

                mse_l = cumsum2[n_l - 1] / n_l - (cumsum[n_l - 1] / n_l) ** 2
                mse_r = (cumsum2[n_S - 1] - cumsum2[n_l - 1]) / n_r - ((cumsum[n_S - 1] - cumsum[n_l - 1]) / n_r) ** 2

                reduction = mse_S - (n_l / n_S * mse_l) - (n_r / n_S * mse_r)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = j
                    best_threshold = threshold

        return (None, None) if best_reduction == 0.0 else (best_feature, best_threshold)

    def _build(X, y, depth):
        n_S = len(y)
        if (depth >= max_depth) or (n_S < min_samples_split) or np.var(y) == 0.0:
            return {
                'leaf': True,
                'value': np.mean(y)
            }

        feature, threshold = find_best_split(X, y)

        if feature is None or threshold is None:
            return {
                'leaf': True,
                'value': np.mean(y)
            }

        mask = X[:, feature] <= threshold
        X_left = X[mask]
        X_right = X[~mask]
        y_left = y[mask]
        y_right = y[~mask]

        L = _build(X_left, y_left, depth + 1)
        R = _build(X_right, y_right, depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': L,
            'right': R,
        }

    def _predict_one(node, x):
        if node['leaf']:
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return _predict_one(node['left'], x)
        else:
            return _predict_one(node['right'], x)

    root = _build(X_train, y_train, 0)

    y_test = [round(_predict_one(root, x), 4) for x in X_test]

    return y_test

X_train = [[0], [1], [2], [3]]
y_train = [0.0, 0.0, 1.0, 1.0]
X_test = [[0.5], [2.5]]
max_depth = 1
print(decision_tree_regressor(X_train, y_train, X_test, max_depth))