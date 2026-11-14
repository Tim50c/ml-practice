import numpy as np


def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10,
                       seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.

    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Predicted labels for X_test
    """
    rng = np.random.default_rng(seed)

    n_samples, n_features = X_train.shape
    stumps = []

    for _ in range(n_estimators):
        indices = rng.integers(0, n_samples, size=n_samples)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        total_pos = np.sum(y_boot)

        best_err = n_samples + 1
        best_feature = None
        best_threshold = None
        best_c_left = None
        best_c_right = None

        for j in range(n_features):
            # Sort the feature values to efficiently evaluate splits
            order = np.argsort(X_boot[:, j])
            v_s = X_boot[order, j]
            y_s = y_boot[order]

            n_left = 0
            pos_left = 0
            n_right = n_samples
            pos_right = total_pos

            for k in range(n_samples):
                # Update running totals FIRST
                n_left += 1
                pos_left += y_s[k]
                n_right -= 1
                pos_right -= y_s[k]

                # Skip threshold evaluation if the next value is identical
                # Otherwise, identical values get split into different nodes
                if k < n_samples - 1 and v_s[k] == v_s[k + 1]:
                    continue

                # Calculate errors based on assigning the majority class to each leaf
                err_left = n_left - max(pos_left, n_left - pos_left)
                err_right = (n_right - max(pos_right, n_right - pos_right)) if n_right > 0 else 0

                err = err_left + err_right

                if err < best_err:
                    best_err = err
                    best_feature = j
                    best_threshold = v_s[k]

                    # Store the majority class for left and right branches
                    best_c_left = 1 if pos_left > n_left - pos_left else 0
                    best_c_right = 0 if n_right == 0 else (1 if pos_right > n_right - pos_right else 0)

        stumps.append((best_feature, best_threshold, best_c_left, best_c_right))

    n_test = X_test.shape[0]
    votes = np.zeros(n_test)

    # Aggregate predictions across all stumps
    for m in range(n_estimators):
        j, threshold, c0, c1 = stumps[m]
        votes += np.where(X_test[:, j] <= threshold, c0, c1)

    # Final majority vote
    y_pred = np.where(votes > n_estimators / 2, 1, 0)

    return y_pred