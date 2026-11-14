import numpy as np


def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.

    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)

    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # Gradient descent
    epochs = 10000
    n = len(y)
    lr = 1e-4

    beta1 = 0.9
    beta2 = 0.9
    m = 0.0
    v = 0.0

    eps = 1e-5

    for t in range(1, epochs + 1):
        y_hat = X @ W + b

        errors = y_hat - y

        grad_W = 2 / n * (X.T @ errors)
        grad_b = 2 / n * np.sum(errors)

        m = beta1 * m + (1 - beta1) * grad_W
        v = beta2 * v + (1 - beta2) * grad_W ** 2

        m_corrected = m / (1 - beta1 ** t)
        v_corrected = v / (1 - beta2 ** t)

        W = W - lr * m_corrected / (np.sqrt(v_corrected) + eps)
        b = b - lr * grad_b

    return W, b


import numpy as np


def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.

    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels

    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    epochs = 2000

    lr = 1e-3

    X_train = np.concatenate([X_train, X_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    n_samples, n_features = X_train.shape

    W = np.zeros((n_features,), dtype=float)
    b = 0.0

    for _ in range(epochs):
        z = X_train @ W + b
        probs = 1 / (1 + np.exp(-z))

        grad_W = X_train.T @ (probs - y_train)
        grad_b = np.sum(probs - y_train)

        W = W - lr * grad_W
        b = b - lr * grad_b

    def predict(X):
        z_test = X @ W + b
        probs_test = 1 / (1 + np.exp(-z_test))

        return np.where(probs_test >= 0.5, 1, 0)

    return predict


if __name__ == "__main__":
    main()
