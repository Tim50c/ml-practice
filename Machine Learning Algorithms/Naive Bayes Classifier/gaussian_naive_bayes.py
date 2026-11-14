import numpy as np


def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.

    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)

    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    eps = 1e-9

    n_train, d = X_train.shape

    classes = np.unique(y_train)
    k = len(classes)

    mask = y_train[:, None] == classes[None, :]
    Nc = np.sum(mask, axis=0)
    pi_c = Nc / n_train

    mu = (X_train.T @ mask) / Nc
    mu2 = ((X_train ** 2).T @ mask) / Nc

    var = np.maximum(0, mu2 - mu ** 2) + eps

    V = 1 / var
    W = mu * V
    b = np.log(pi_c) - 0.5*np.sum(np.log(2*np.pi*var) + (mu**2)*V, axis=0)

    probs = X_test @ W - 0.5*((X_test ** 2) @ V) + b
    y_pred = classes[np.argmax(probs, axis=1)]

    return y_pred


print(gaussian_naive_bayes(np.array([[0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [5.0, 0.0], [4.0, 1.0], [3.0, 2.0]]), np.array([0, 0, 0, 1, 1, 1]), np.array([[1.0, 4.0], [4.0, 1.0], [2.5, 2.5]])))
