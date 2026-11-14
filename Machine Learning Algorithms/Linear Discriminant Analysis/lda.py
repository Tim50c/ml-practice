import numpy as np

def lda_predict(X_train, y_train, X_test):
    """
    Fit a binary LDA classifier on (X_train, y_train) and predict labels for X_test.

    Args:
        X_train: array-like of shape (n_samples, n_features)
        y_train: array-like of shape (n_samples,) with values in {0, 1}
        X_test:  array-like of shape (n_test, n_features)

    Returns:
        List[int] of predicted labels (0 or 1) for each row in X_test.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)

    Sw = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)

    w = np.linalg.inv(Sw) @ (mu1 - mu0)

    c = w.T @ (mu0 + mu1) / 2

    y_test = np.where(X_test @ w > c, 1, 0)

    return y_test.tolist()

X_train = [[0,0],[0,1],[1,0],[5,5],[5,6],[6,5]]
y_train = [0,0,0,1,1,1]
X_test = [[0.5,0.5],[5.5,5.5],[3,3]]
print(lda_predict(X_train, y_train, X_test))

