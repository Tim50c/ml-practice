import numpy as np


class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.

    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.

    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """

    def __init__(self):
        self.n_components = 10
        # Projection matrix, size: (d, self.n_components)
        self._p: np.ndarray[np.ndarray] | None = None
        # means of features
        self._means: np.ndarray | None = None

    def fit(self, X):
        """
        Learn the reduction from training data.

        Args:
            X: Training data, shape (n_samples, n_features)

        Returns:
            self
        """
        n, d = X.shape
        self._means = np.mean(X, axis=0)

        X_centered = X - self._means
        cov = 1 / (n - 1) * (X_centered.T @ X_centered)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        self._p = eigenvectors[:, -self.n_components:][:, ::-1]

        return self

    def transform(self, X):
        """
        Apply the learned reduction to data.

        Args:
            X: Data to transform, shape (n_samples, n_features)

        Returns:
            X_reduced: shape (n_samples, 10)
        """
        return (X - self._means) @ self._p

    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


np.random.seed(42)
X = np.random.randn(100, 50)
reducer = MyReducer()
reducer.fit(X)
X_t = reducer.transform(X)
print(X_t.shape == (100, 10))