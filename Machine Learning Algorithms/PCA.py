import numpy as np


class CovariancePCA:
    """
    Principal Component Analysis (PCA) using covariance matrix and eigendecomposition.
    """

    def __init__(self, n_components: int = 10):
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")

        self.n_components = n_components

        # Learned attributes (suffixed with _ to denote they are learned during fit)
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Learn the reduction from training data.

        Args:
            X: Training data, shape (n_samples, n_features)

        Returns:
            self
        """
        n_samples, n_features = X.shape

        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"number of features ({n_features})."
            )

        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute the covariance matrix
        # Using np.cov is standard, but keeping your manual dot product
        # approach is perfectly fine and often slightly faster.
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # 3. Eigendecomposition
        # eigh is highly optimized for symmetric/Hermitian matrices
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Store the top n_components
        self.components_ = sorted_eigenvectors[:, :self.n_components]

        # 6. Calculate explained variance metrics
        self.explained_variance_ = sorted_eigenvalues[:self.n_components]
        total_variance = np.sum(sorted_eigenvalues)

        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the learned reduction to data.

        Args:
            X: Data to transform, shape (n_samples, n_features)

        Returns:
            X_reduced: shape (n_samples, n_components)
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("This PCA instance is not fitted yet. Call 'fit' first.")

        X_centered = X - self.mean_
        return X_centered @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and apply the dimensionality reduction on X."""
        return self.fit(X).transform(X)