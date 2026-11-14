import numpy as np


def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5, random_seed: int = 42) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider
        random_seed: Random seed for reproducibility

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    np.random.seed(random_seed)
    X_synthetic = np.zeros((n_synthetic, X_minority.shape[1]))
    for i in range(n_synthetic):
        ind = np.random.choice(np.arange(len(X_minority)))
        x_i = X_minority[ind]
        distances = np.argsort(np.sqrt(np.sum((X_minority - x_i) ** 2, axis=1)))
        knn = X_minority[distances[1:k + 1]]
        xnn = knn[np.random.choice(np.arange(len(knn)))]
        lam = np.random.uniform(0, 1)
        x_synthetic = x_i + lam * (xnn - x_i)
        X_synthetic[i] = x_synthetic

    return X_synthetic

X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]])
result = smote(X, 3, k=2, random_seed=42)
print(np.round(result, 4))