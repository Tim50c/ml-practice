import numpy as np


def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    n_samples, n_features = X.shape

    classes, counts = np.unique(y, return_counts=True)

    n_test = np.sum((counts*test_size).astype(int))
    n_train = n_samples - n_test

    X_train = np.zeros((n_train, n_features))
    X_test = np.zeros((n_test, n_features))
    y_train = np.zeros((n_train,))
    y_test = np.zeros((n_test,))

    ind_train = 0
    ind_test = 0

    for c in classes:
        indices = np.flatnonzero(y == c)
        nc_test = int(len(indices)*test_size)
        nc_train = len(indices) - nc_test

        np.random.shuffle(indices)

        X_test[ind_test:ind_test + nc_test] = X[indices[0:nc_test]]
        y_test[ind_test:ind_test + nc_test] = y[indices[0:nc_test]]

        ind_test += nc_test

        X_train[ind_train:ind_train + nc_train] = X[indices[0:nc_train]]
        y_train[ind_train:ind_train + nc_train] = y[indices[0:nc_train]]

        ind_train += nc_train

    return X_train, X_test, y_train, y_test



X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
y = np.array([0, 0, 0, 1, 1, 1])
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, 0.5, random_seed=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
