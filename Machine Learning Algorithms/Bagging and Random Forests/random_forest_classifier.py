import numpy as np


def random_forest_fit_predict(X_train,
                              y_train,
                              X_test,
                              n_trees: int = 5,
                              max_depth: int = 3,
                              max_features: int = 2,
                              seed: int = 0):
	"""
	Train a random forest classifier and predict on X_test.
	Returns a list of integer class predictions.
	"""

	def _majority_vote(predictions):
		vals, counts = np.unique(predictions, return_counts=True)
		return vals[np.argmax(counts)]

	def find_best_split(X, y):
		n_S, n_features = X.shape
		features = rng.choice(n_features, max_features, replace=False)

		best_gini = np.inf
		best_feature = None
		best_threshold = None

		for j in features:
			ind = np.argsort(X[:, j], kind='stable')
			Xj = X[ind, j]
			yj = y[ind]

			vals = np.unique(Xj)
			thresholds = (vals[1:] + vals[:-1]) / 2
			split_indices = np.searchsorted(Xj, thresholds, side='right')

			for s in range(len(split_indices)):
				thresh = thresholds[s]

				n_l = split_indices[s]
				n_r = n_S - n_l

				if n_l == 0 or n_r == 0:
					continue

				_, counts_l = np.unique(yj[:n_l], return_counts=True)
				p_l = counts_l / n_l

				_, counts_r = np.unique(yj[n_l:], return_counts=True)
				p_r = counts_r / n_r

				g_l = 1.0 - np.sum(p_l**2)
				g_r = 1.0 - np.sum(p_r**2)

				gini = (n_l / n_S) * g_l + (n_r / n_S) * g_r

				if gini < best_gini:
					best_gini = gini
					best_feature = j
					best_threshold = thresh

		return (None, None) if np.isinf(best_gini) else (best_feature, best_threshold)

	def _build(X, y, depth):
		n_S = len(y)

		if (depth >= max_depth) or (n_S < 2) or len(np.unique(y)) == 1:
			return {
				'leaf': True,
				'value': _majority_vote(y),
			}

		feature, threshold = find_best_split(X, y)

		if feature is None or threshold is None:
			return {
				'leaf': True,
				'value': _majority_vote(y),
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


	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	X_test = np.asarray(X_test)

	rng = np.random.default_rng(seed)

	n_samples, n_features = X_train.shape
	trees = []

	for _ in range(n_trees):
		indices = rng.integers(0, n_samples, size=n_samples)

		X_boot = X_train[indices]
		y_boot = y_train[indices]

		tree = _build(X_boot, y_boot, 0)
		trees.append(tree)

	n_test = X_test.shape[0]
	tree_predictions = np.array([[_predict_one(tree, x) for x in X_test] for tree in trees])

	# Final majority vote
	y_pred = np.zeros((n_test,), dtype=y_train.dtype)

	for i in range(n_test):
		y_pred[i] = _majority_vote(tree_predictions[:, i])

	return y_pred.tolist()

X_train = [[0,0],[0,1],[5,5],[5,6],[10,0],[10,1]]
y_train = [0,0,1,1,2,2]
X_test = [[0.2,0.3],[5.2,5.4],[9.8,0.2]]
print(random_forest_fit_predict(X_train, y_train, X_test, n_trees=11, max_depth=4, max_features=2, seed=7))
