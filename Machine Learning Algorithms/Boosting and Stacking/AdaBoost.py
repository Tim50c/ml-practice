import numpy as np
import math

def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))
	clfs = []
	eps = 1e-10

	for _ in range(n_clf):
		best_polarity = 1
		best_threshold = None
		best_feature = None
		best_err = 1.0

		for j in range(n_features):
			indices = np.argsort(X[:, j], kind='stable')
			Xj = X[indices, j]
			yj = y[indices]
			wj = w[indices]

			thresholds = np.unique(Xj)
			split_indices = np.searchsorted(Xj, thresholds, side='right')

			for s in range(len(split_indices)):
				guesses = np.ones(n_samples, dtype=y.dtype)
				guesses[:split_indices[s]] = -1

				err = np.sum(wj*(guesses != yj))
				
				polarity = 1
				if err > 0.5:
					err = 1 - err
					polarity = -1

				if err < best_err:
					best_err = err
					best_threshold = thresholds[s]
					best_feature = j
					best_polarity = polarity


		alpha = 0.5*np.log((1 - best_err)/(best_err + eps))

		h = np.where(X[:, best_feature] <= best_threshold, -best_polarity, best_polarity)
		
		w = w*np.exp(-alpha*y*h)
		w = w / np.sum(w)

		clfs.append({
			'polarity': best_polarity,
			'threshold': best_threshold,
			'feature_index': best_feature,
			'alpha': alpha,
		})
		

	return clfs
    
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 1, -1, -1])
n_clf = 3
clfs = adaboost_fit(X, y, n_clf)
print(clfs)
