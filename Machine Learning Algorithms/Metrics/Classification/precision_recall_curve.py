import numpy as np


def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
	"""
	Compute precision-recall pairs for different probability thresholds.

	Args:
		y_true: List of true binary labels (0 or 1)
		y_scores: List of predicted probabilities or confidence scores

	Returns:
		Tuple of (precisions, recalls, thresholds) where each is a list
	"""
	y = np.asarray(y_true, dtype=int)
	y_hat = np.asarray(y_scores, dtype=float)

	order = np.argsort(y_hat)[::-1]
	y_sorted = y[order]
	y_hat_sorted = y_hat[order]

	cum_pos = np.cumsum(y_sorted)
	pos_true = cum_pos[-1]

	thresholds = np.unique(y_hat_sorted)[::-1]
	l = len(thresholds)

	indices = np.searchsorted(-y_hat_sorted, -thresholds, side='right')

	precisions = np.zeros(l, dtype=float)
	recalls = np.zeros(l, dtype=float)

	for i, idx in enumerate(indices):
		pos_pred = idx
		tp = cum_pos[idx - 1]

		precisions[i] = tp / pos_pred if pos_pred != 0 else 1.0
		recalls[i] = tp / pos_true if pos_true != 0 else 0.0

	return precisions.tolist(), recalls.tolist(), thresholds.tolist()

y_true = [1, 0, 1, 0, 1]
y_scores = [0.9, 0.8, 0.6, 0.4, 0.2]
p, r, t = precision_recall_curve(y_true, y_scores)
print([round(x, 4) for x in p], [round(x, 4) for x in r], t)