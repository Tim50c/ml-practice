import numpy as np


def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.

    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class

    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    y = np.asarray(y_true, dtype=int)
    y_hat = np.asarray(y_scores, dtype=float)

    thresh = np.sort(np.unique(y_hat))[::-1]
    p = np.sum(y)
    n = len(y) - p

    fpr = np.zeros((len(thresh) + 1,))
    tpr = np.zeros((len(thresh) + 1,))

    for i, t in enumerate(thresh):
        y_t = np.where(y_hat >= t, 1, 0)

        tp = np.sum((y_t == y)[y_t == 1])
        fp = np.sum((y_t != y)[y_t == 1])

        fpr[i + 1] = fp / n
        tpr[i + 1] = tp / p

    return fpr.tolist(), tpr.tolist()


fpr, tpr = compute_roc_curve([0, 0, 1, 1], [0.5, 0.5, 0.9, 0.9])
print((fpr, tpr))
