import numpy as np


def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.

    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j

    Returns:
        List of final predictions using majority vote
    """
    preds = np.asarray(predictions)
    vals = [np.unique(preds[:, i], return_counts=True) for i in range(preds.shape[1])]

    return [val[0][np.argmax(val[1])] for val in vals]

predictions = [[0, 1, 2], [0, 2, 2], [1, 1, 2]]
print(hard_voting_classifier(predictions))
