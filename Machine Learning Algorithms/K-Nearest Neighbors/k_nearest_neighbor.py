import numpy as np


def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point

    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return

    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points = np.asarray(points)
    query_point = np.asarray(query_point)

    distances = np.linalg.norm(points - query_point, axis=1)
    indices = np.argsort(distances, kind='stable')

    k_nearest = points[indices[0:k]]

    return [tuple(point) for point in k_nearest]

print(k_nearest_neighbors([(1, 2), (3, 4), (1, 1), (5, 6), (2, 3)], (2, 2), 3))
