def apply_weight_decay(parameters: list[list[float]],
                       gradients: list[list[float]],
                       lr: float,
                       weight_decay: float,
                       apply_to_all: list[bool]) -> list[list[float]]:
    """
    Apply weight decay (L2 regularization) to parameters.

    Args:
        parameters: List of parameter arrays
        gradients: List of gradient arrays
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay

    Returns:
        Updated parameters
    """
    groups = len(parameters)
    n = len(parameters[0])

    for i in range(groups):
        for j in range(n):
            if apply_to_all[i]:
                parameters[i][j] = parameters[i][j] - lr * weight_decay * parameters[i][j]

            parameters[i][j] = parameters[i][j] - lr * gradients[i][j]

    return parameters


print([[round(x, 4) for x in arr] for arr in apply_weight_decay([[1.0, 2.0, 3.0]], [[0.1, 0.2, 0.3]], 0.1, 0.01, [True])])