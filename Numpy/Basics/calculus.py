import numpy as np


def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.

    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative

    Returns:
        The derivative value f'(x)
    """
    # Your code here
    g = np.asarray(g_coeffs, dtype=float)
    h = np.asarray(h_coeffs, dtype=float)

    g_derivatives = np.polyder(g) if len(g) > 1 else np.array([0.0])
    h_derivatives = np.polyder(h) if len(h) > 1 else np.array([0.0])

    g_val = np.polyval(g, x)
    h_val = np.polyval(h, x)
    g_der_val = np.polyval(g_derivatives, x)
    h_der_val = np.polyval(h_derivatives, x)

    return (g_der_val * h_val - g_val * h_der_val) / (h_val ** 2)


def poly_term_derivative(c: float, x: float, n: float) -> float:
    if x == 0.0 or n == 0.0:
        return 0.0
    return c * n * x ** (n - 1)


def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.

    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences

    Returns:
        Jacobian matrix as list of lists
    """
    fx = np.asarray(f(x))
    m = fx.size

    x = np.asarray(x, dtype=float)
    n = x.size

    J = np.zeros((n, m), dtype=float)

    for i in range(n):
        x_plus_h = x.copy()
        x_minus_h = x.copy()

        x_plus_h[i] += h
        x_minus_h[i] -= h

        f_plus = np.asarray(f(x_plus_h))
        f_minus = np.asarray(f(x_minus_h))

        J[:, i] = (f_plus - f_minus) / (2 * h)

    return J.tolist()


def softmax_derivative(x: list[float]):
    """
    Compute the Jacobian matrix of the softmax function.

    Args:
        x: Input vector of real numbers

    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    x = np.asarray(x, dtype=float)
    power = x - np.max(x)
    sum_exp = np.sum(np.exp(power))
    s = np.exp(power) / sum_exp

    return s


import numpy as np


def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.

    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')

    Returns:
        Optimized weights
    """
    if method == 'batch':
        for i in range(n_epochs):
            e = X @ weights - y
            weights -= learning_rate * (2 / len(y)) * (X.T @ e)

    if method == 'stochastic':
        for i in range(n_epochs):
            for j in range(len(y)):
                e = X[j, :] @ weights - y[j]
                weights -= learning_rate * 2 * e * X[j]

    if method == 'mini_batch':
        for i in range(n_epochs):
            batches = int(len(y) / batch_size)
            for j in range(batches):
                begin = j * batch_size
                end = j * (batch_size + 1)
                e = X[begin:end, :] @ weights - y[begin:end]
                weights -= learning_rate * (2 / batch_size) * (X[begin:end, :].T @ e)

    return weights


def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

    Returns:
        float: The x-coordinate of the minimum point.
    """
    tolerance = 1e-6
    f = lambda t: t ** 4 - 3 * t ** 3 + 2
    f_prime = lambda t: 4 * t ** 3 - 9 * t ** 2
    x = start_x
    eta0 = 0.1

    for i in range(1000):
        eta = eta0 * pow(i + 1.0, -0.5)
        x_new = x - eta * f_prime(x)
        if f(x) - f(x_new) <= tolerance:
            break

        x = x_new

    return x


r = find_treasure(-1)
print(r)
print(abs(r - 2.25) < 1e-2)
