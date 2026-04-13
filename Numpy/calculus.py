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
    return c*n*x**(n-1)

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


print(softmax_derivative([1.0, 2.0, 3.0]))