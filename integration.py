"""Methods for numerical integration."""

import numpy as np


def simpson(f, a, b, n):
    """Calculate the integral from 1/3 Simpson's Rule.

    Args:
        f (function): the equation f(x).
        a (float): the initial point.
        b (float): the final point.
        n (int): number of intervals.

    Returns:
        xi (float): numerical approximation of the definite integral.
    """
    h = (b - a) / n

    sum_odd = 0
    sum_even = 0

    for i in range(0, n - 1):
        x = a + (i + 1) * h
        if (i + 1) % 2 == 0:
            sum_even += f(x)
        else:
            sum_odd += f(x)

    xi = h / 3 * (f(a) + 2 * sum_even + 4 * sum_odd + f(b))
    return xi


def trapezoidal(f, a, b, n):
    """Calculate the integral from the Trapezoidal Rule.

    Args:
        f (function): the equation f(x).
        a (float): the initial point.
        b (float): the final point.
        n (int): number of intervals.

    Returns:
        xi (float): numerical approximation of the definite integral.
    """
    h = (b - a) / n

    sum_x = 0

    for i in range(0, n - 1):
        x = a + (i + 1) * h
        sum_x += f(x)

    xi = h / 2 * (f(a) + 2 * sum_x + f(b))
    return xi


def simpson_array(x, y):
    """Calculate the integral from 1/3 Simpson's Rule.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        xi (float): numerical approximation of the definite integral.
    """
    if y.size != y.size:
        raise ValueError("'x' and 'y' must have same size.")

    h = x[1] - x[0]
    n = x.size

    sum_odd = 0
    sum_even = 0

    for i in range(1, n - 1):
        if (i + 1) % 2 == 0:
            sum_even += y[i]
        else:
            sum_odd += y[i]

    xi = h / 3 * (y[0] + 2 * sum_even + 4 * sum_odd + y[n - 1])
    return xi


def trapezoidal_array(x, y):
    """Calculate the integral from the Trapezoidal Rule.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        xi (float): numerical approximation of the definite integral.
    """
    if y.size != y.size:
        raise ValueError("'x' and 'y' must have same size.")

    h = x[1] - x[0]
    n = x.size

    sum_x = 0

    for i in range(1, n - 1):
        sum_x += y[i]

    xi = h / 2 * (y[0] + 2 * sum_x + y[n - 1])
    return xi


def romberg(f, a, b, n):
    """Calculate the integral from the Romberg method.

    Args:
        f (function): the equation f(x).
        a (float): the initial point.
        b (float): the final point.
        n (int): number of intervals.

    Returns:
        xi (float): numerical approximation of the definite integral.
    """
    # Initialize the Romberg integration table
    r = np.zeros((n, n))

    # Compute the trapezoid rule for the first column (h = b - a)
    h = b - a
    r[0, 0] = 0.5 * h * (f(a) + f(b))

    # Iterate for each level of refinement
    for i in range(1, n):
        h = 0.5 * h  # Halve the step size
        # Compute the composite trapezoid rule
        sum_f = 0
        for j in range(1, 2**i, 2):
            x = a + j * h
            sum_f += f(x)
        r[i, 0] = 0.5 * r[i - 1, 0] + h * sum_f

        # Richardson extrapolation for higher order approximations
        for k in range(1, i + 1):
            r[i, k] = r[i, k - 1] + \
                (r[i, k - 1] - r[i - 1, k - 1]) / ((4**k) - 1)

    return float(r[n - 1, n - 1])
