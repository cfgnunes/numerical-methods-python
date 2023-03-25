"""Methods for Interpolation."""

import numpy as np


def lagrange(x, y, x_int):
    """Interpolates a value using the 'Lagrange polynomial'.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.
        x_int (float): value to interpolate.

    Returns:
        y_int (float): interpolated value.
    """
    m = x.size
    y_int = 0

    for i in range(0, m):
        p = y[i]
        for j in range(0, m):
            if i != j:
                p = p * (x_int - x[j]) / (x[i] - x[j])
        y_int = y_int + p

    return [y_int]


def newton(x, y, x_int):
    """Interpolates a value using the 'Newton polynomial'.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.
        x_int (float): value to interpolate.

    Returns:
        y_int (float): interpolated value.
    """
    m = x.size
    del_y = y.copy()

    # Calculate the divided differences
    for k in range(1, m):
        for i in range(m - 1, k - 1, -1):
            del_y[i] = (del_y[i] - del_y[i - 1]) / (x[i] - x[i - k])

    # Evaluate the Newton polynomial
    y_int = del_y[-1]
    for i in range(m - 2, -1, -1):
        y_int = y_int * (x_int - x[i]) + del_y[i]

    return [y_int]


def neville(x, y, x_int):
    """Interpolates a value using the 'Neville polynomial'.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.
        x_int (float): value to interpolate.

    Returns:
        y_int (float): interpolated value.
        q (numpy.ndarray): coefficients matrix.
    """
    n = x.size
    q = np.zeros((n, n - 1))

    # Insert 'y' in the first column of the matrix 'q'
    q = np.concatenate((y[:, None], q), axis=1)

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = ((x_int - x[i - j]) * q[i, j - 1] -
                       (x_int - x[i]) * q[i - 1, j - 1]) / (x[i] - x[i - j])

    y_int = q[n - 1, n - 1]
    return [y_int, q]
