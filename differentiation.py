"""Numerical differentiation."""

import numpy as np


def backward_difference(x, y):
    """Calculate the first derivative.

    Uses a forward difference at the first point and backward differences
    at the remaining points. Spacing in 'x' need not be uniform.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        dy (numpy.ndarray): the first derivative values.
    """
    if x.size < 2 or y.size < 2:
        raise ValueError("'x' and 'y' arrays must have 2 values or more.")

    if x.size != y.size:
        raise ValueError("'x' and 'y' must have same size.")

    def dy_difference(h, y0, y1):
        return (y1 - y0) / h

    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i == 0:
            hx = x[i + 1] - x[i]
            dy[i] = dy_difference(hx, y[i], y[i + 1])
        else:
            hx = x[i] - x[i - 1]
            dy[i] = dy_difference(hx, y[i - 1], y[i])

    return dy


def three_point(x, y):
    """Calculate the first derivative.

    All values in 'x' must be equally spaced.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        dy (numpy.ndarray): the first derivative values.
    """
    if x.size < 3 or y.size < 3:
        raise ValueError("'x' and 'y' arrays must have 3 values or more.")

    if x.size != y.size:
        raise ValueError("'x' and 'y' must have same size.")

    def dy_mid(h, y0, y2):
        return (1 / (2 * h)) * (y2 - y0)

    def dy_end(h, y0, y1, y2):
        return (1 / (2 * h)) * (-3 * y0 + 4 * y1 - y2)

    hx = x[1] - x[0]
    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i == 0:
            dy[i] = dy_end(hx, y[i], y[i + 1], y[i + 2])
        elif i == n - 1:
            dy[i] = dy_end(-hx, y[i], y[i - 1], y[i - 2])
        else:
            dy[i] = dy_mid(hx, y[i - 1], y[i + 1])

    return dy


def five_point(x, y):
    """Calculate the first derivative.

    All values in 'x' must be equally spaced.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        dy (numpy.ndarray): the first derivative values.
    """
    if x.size < 5 or y.size < 5:
        raise ValueError("'x' and 'y' arrays must have 5 values or more.")

    if x.size != y.size:
        raise ValueError("'x' and 'y' must have same size.")

    def dy_mid(h, y0, y1, y3, y4):
        return (1 / (12 * h)) * (y0 - 8 * y1 + 8 * y3 - y4)

    def dy_end(h, y0, y1, y2, y3, y4):
        return (1 / (12 * h)) * \
            (-25 * y0 + 48 * y1 - 36 * y2 + 16 * y3 - 3 * y4)

    def dy_near(h, y0, y1, y2, y3, y4):
        return (1 / (12 * h)) * \
            (-3 * y0 - 10 * y1 + 18 * y2 - 6 * y3 + y4)

    hx = x[1] - x[0]
    n = x.size
    dy = np.zeros(n)
    for i in range(0, n):
        if i == 0:
            dy[i] = dy_end(hx, y[i], y[i + 1], y[i + 2], y[i + 3], y[i + 4])
        elif i == 1:
            dy[i] = dy_near(hx, y[i - 1], y[i], y[i + 1], y[i + 2], y[i + 3])
        elif i == n - 1:
            dy[i] = dy_end(-hx, y[i], y[i - 1], y[i - 2], y[i - 3], y[i - 4])
        elif i == n - 2:
            dy[i] = dy_near(-hx, y[i + 1], y[i], y[i - 1], y[i - 2], y[i - 3])
        else:
            dy[i] = dy_mid(hx, y[i - 2], y[i - 1], y[i + 1], y[i + 2])

    return dy
