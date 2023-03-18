"""Iterative Methods for Linear Systems."""

import math

import numpy as np


def backward_substitution(upper, d):
    """Solve the upper linear system ux=d.

    Args:
        upper: upper triangular matrix.
        d: an array containing d values.

    Returns:
        x: solution of linear the system.
    """
    [n, m] = upper.shape
    b = d.astype(float)

    if n != m:
        raise ValueError("'upper' must be a square matrix.")

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if upper[i, i] == 0:
            raise ValueError("Matrix 'upper' is singular.")

        x[i] = b[i] / upper[i, i]
        b[0:i] = b[0:i] - upper[0:i, i] * x[i]

    return [x]


def forward_substitution(lower, c):
    """Solve the lower linear system lx=c.

    Args:
        lower: lower triangular matrix.
        c: an array containing c values.

    Returns:
        x: solution of linear the system.
    """
    [n, m] = lower.shape
    b = c.astype(float)

    if n != m:
        raise ValueError("'lower' must be a square matrix.")

    x = np.zeros(n)

    for i in range(0, n):
        if lower[i, i] == 0:
            raise ValueError("Matrix 'lower' is singular.")

        x[i] = b[i] / lower[i, i]
        b[i + 1:n] = b[i + 1:n] - lower[i + 1:n, i] * x[i]

    return [x]


def gauss_elimination_pp(a, b):
    """Gaussian Elimination with Partial Pivoting.

    Calculate the upper triangular matrix from linear system Ax=b (make a row
    reduction).

    Args:
        a: matrix A from system Ax=b.
        b: an array containing b values.

    Returns:
        a: augmented upper triangular matrix.
    """
    [n, m] = a.shape

    if n != m:
        raise ValueError("'a' must be a square matrix.")

    # Produces the augmented matrix
    a = np.concatenate((a, b[:, None]), axis=1).astype(float)

    # Elimination process starts
    for i in range(0, n - 1):
        p = i

        # Comparison to select the pivot
        for j in range(i + 1, n):
            if math.fabs(a[j, i]) > math.fabs(a[i, i]):
                # Swap rows
                a[[i, j]] = a[[j, i]]

        # Checking for nullity of the pivots
        while p < n and a[p, i] == 0:
            p += 1

        if p == n:
            print("Info: No unique solution.")
        else:
            if p != i:
                # Swap rows
                a[[i, p]] = a[[p, i]]

        for j in range(i + 1, n):
            a[j, :] = a[j, :] - a[i, :] * (a[j, i] / a[i, i])

    # Checking for nonzero of last entry
    if a[n - 1, n - 1] == 0:
        print("Info: No unique solution.")

    return [a]
