"""Methods for Linear Systems."""

import numpy as np


def jacobi(a, b, x0, toler, iter_max):
    """Jacobi method: solve Ax = b given an initial approximation x0.

    Args:
        a: matrix A from system Ax=b.
        b: an array containing b values.
        x0: initial approximation of the solution.
        toler: tolerance (stopping criterion).
        iter_max: maximum number of iterations (stopping criterion).

    Returns:
        x: solution of linear the system.
        iter: number of iterations used by the method.
    """
    # D and M matrices
    d = np.diag(np.diag(a))
    m = a - d

    # Iterative process
    x = None
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(d, (b - np.dot(m, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= toler:
            break
        x0 = x.copy()

    return [x, i]


def gauss_seidel(a, b, x0, toler, iter_max):
    """Gauss-Seidel method: solve Ax = b given an initial approximation x0.

    Args:
        a: matrix A from system Ax=b.
        b: an array containing b values.
        x0: initial approximation of the solution.
        toler: tolerance (stopping criterion).
        iter_max: maximum number of iterations (stopping criterion).

    Returns:
        x: solution of linear the system.
        iter: number of iterations used by the method.
    """
    # L and U matrices
    lower = np.tril(a)
    upper = a - lower

    # Iterative process
    x = None
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(lower, (b - np.dot(upper, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= toler:
            break
        x0 = x.copy()

    return [x, i]
