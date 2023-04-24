"""Methods for polynomials."""

import math

import numpy as np


def briot_ruffini(a, root):
    """Divide a polynomial by another polynomial.

    The format is: P(x) = Q(x) * (x-root) + rest.

    Args:
        a (numpy.ndarray): the coefficients of the input polynomial.
        root (float): one of the polynomial roots.

    Returns:
        b (numpy.ndarray): the coefficients of the output polynomial.
        rest (float): polynomial division Rest.
    """
    n = a.size - 1
    b = np.zeros(n)

    b[0] = a[0]

    for i in range(1, n):
        b[i] = b[i - 1] * root + a[i]

    rest = b[n - 1] * root + a[n]

    return b, rest


def newton_divided_difference(x, y):
    """Find the coefficients of Newton's divided difference.

    Also, find Newton's polynomial.

    Args:
        x (numpy.ndarray): x values.
        y (numpy.ndarray): y values.

    Returns:
        f (numpy.ndarray): Newton's divided difference coefficients.
    """
    n = x.size
    q = np.zeros((n, n - 1))

    # Insert 'y' in the first column of the matrix 'q'
    q = np.concatenate((y[:, None], q), axis=1)

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = (q[i, j - 1] - q[i - 1, j - 1]) / (x[i] - x[i - j])

    # Copy the diagonal values of the matrix q to the vector f
    f = np.zeros(n)
    for i in range(0, n):
        f[i] = q[i, i]

    # Prints the polynomial
    print("The polynomial is:")
    print(f"p(x)={f[0]:+.3f}", end="")
    for i in range(1, n):
        print(f"{f[i]:+.3f}", end="")
        for j in range(1, i + 1):
            print(f"(x{(x[j] * -1):+.3f})", end="")
    print("")

    return f


def root_limits(c):
    """Find the limits of the real roots of a polynomial equation.

    Using Lagrange's Theorem, whose proof is given by Demidovich and Maron.

    Args:
        c (numpy.ndarray): polynomial coefficients.

    Returns:
        lim (numpy.ndarray): lower and upper limits of positive and
            negative roots, respectively.
    """
    lim = np.zeros(4)
    n = len(c) - 1
    c = np.concatenate(([0], c))
    c = np.concatenate((c, [0]))

    if c[1] == 0:
        raise ValueError("The first coefficient is null.")

    t = n + 1
    c[t + 1] = 0

    # If c[t+1] is null, then the polynomial is deflated.
    while True:
        if c[t] != 0:
            break
        t -= 1

    # Compute the four limits of real roots.
    for i in range(0, 4):
        if i in (1, 3):
            # Inversion of the order of the coefficients.
            for j in range(1, t // 2 + 1):
                c[j], c[t - j + 1] = c[t - j + 1], c[j]
        else:
            if i == 2:
                # Reinversion of the order and exchange
                # of signs of the coefficients.
                for j in range(1, t // 2 + 1):
                    c[j], c[t - j + 1] = c[t - j + 1], c[j]
                for j in range(t - 1, 0, -2):
                    c[j] = -c[j]

        # If c[1] is negative, then all coefficients are swapped.
        if c[1] < 0:
            for j in range(1, t + 1):
                c[j] = -c[j]

        # Calculation of 'k', the largest index of the negative coefficients.
        k = 2
        while True:
            if c[k] < 0 or k > t:
                break
            k += 1

        # Calculation of 'b', the largest negative coefficient in modulus.
        if k <= t:
            b = 0
            for j in range(2, t + 1):
                if c[j] < 0 and math.fabs(c[j]) > b:
                    b = math.fabs(c[j])

            # Limit of positive roots of 'P(x) = 0' and auxiliary equations.
            lim[i] = 1 + (b / c[1]) ** (1 / (k - 1))
        else:
            lim[i] = 10 ** 100

    # Limit of positive and negative roots of 'P(x) = 0'.
    lim[0], lim[1], lim[2], lim[3] = 1 / lim[1], lim[0], -lim[2], -1 / lim[3]

    return lim
