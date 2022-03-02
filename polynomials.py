"""Methods for polynomials."""


import numpy as np


def briot_ruffini(a, root):
    """Divides a polynomial by another polynomial.

    The format is: P(x) = Q(x) * (x-root) + rest.

    Args:
        a: array containing the coefficients of the input polynomial.
        root: one of the polynomial roots.

    Returns:
        b: array containing the coefficients of the output polynomial.
        rest: polynomial division Rest.
    """
    n = a.size - 1
    b = np.zeros(n)

    b[0] = a[0]

    for i in range(1, n):
        b[i] = b[i - 1] * root + a[i]

    rest = b[n - 1] * root + a[n]

    return [b, rest]


def newton_divided_difference(x, y):
    """Find the coefficients of Newton's divided difference.

    Also findthe Newton's polynomial.

    Args:
        x: array containing x values.
        y: array containing y values.

    Returns:
        f: array containing Newton's divided difference coefficients.
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
    print("p(x)={:+.4f}".format(f[0]), end="")
    for i in range(1, n):
        print("{:+.4f}".format(f[i]), end="")
        for j in range(1, i + 1):
            print("(x{:+.4f})".format(x[j] * -1), end="")
    print("")

    return [f]
