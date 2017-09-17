import numpy as np


def briot_ruffini(a: np.array, raiz: float) -> [np.array, float]:
    """
    Divides a polynomial by another polynomial in the format (x-root)
    P(x) = Q(x) * (x-raiz) + rest
    Inputs:
            a: Vector that contains the coefficients of the input polynomial
         root: One of the polynomial roots
    Outpus:
            b: Vector containing the coefficients of the output polynomial
         rest: Polynomial division Rest
    """

    n = a.size - 1
    b = np.zeros(n)

    b[0] = a[0]

    for i in range(1, n):
        b[i] = b[i - 1] * raiz + a[i]

    rest = b[n - 1] * raiz + a[n]

    return [b, rest]


def newton_divided_difference(x: np.array, y: np.array) -> [np.array]:
    """
    Find the coefficients of Newton's divided difference and the Newton's polynomial
    Inputs:
            x: X values
            y: Y values
    Outpus:
            f: Vector containing Newton's divided difference coefficients
    """

    n = x.size
    q = np.zeros((n, n - 1))
    q = np.concatenate((y[:, None], q), axis=1)  # Insert 'y' in the first column of the matrix 'q'

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = (q[i, j - 1] - q[i - 1, j - 1]) / (x[i] - x[i - j])

    # Copy the diagonal values of the matrix q to the vector f
    f = np.zeros(n)
    for i in range(0, n):
        f[i] = q[i, i]

    # Prints the polynomial
    print("The polynomial is:")
    print("p(x)=%+.4f" % f[0], end="")
    for i in range(1, n):
        print("%+.4f" % f[i], end="")
        for j in range(1, i + 1):
            print("(x%+.4f)" % (x[j] * -1), end="")
    print("")

    return [f]
