import math
import numpy as np


def backward_substitution(u, d):
    '''
    Solve the upper linear system ux=d
    Inputs:
            u: Upper triangular matrix
            d: Array containing d values
    Outputs:
            x: Solution of linear system
    '''

    [n, m] = u.shape
    b = d.astype(float)

    if n != m:
        raise "Error: 'u' must be a square matrix."

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if u[i, i] == 0:
            raise "Error: Matrix 'u' is singular."

        x[i] = b[i] / u[i, i]
        b[0:i] = b[0:i] - u[0:i, i] * x[i]

    return [x]


def forward_substitution(l, c):
    '''
    Solve the lower linear system lx=c
    Inputs:
            l: Lower triangular matrix
            c: Array containing c values
    Outputs:
            x: Solution of linear system
    '''

    [n, m] = l.shape
    b = c.astype(float)

    if n != m:
        raise "Error: 'l' must be a square matrix."

    x = np.zeros(n)

    for i in range(0, n):
        if l[i, i] == 0:
            raise "Error: Matrix 'l' is singular."

        x[i] = b[i] / l[i, i]
        b[i + 1:n] = b[i + 1:n] - l[i + 1:n, i] * x[i]

    return [x]


def gauss_elimination_pp(a, b):
    '''
    Gaussian Elimination with Partial Pivoting - Calculate the
    upper triangular matrix from linear system Ax=b (do a row reduction)
    Inputs:
            a: Matrix A from system Ax=b
            b: Array containing b values
    Outputs:
            a: Augmented upper triangular matrix
    '''

    [n, m] = a.shape

    if n != m:
        raise "Error: 'l' must be a square matrix."

    # Produces the augmented matrixsss
    a = np.concatenate((a, b[:, None]), axis=1).astype(float)

    # Elimination process starts
    for i in range(0, n - 1):
        p = i

        # Comparison to select the pivot
        for j in range(i + 1, n):
            if math.fabs(a[j, i]) > math.fabs(a[i, i]):
                # Swap rows
                a[[i, j]] = a[[j, i]]

        # Cheking for nullity of the pivots
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
