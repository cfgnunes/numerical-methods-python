import numpy as np


def backward_substitution(u: np.array, d: np.array) -> [np.array]:
    '''
    Solve the upper linear system ux=d
    Inputs:
            u: Upper triangular matrix
            d: Array containing d values
    Outputs:
            x: Solution of linear system
    '''

    n = u.shape[0]
    m = u.shape[1]

    if n != m:
        raise ("Error: 'u' must be a square matrix.")

    x = np.zeros(n)

    x[n - 1] = d[n - 1] / u[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_x = 0
        for j in range(i + 1, n):
            sum_x += u[i, j] * x[j]

        x[i] = (d[i] - sum_x) / u[i, i]

    return [x]


def forward_substitution(l: np.array, c: np.array) -> [np.array]:
    '''
    Solve the lower linear system lx=c
    Inputs:
            l: Lower triangular matrix
            c: Array containing c values
    Outputs:
            x: Solution of linear system
    '''

    n = l.shape[0]
    m = l.shape[1]

    if n != m:
        raise ("Error: 'l' must be a square matrix.")

    x = np.zeros(n)

    x[0] = c[0] / l[0, 0]
    for i in range(1, n):
        sum_x = 0
        for j in range(0, i):
            sum_x += l[i, j] * x[j]

        x[i] = (c[i] - sum_x) / l[i, i]

    return [x]
