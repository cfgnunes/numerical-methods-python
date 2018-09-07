import numpy as np


def jacobi(a, b, x0, tol, iter_max):
    '''
    Jacobi method: solve Ax = b given an initial approximation x0
    Inputs:
            a: Matrix A from system Ax=b
            b: Array containing b values
           x0: Initial approximation of solution
          tol: Tolerance
     iter_max: Maximum number of iterations
    Outpus:
            x: Solution of linear system
         iter: Used iterations
    '''

    # D and M matrices
    d = np.diag(np.diag(a))
    m = a - d

    # Iterative process
    i = 1
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(d, (b - np.dot(m, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= tol:
            break
        x0 = x.copy()

    return [x, i]


def gauss_seidel(a, b, x0, tol, iter_max):
    '''
    Gauss-Seidel method: solve Ax = b given an initial approximation x0
    Inputs:
            a: Matrix A from system Ax=b
            b: Array containing b values
           x0: Initial approximation of solution
          tol: Tolerance
     iter_max: Maximum number of iterations
    Outpus:
            x: Solution of linear system
         iter: Used iterations
    '''

    # L and U matrices
    l = np.tril(a)
    u = a - l

    # Iterative process
    i = 1
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(l, (b - np.dot(u, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= tol:
            break
        x0 = x.copy()

    return [x, i]
