import numpy as np


def jacobi(a: np.array, b: np.array, x0: np.array, tol: float, iter_max: int) -> [np.array, float]:
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
    for iter in range(1, iter_max + 1):
        x = np.linalg.solve(d, (b - np.dot(m, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= tol:
            break
        x0 = x.copy()

    return [x, iter]


def gauss_seidel(a: np.array, b: np.array, x0: np.array, tol: float, iter_max: int) -> [np.array, float]:
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
    for iter in range(1, iter_max + 1):
        x = np.linalg.solve(l, (b - np.dot(u, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= tol:
            break
        x0 = x.copy()

    return [x, iter]
