import numpy as np


def lagrange(x: np.array, y: np.array, x_int: float) -> [float]:
    """
    Interpola um valor usando polinômio de Lagrange
    Parâmetros de entrada:
            x: Vetor contendo as abscissas
            y: Vetor contendo as ordenadas
        x_int: Valor a interpolar
    Parâmetros de saída:
        y_int: Valor interpolado
    """

    n = x.size
    y_int = 0

    for i in range(0, n):
        p = y[i]
        for j in range(0, n):
            if i != j:
                p = p * (x_int - x[j]) / (x[i] - x[j])
        y_int = y_int + p

    return [y_int]


def neville(x: np.array, y: np.array, x_int: float) -> [float, np.array]:
    """
    Interpola um valor usando polinômio de Neville
    Parâmetros de entrada:
            x: Vetor contendo as abscissas
            y: Vetor contendo as ordenadas
        x_int: Valor a interpolar
    Parâmetros de saída:
        y_int: Valor interpolado
            q: Matriz de coeficientes
    """

    n = x.size
    q = np.zeros((n, n - 1))
    q = np.concatenate((y[:, None], q), axis=1)  # Insere y na primeira coluna da matriz q

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = ((x_int - x[i - j]) * q[i, j - 1] - (x_int - x[i]) * q[i - 1, j - 1]) / (x[i] - x[i - j])

    y_int = q[n - 1, n - 1]
    return [y_int, q]
