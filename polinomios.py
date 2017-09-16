import numpy as np


def briot_ruffini(a: np.array, raiz: float) -> [np.array, float]:
    """
    Divide um polinômio por outro ponilômio no formato (x-raiz)
    P(x) = Q(x) * (x-raiz) + resto
    Parâmetros de entrada:
            a: Vetor que contém os coeficientes do polinômio de entrada
         raiz: Uma das raízes do polinômio
    Parâmetros de saída:
            b: Vetor que contém os coeficientes do polinômio de saída
        resto: Resto da divisão do polinômio
    """

    n = a.size - 1
    b = np.zeros(n)

    b[0] = a[0]

    for i in range(1, n):
        b[i] = b[i - 1] * raiz + a[i]

    resto = b[n - 1] * raiz + a[n]

    return [b, resto]


def diferencas_divididas_newton(x: np.array, y: np.array) -> [np.array]:
    """
    Encontra os coeficientes da diferença dividida de Newton
    Parâmetros de entrada:
            x: Vetor contendo as abscissas
            y: Vetor contendo as ordenadas
    Parâmetros de saída:
            f: Vetor contendo os coeficientes da diferença dividida de Newton
    """

    n = x.size
    q = np.zeros((n, n - 1))
    q = np.concatenate((y[:, None], q), axis=1)  # Insere y na primeira coluna da matriz q

    for i in range(1, n):
        for j in range(1, i + 1):
            q[i, j] = (q[i, j - 1] - q[i - 1, j - 1]) / (x[i] - x[i - j])

    # Copia os valores da diagonal da matriz q para o vetor f
    f = np.zeros(n)
    for i in range(0, n):
        f[i] = q[i, i]

    # Imprime o polinômio
    print("Polinômio resultante:")
    print("p(x)=%+.4f" % f[0], end="")
    for i in range(1, n):
        print("%+.4f" % f[i], end="")
        for j in range(1, i + 1):
            print("(x%+.4f)" % (x[j] * -1), end="")

    print("")

    return [f]
