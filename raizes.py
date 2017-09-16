import numpy as np
import math


def bissecao(funcao, a: float, b: float, tol: float, iter_max: int) -> [float, int, bool]:
    """
    Calcula a raiz de uma equação pelo método da bisseção
    Parâmetros de entrada:
            a: Limite inferior
            b: Limite superior
          tol: Tolerância
      iterMax: Número máximo de iterações
    Parâmetros de saída:
         raiz: Raiz
         iter: Número de iterações
    cond_erro: Condição de erro
        cond_erro = 0 se a raiz foi encontrada
        cond_erro = 1 em caso contrário
    """

    # Avaliar a função em a
    Fa = funcao(a)

    # Avaliar a função em b
    Fb = funcao(b)

    if Fa * Fb > 0:
        raise("Erro: A função não muda de sinal nos extremos do intervalo dado.")

    deltaX = abs(b - a) / 2

    iter = 0
    x = 0
    cond_erro = True
    for iter in range(0, iter_max + 1):
        x = (a + b) / 2

        # Avaliar a função em x
        Fx = funcao(x)
        print('iter: %.3d\t a: %.4f\t Fa: %.4f\t b: %.4f\t Fb: %.4f\t x: %.4f\t Fx: %.4f\t deltaX: %.4f\n' % (iter, a, Fa, b, Fb, x, Fx, deltaX), end="")

        if deltaX <= tol and math.fabs(Fx) <= tol:
            cond_erro = False
            break

        if (Fa * Fx > 0):
            a = x
            Fa = Fx
        else:
            b = x

        deltaX = deltaX / 2
    else:
        print("Aviso: O método não convergiu.")

    raiz = x

    return [raiz, iter, cond_erro]
