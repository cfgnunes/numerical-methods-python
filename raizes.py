import math


def bissecao(funcao, a: float, b: float, tol: float, iter_max: int) -> [float, int, bool]:
    """
    Calcula a raiz de uma equação pelo método da bisseção
    Parâmetros de entrada:
            a: Limite inferior
            b: Limite superior
          tol: Tolerância
     iter_max: Número máximo de iterações
    Parâmetros de saída:
         raiz: Raiz
         iter: Número de iterações utilizadas
    cond_erro: Condição de erro
        cond_erro = 0, se a raíz foi encontrada
        cond_erro = 1, caso contrário
    """

    # Avalia a função em a
    Fa = funcao(a)

    # Avalia a função em b
    Fb = funcao(b)

    if Fa * Fb > 0:
        raise ("Erro: A função não muda de sinal nos extremos do intervalo dado.")

    deltaX = math.fabs(b - a) / 2

    x = 0
    cond_erro = True
    iter = 0
    for iter in range(0, iter_max + 1):
        x = (a + b) / 2

        # Avalia a função em x
        Fx = funcao(x)
        print('iter: %.3d\t x: %+.4f\t Fx: %+.4f\t deltaX: %+.4f\n' % (iter, x, Fx, deltaX), end="")

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


def secante(funcao, a: float, b: float, tol: float, iter_max: int) -> [float, int, bool]:
    """
    Calcula a raiz de uma equação pelo método da secante
    Parâmetros de entrada:
            a: Limite inferior
            b: Limite superior
          tol: Tolerância
     iter_max: Número máximo de iterações
    Parâmetros de saída:
         raiz: Raiz
         iter: Número de iterações utilizadas
    cond_erro: Condição de erro
                  cond_erro = 0, se a raíz foi encontrada
                  cond_erro = 1, caso contrário
    """

    # Avalia a função em a
    Fa = funcao(a)

    # Avalia a função em b
    Fb = funcao(b)

    if math.fabs(Fa) < math.fabs(Fb):
        a, b = b, a
        Fa, Fb = Fb, Fa

    x = b
    Fx = Fb

    cond_erro = True
    iter = 0
    for iter in range(0, iter_max + 1):
        deltaX = -Fx / (Fb - Fa) * (b - a)
        x += deltaX

        # Avalia a função em x
        Fx = funcao(x)
        print("iter: %.3d\t x: %+.4f\t Fx: %+.4f\t deltaX: %+.4f\n" % (iter, x, Fx, deltaX), end="")
        if math.fabs(deltaX) <= tol and math.fabs(Fx) <= tol:
            cond_erro = False
            break

        a, b = b, x
        Fa, Fb = Fb, Fx

    else:
        print("Aviso: O método não convergiu.")

    raiz = x

    return [raiz, iter, cond_erro]
