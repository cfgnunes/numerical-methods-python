import numpy as np
import math
import interpolacao
import polinomios
import raizes

def debug(variable):
    print (variable, "=", eval(variable))

# Método da Bisseção (encontrar raízes de uma equação)
#   Vantagens:
#       É um método confiável com convergência garantida;
#       É um método simples que faz a busca da raíz por meio de uma busca binária;
#       Não há necessidade de calcular a derivada da função.
#   Desvantagens:
#       Convergência lenta;
#       É necessário informar um intervalo de busca [a, b];
#       O intervalo informado tem que possuir uma troca de sinal, f(a)*f(b)<0.
print("\n> Executando o método da Bisseção:")
funcao = lambda x: 4 * x ** 3 + x + math.cos(x) - 10
tol = 10 ** -5
iter_max = 100
a = 1.0
b = 2.0
debug("tol")
debug("iter_max")
debug("a")
debug("b")
[raiz, iter, cond_erro] = raizes.bissecao(funcao, a, b, tol, iter_max)
debug("raiz")
debug("iter")
debug("cond_erro")

# Método da Secante (encontrar raízes de uma equação)
#   Vantagens:
#       É um método rápido (mais lento que o método de Newton);
#       É um método baseado no método de Newton, mas não utiliza a derivada da função.
#   Desvantagem:
#       Pode divergir se a função não for aproximadamente linear no intervalo que contém a raiz;
#       É necessário informar dois pontos 'a' e 'b' em que f(a)-f(b) tem que ser diferente de zero.
print("\n> Executando o método da Secante:")
funcao = lambda x: 4 * x ** 3 + x + math.cos(x) - 10
tol = 10 ** -5
iter_max = 100
a = 1.0
b = 2.0
debug("tol")
debug("iter_max")
debug("a")
debug("b")
[raiz, iter, cond_erro] = raizes.secante(funcao, a, b, tol, iter_max)
debug("raiz")
debug("iter")
debug("cond_erro")

print("\n> Executando o método de Lagrange:")
x = np.array([2, 11 / 4, 4])
y = np.array([1 / 2, 4 / 11, 1 / 4])
x_int = 3
debug("x")
debug("y")
debug("x_int")
[y_int] = interpolacao.lagrange(x, y, x_int)
debug("y_int")

print("\n> Executando o método de Neville:")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
x_int = 1.5
debug("x")
debug("y")
debug("x_int")
[y_int, q] = interpolacao.neville(x, y, x_int)
debug("y_int")
debug("q")

print("\n> Executando o método de Briot-Ruffini:")
a = np.array([2, 0, -3, 3, -4])
raiz = -2
debug("a")
debug("raiz")
[b, resto] = polinomios.briot_ruffini(a, raiz)
debug("b")
debug("resto")

print("\n> Executando o método de Diferenças Divididas de Newton")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
debug("x")
debug("y")
[f] = polinomios.diferencas_divididas_newton(x, y)
debug("f")
