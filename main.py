import numpy as np
import interpolacao
import polinomios

print("\n> Executando o método de Lagrange:")
x = np.array([2, 11 / 4, 4])
y = np.array([1 / 2, 4 / 11, 1 / 4])
x_int = 3
print("x:", x)
print("y:", y)
print("x_int:", x_int)
[y_int] = interpolacao.lagrange(x, y, x_int)
print("y_int:", y_int)

print("\n> Executando o método de Neville:")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
x_int = 1.5
print("x:", x)
print("y:", y)
print("x_int:", x_int)
[y_int, q] = interpolacao.neville(x, y, x_int)
print("y_int:", y_int)
print("q:\n", q)

print("\n> Executando o método de Briot-Ruffini:");
a = np.array([2, 0, -3, 3, -4])
raiz = -2;
print("a:", a)
print("raiz:", raiz)
[b, resto] = polinomios.briot_ruffini(a, raiz)
print("b:", b)
print("resto:", resto)

print("\n> Executando o método de Diferenças Divididas de Newton")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
print("x:", x)
print("y:", y)
[f] = polinomios.diferencas_divididas_newton(x, y)
print("f:", f)
