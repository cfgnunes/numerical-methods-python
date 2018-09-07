#!/usr/bin/env python3

"""
Numerical methods implementation in Python 3.
Author: Cristiano Nunes
E-mail: <cfgnunes@gmail.com>
Repository: https://github.com/cfgnunes/numerical-methods-python
"""

import math
import numpy as np

import differentiation
import integration
import interpolation
import linear_systems
import linear_systems_iterative
import ode
import polynomials
import solutions


def debug(variable):
    print(variable, "=", eval(variable))


def print_running(message):
    print("\n\n> Running", message)

# Bisection method (find roots of an equation)
#   Pros:
#       It is a reliable method with guaranteed convergence;
#       It is a simple method that does the search of the root by means of a
#           binary search;
#       There is no need to calculate the derivative of the function.
#   Cons:
#       Slow convergence;
#       It is necessary to enter a search interval [a, b];
#       The interval reported must have a signal exchange, f (a) * f (b) <0.

print_running("Solutions: Bisection method")
f = lambda x: 4 * x ** 3 + x + math.cos(x) - 10
tol = 10 ** -5
iter_max = 100
a = 1.0
b = 2.0
debug("tol")
debug("iter_max")
debug("a")
debug("b")
[root, i, converged] = solutions.bisection(f, a, b, tol, iter_max)
debug("root")
debug("i")
debug("converged")

# Newton method (find roots of an equation)
#   Pros:
#       It is a fast method.
#    Cons:
#       It may diverge;
#       It is necessary to calculate the derivative of the function;
#       It is necessary to give an initial x0 value where
#           f'(x0) must be nonzero.

print_running("Solutions: Newton method")
f = lambda x: 4 * x ** 3 + x + math.cos(x) - 10
df = lambda x: 12 * x ** 2 + 1 - math.sin(x)
tol = 10 ** -5
iter_max = 100
x0 = 1.0
debug("tol")
debug("iter_max")
debug("x0")
[root, i, converged] = solutions.newton(f, df, x0, tol, iter_max)
debug("root")
debug("i")
debug("converged")

# Secant method (find roots of an equation)
#   Pros:
#       It is a fast method (slower than Newton's method);
#       It is based on the Newton method, but does not need the derivative
#           of the function.
#   Cons:
#       It may diverge if the function is not approximately linear in the
#           range containing the root;
#       It is necessary to give two points 'a' and 'b' where
#           f(a)-f(b) must be nonzero.

print_running("Solutions: Secant method")
f = lambda x: 4 * x ** 3 + x + math.cos(x) - 10
tol = 10 ** -5
iter_max = 100
a = 1.0
b = 2.0
debug("tol")
debug("iter_max")
debug("a")
debug("b")
[root, i, converged] = solutions.secant(f, a, b, tol, iter_max)
debug("root")
debug("i")
debug("converged")

print_running("Interpolation: Lagrange method")
x = np.array([2, 11 / 4, 4])
y = np.array([1 / 2, 4 / 11, 1 / 4])
x_int = 3
debug("x")
debug("y")
debug("x_int")
[y_int] = interpolation.lagrange(x, y, x_int)
debug("y_int")

print_running("Interpolation: Neville method")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
x_int = 1.5
debug("x")
debug("y")
debug("x_int")
[y_int, q] = interpolation.neville(x, y, x_int)
debug("y_int")
debug("q")

print_running("Polynomials: Briot-Ruffini method")
a = np.array([2, 0, -3, 3, -4])
root = -2
debug("a")
debug("root")
[b, rest] = polynomials.briot_ruffini(a, root)
debug("b")
debug("rest")

print_running("Polynomials: Newton's Divided-Difference method")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
debug("x")
debug("y")
[f] = polynomials.newton_divided_difference(x, y)
debug("f")

print_running("Differentiation: Backward-difference method")
x = np.array([0.0, 0.2, 0.4])
y = np.array([0.00000, 0.74140, 1.3718])
debug("x")
debug("y")
[dy] = differentiation.derivative_backward_difference(x, y)
debug("dy")

print_running("Differentiation: Three-Point method")
x = np.array([1.1, 1.2, 1.3, 1.4])
y = np.array([9.025013, 11.02318, 13.46374, 16.44465])
debug("x")
debug("y")
[dy] = differentiation.derivative_three_point(x, y)
debug("dy")

print_running("Differentiation: Five-Point method")
x = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
y = np.array([-1.709847, -1.373823, -1.119214,
              -0.9160143, -0.7470223, -0.6015966])
debug("x")
debug("y")
[dy] = differentiation.derivative_five_point(x, y)
debug("dy")

print_running("Integration: Trapezoidal Rule")
x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
y = np.array([124, 134, 148, 156, 147, 133,
              121, 109, 99, 85, 78, 89, 104, 116, 123])
debug("x")
debug("y")
[xi] = integration.composite2_trapezoidal(x, y)
debug("xi")

print_running("Integration: Trapezoidal Rule")
f = lambda x: x ** 2 * math.log(x ** 2 + 1)
a = 0.0
b = 2.0
h = 0.25
n = int((b - a) / h)
debug("a")
debug("b")
debug("n")
[xi] = integration.composite_trapezoidal(f, b, a, n)
debug("xi")

print_running("Integration: Composite 1/3 Simpsons Rule")
x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
y = np.array([124, 134, 148, 156, 147, 133,
              121, 109, 99, 85, 78, 89, 104, 116, 123])
debug("x")
debug("y")
[xi] = integration.composite2_simpson(x, y)
debug("xi")

print_running("Integration: Composite 1/3 Simpsons Rule")
f = lambda x: x ** 2 * math.log(x ** 2 + 1)
a = 0.0
b = 2.0
h = 0.25
n = int((b - a) / h)
debug("a")
debug("b")
debug("n")
[xi] = integration.composite_simpson(f, b, a, n)
debug("xi")

print_running("ODE: Euler method")
f = lambda x, y: y - x ** 2 + 1
a = 0.0
b = 2.0
n = 10
ya = 0.5
debug("a")
debug("b")
debug("n")
debug("ya")
[vx, vy] = ode.euler(f, a, b, n, ya)
debug("vx")
debug("vy")

print_running("ODE: Taylor (Order Two) method")
f = lambda x, y: y - x ** 2 + 1
df1 = lambda x, y: y - x ** 2 + 1 - 2 * x
a = 0.0
b = 2.0
n = 10
ya = 0.5
debug("a")
debug("b")
debug("n")
debug("ya")
[vx, vy] = ode.taylor2(f, df1, a, b, n, ya)
debug("vx")
debug("vy")

print_running("ODE: Taylor (Order Four) method")
f = lambda x, y: y - x ** 2 + 1
df1 = lambda x, y: y - x ** 2 + 1 - 2 * x
df2 = lambda x, y: y - x ** 2 + 1 - 2 * x - 2
df3 = lambda x, y: y - x ** 2 + 1 - 2 * x - 2
a = 0.0
b = 2.0
n = 10
ya = 0.5
debug("a")
debug("b")
debug("n")
debug("ya")
[vx, vy] = ode.taylor4(f, df1, df2, df3, a, b, n, ya)
debug("vx")
debug("vy")

print_running("ODE: Runge-Kutta (Order Four) method")
f = lambda x, y: y - x ** 2 + 1
a = 0.0
b = 2.0
n = 10
ya = 0.5
debug("a")
debug("b")
debug("n")
debug("ya")
[vx, vy] = ode.rk4(f, a, b, n, ya)
debug("vx")
debug("vy")

print_running("ODE: Runge-Kutta (Order Four) method \
              for systems of differential equations")
f = []
f.append(lambda x, y: - 4 * y[0] + 3 * y[1] + 6)
f.append(lambda x, y: - 2.4 * y[0] + 1.6 * y[1] + 3.6)
a = 0.0
b = 0.5
h = 0.1
n = int((b - a) / h)
ya = np.zeros(len(f))
ya[0] = 0.0
ya[1] = 0.0
debug("a")
debug("b")
debug("n")
debug("ya")
[vx, vy] = ode.rk4_system(f, a, b, n, ya)
debug("vx")
debug("vy")

print_running("Linear Systems: Gaussian Elimination")
a = np.array([[1, -1, 2, -1], [2, -2, 3, -3], [1, 1, 1, 0], [1, -1, 4, 3]])
b = np.array([-8, -20, -2, 4])
debug("a")
debug("b")
[a] = linear_systems.gauss_elimination_pp(a, b)
debug("a")

print_running("Linear Systems: Backward Substitution")
u = a[:, 0:-1]
d = a[:, -1]
debug("u")
debug("d")
[x] = linear_systems.backward_substitution(u, d)
debug("x")

print_running("Linear Systems: Forward Substitution")
l = np.array([[3, 0, 0, 0], [-1, 1, 0, 0], [3, -2, -1, 0], [1, -2, 6, 2]])
c = np.array([5, 6, 4, 2])
debug("l")
debug("c")
[x] = linear_systems.forward_substitution(l, c)
debug("x")

print_running("Iteractive Linear Systems: Jacobi")
a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
b = np.array([6, 25, -11, 15])
x0 = np.array([0, 0, 0, 0])
tol = 10 ** -3
iter_max = 10
debug("a")
debug("b")
debug("x0")
debug("tol")
debug("iter_max")
[x, i] = linear_systems_iterative.jacobi(a, b, x0, tol, iter_max)
debug("x")
debug("i")

print_running("Iteractive Linear Systems: Gauss-Seidel")
a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]])
b = np.array([6, 25, -11, 15])
x0 = np.array([0, 0, 0, 0])
tol = 10 ** -3
iter_max = 10
debug("a")
debug("b")
debug("x0")
debug("tol")
debug("iter_max")
[x, i] = linear_systems_iterative.gauss_seidel(a, b, x0, tol, iter_max)
debug("x")
debug("i")
