import numpy as np
import math
import interpolation
import polynomials
import solutions

"""
Numerical methods implementation in Python 3.
Author: Cristiano Nunes
E-mail: <cfgnunes@gmail.com>
Repository: https://github.com/cfgnunes/numerical-methods
"""

def debug(variable):
    print (variable, "=", eval(variable))

def print_running(message):
    print("\n> Running", message)

"""
Bisection method (find roots of an equation)
    Pros:
        It is a reliable method with guaranteed convergence;
        It is a simple method that does the search of the root by means of a binary search;
        There is no need to calculate the derivative of the function.
    Cons:
        Slow convergence;
        It is necessary to enter a search interval [a, b];
        The interval reported must have a signal exchange, f (a) * f (b) <0.
"""
print_running("Bisection method")
f = lambda x: 4 * x ** 3 + x + math.cos(x) - 10
tol = 10 ** -5
iter_max = 100
a = 1.0
b = 2.0
debug("tol")
debug("iter_max")
debug("a")
debug("b")
[root, iter, converged] = solutions.bisection(f, a, b, tol, iter_max)
debug("root")
debug("iter")
debug("converged")

"""
Secant method (find roots of an equation)
    Pros:
        It is a fast method (slower than Newton's method);
        It is based on the Newton method, but does not need the derivative of the function.
    Cons:
        It may diverge if the function is not approximately linear in the range containing the root;
        It is necessary to give two points 'a' and 'b' where f (a) -f (b) must be nonzero.
"""
print_running("Secant method")
f = lambda x: 4 * x ** 3 + x + math.cos(x) - 10
tol = 10 ** -5
iter_max = 100
a = 1.0
b = 2.0
debug("tol")
debug("iter_max")
debug("a")
debug("b")
[root, iter, converged] = solutions.secant(f, a, b, tol, iter_max)
debug("root")
debug("iter")
debug("converged")

print_running("Lagrange method")
x = np.array([2, 11 / 4, 4])
y = np.array([1 / 2, 4 / 11, 1 / 4])
x_int = 3
debug("x")
debug("y")
debug("x_int")
[y_int] = interpolation.lagrange(x, y, x_int)
debug("y_int")

print_running("Neville method")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
x_int = 1.5
debug("x")
debug("y")
debug("x_int")
[y_int, q] = interpolation.neville(x, y, x_int)
debug("y_int")
debug("q")

print_running("Briot-Ruffini method")
a = np.array([2, 0, -3, 3, -4])
root = -2
debug("a")
debug("root")
[b, rest] = polynomials.briot_ruffini(a, root)
debug("b")
debug("rest")

print_running("Newton's Divided-Difference method")
x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
debug("x")
debug("y")
[f] = polynomials.newton_divided_difference(x, y)
debug("f")
