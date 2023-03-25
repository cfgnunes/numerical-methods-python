"""
Numerical methods implementation in Python.

Author: Cristiano Fraga G. Nunes <cfgnunes@gmail.com>

The minimum required Python version is 3.6.
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


def print_docstring(func):
    """Print the docstring of a function (decorator)."""
    def wrapper(*args, **kwargs):
        print(func.__doc__)
        result = func(*args, **kwargs)
        print("\n")
        return result
    return wrapper


@print_docstring
def example_solution_bisection():
    """Run an example 'Solutions: Bisection'."""
    # Bisection method (find roots of an equation)
    #   Pros:
    #       It is a reliable method with guaranteed convergence;
    #       It is a simple method that searches for the root by employing a
    #           binary search;
    #       There is no need to calculate the derivative of the function.
    #   Cons:
    #       Slow convergence;
    #       It is necessary to enter a search interval [a, b];
    #       The interval reported must have a signal exchange, f (a) * f (b)<0.

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Execution:")
    [root, i, converged] = solutions.bisection(f, a, b, toler, iter_max)

    print("Output:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_secant():
    """Run an example 'Solutions: Secant'."""
    # Secant method (find roots of an equation)
    #   Pros:
    #       It is a fast method (slower than Newton's method);
    #       It is based on the Newton method but does not need the derivative
    #           of the function.
    #   Cons:
    #       It may diverge if the function is not approximately linear in the
    #           range containing the root;
    #       It is necessary to give two points, 'a' and 'b' where
    #           f(a)-f(b) must be nonzero.

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Execution:")
    [root, i, converged] = solutions.secant(f, a, b, toler, iter_max)

    print("Output:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_regula_falsi():
    """Run an example 'Solutions: Regula Falsi'."""

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Execution:")
    [root, i, converged] = solutions.regula_falsi(f, a, b, toler, iter_max)

    print("Output:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_pegasus():
    """Run an example 'Solutions: Pegasus'."""

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Execution:")
    [root, i, converged] = solutions.pegasus(f, a, b, toler, iter_max)

    print("Output:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_muller():
    """Run an example 'Solutions: Muller'."""

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    a = -1.0
    b = 2.0
    toler = 0.01
    iter_max = 100

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Execution:")
    [root, i, converged] = solutions.muller(f, a, b, toler, iter_max)

    print("Output:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_solution_newton():
    """Run an example 'Solutions: Newton'."""
    # Newton method (find roots of an equation)
    #   Pros:
    #       It is a fast method.
    #    Cons:
    #       It may diverge;
    #       It is necessary to calculate the derivative of the function;
    #       It is necessary to give an initial x0 value where
    #           f'(x0) must be nonzero.

    def f(x):
        return 2 * x ** 3 - math.cos(x + 1) - 3

    def df(x):
        return 12 * x ** 2 + 1 - math.sin(x)

    x0 = 1.0
    toler = 0.01
    iter_max = 100

    print("Inputs:")
    print(f"x0 = {x0}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    print("Execution:")
    [root, i, converged] = solutions.newton(f, df, x0, toler, iter_max)

    print("Output:")
    print(f"root = {root:.5f}")
    print(f"i = {i}")
    print(f"converged = {converged}")


@print_docstring
def example_interpolation_lagrange():
    """Run an example 'Interpolation: Lagrange'."""
    x = np.array([2, 11 / 4, 4])
    y = np.array([1 / 2, 4 / 11, 1 / 4])
    x_int = 3

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    [y_int] = interpolation.lagrange(x, y, x_int)

    print("Output:")
    print(f"y_int = {y_int:.5f}")


@print_docstring
def example_interpolation_newton():
    """Run an example 'Interpolation: Newton'."""
    x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
    y = np.array([0.3162, 0.5477, 0.6325, 0.7746, 0.8367])
    x_int = 0.2

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    [y_int] = interpolation.newton(x, y, x_int)

    print("Output:")
    print(f"y_int = {y_int:.5f}")


@print_docstring
def example_interpolation_gregory_newton():
    """Run an example 'Interpolation: Gregory-Newton'."""
    x = np.array([110, 120, 130])
    y = np.array([2.0410, 2.0790, 2.1140])
    x_int = 115

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    [y_int] = interpolation.gregory_newton(x, y, x_int)

    print("Output:")
    print(f"y_int = {y_int:.5f}")


@print_docstring
def example_interpolation_neville():
    """Run an example 'Interpolation: Neville'."""
    x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
    x_int = 1.5

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x_int = {x_int}")

    [y_int, q] = interpolation.neville(x, y, x_int)

    print("Output:")
    print(f"y_int = {y_int:.5f}")
    print(f"q =\n{q}")


@print_docstring
def example_polynomial_root_limits():
    """Run an example 'Polynomials: Root limits'."""
    c = np.array([1, 2, -13, -14, 24])

    print("Inputs:")
    print(f"c = {c}")

    limits = polynomials.root_limits(c)

    print("Output:")
    print(f"limits = {limits}")


@print_docstring
def example_polynomial_briot_ruffini():
    """Run an example 'Polynomials: Briot-Ruffini'."""
    a = np.array([2, 0, -3, 3, -4])
    root = -2

    print("Inputs:")
    print(f"a = {a}")
    print(f"root = {root:.5f}")

    [b, rest] = polynomials.briot_ruffini(a, root)

    print("Output:")
    print(f"b = {b}")
    print(f"rest = {rest}")


@print_docstring
def example_polynomial_newton_divided_difference():
    """Run an example 'Polynomials: Newton's Divided-Difference'."""
    x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")

    [f] = polynomials.newton_divided_difference(x, y)

    print("Output:")
    print(f"f = {f}")


@print_docstring
def example_differentiation_backward_difference():
    """Run an example 'Differentiation: Backward-difference'."""
    x = np.array([0.0, 0.2, 0.4])
    y = np.array([0.00000, 0.74140, 1.3718])

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")

    [dy] = differentiation.backward_difference(x, y)

    print("Output:")
    print(f"dy = {dy}")


@print_docstring
def example_differentiation_three_point():
    """Run an example 'Differentiation: Three-Point'."""
    x = np.array([1.1, 1.2, 1.3, 1.4])
    y = np.array([9.025013, 11.02318, 13.46374, 16.44465])

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")

    [dy] = differentiation.three_point(x, y)

    print("Output:")
    print(f"dy = {dy}")


@print_docstring
def example_differentiation_five_point():
    """Run an example 'Differentiation: Five-Point'."""
    x = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
    y = np.array([-1.709847, -1.373823, -1.119214,
                  -0.9160143, -0.7470223, -0.6015966])

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")

    [dy] = differentiation.five_point(x, y)

    print("Output:")
    print(f"dy = {dy}")


@print_docstring
def example_trapezoidal_array():
    """Run an example 'Integration: Trapezoidal Rule'."""
    x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    y = np.array([124, 134, 148, 156, 147, 133,
                  121, 109, 99, 85, 78, 89, 104, 116, 123])

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")

    [xi] = integration.trapezoidal_array(x, y)

    print("Output:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_trapezoidal():
    """Run an example 'Integration: Trapezoidal Rule'."""
    def f(x):
        return x ** 2 * math.log(x ** 2 + 1)

    a = 0.0
    b = 2.0
    h = 0.25
    n = int((b - a) / h)

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")

    [xi] = integration.trapezoidal(f, b, a, n)

    print("Output:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_simpson_array():
    """Run an example 'Integration: Composite 1/3 Simpsons Rule'."""
    x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    y = np.array([124, 134, 148, 156, 147, 133,
                  121, 109, 99, 85, 78, 89, 104, 116, 123])

    print("Inputs:")
    print(f"x = {x}")
    print(f"y = {y}")

    [xi] = integration.simpson_array(x, y)

    print("Output:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_simpson():
    """Run an example 'Integration: Composite 1/3 Simpsons Rule'."""
    def f(x):
        return x ** 2 * math.log(x ** 2 + 1)

    a = 0.0
    b = 2.0
    h = 0.25
    n = int((b - a) / h)

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")

    [xi] = integration.simpson(f, b, a, n)

    print("Output:")
    print(f"xi = {xi:.5f}")


@print_docstring
def example_ode_euler():
    """Run an example 'ODE: Euler'."""
    def f(x, y):
        return y - x ** 2 + 1

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Execution:")
    [vx, vy] = ode.euler(f, a, b, n, ya)

    print("Output:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_taylor2():
    """Run an example 'ODE: Taylor (Order 2)'."""
    def f(x, y):
        return y - x ** 2 + 1

    def df1(x, y):
        return y - x ** 2 + 1 - 2 * x

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Execution:")
    [vx, vy] = ode.taylor2(f, df1, a, b, n, ya)

    print("Output:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_taylor4():
    """Run an example 'ODE: Taylor (Order 4)'."""
    def f(x, y):
        return y - x ** 2 + 1

    def df1(x, y):
        return y - x ** 2 + 1 - 2 * x

    def df2(x, y):
        return y - x ** 2 + 1 - 2 * x - 2

    def df3(x, y):
        return y - x ** 2 + 1 - 2 * x - 2

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Execution:")
    [vx, vy] = ode.taylor4(f, df1, df2, df3, a, b, n, ya)

    print("Output:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_rk4():
    """Run an example 'ODE: Runge-Kutta (Order 4)'."""
    def f(x, y):
        return y - x ** 2 + 1

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    [vx, vy] = ode.rk4(f, a, b, n, ya)

    print("Output:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_ode_rk4_system():
    """Run an example 'ODE: Runge-Kutta (Order 4) for systems of diff. eq.'."""
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

    print("Inputs:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"n = {n}")
    print(f"ya = {ya}")

    print("Execution:")
    [vx, vy] = ode.rk4_system(f, a, b, n, ya)

    print("Output:")
    print(f"vx = {vx}")
    print(f"vy = {vy}")


@print_docstring
def example_gauss_elimination_pp():
    """Run an example 'Linear Systems: Gaussian Elimination'."""
    a = np.array([[1, -1, 2, -1], [2, -2, 3, -3], [1, 1, 1, 0], [1, -1, 4, 3]])
    b = np.array([-8, -20, -2, 4])

    print("Inputs:")
    print(f"a =\n{a}")
    print(f"b = {b}")

    [a] = linear_systems.gauss_elimination_pp(a, b)

    print("Output:")
    print(f"a =\n{a}")

    return a


@print_docstring
def example_backward_substitution(a):
    """Run an example 'Linear Systems: Backward Substitution'."""
    upper = a[:, 0:-1]
    d = a[:, -1]

    print("Inputs:")
    print(f"upper =\n{upper}")
    print(f"d = {d}")

    [x] = linear_systems.backward_substitution(upper, d)

    print("Output:")
    print(f"x = {x}")


@print_docstring
def example_forward_substitution():
    """Run an example 'Linear Systems: Forward Substitution'."""
    lower = np.array([[3, 0, 0, 0], [-1, 1, 0, 0],
                      [3, -2, -1, 0], [1, -2, 6, 2]])
    c = np.array([5, 6, 4, 2])

    print("Inputs:")
    print(f"lower =\n{lower}")
    print(f"c = {c}")

    [x] = linear_systems.forward_substitution(lower, c)

    print("Output:")
    print(f"x = {x}")


@print_docstring
def example_jacobi():
    """Run an example 'Iterative Linear Systems: Jacobi'."""
    a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
                  [2, -1, 10, -1], [0, 3, -1, 8]])
    b = np.array([6, 25, -11, 15])
    x0 = np.array([0, 0, 0, 0])
    toler = 10 ** -3
    iter_max = 10

    print("Inputs:")
    print(f"a =\n{a}")
    print(f"b = {b}")
    print(f"x0 = {x0}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    [x, i] = linear_systems_iterative.jacobi(a, b, x0, toler, iter_max)

    print("Output:")
    print(f"x = {x}")
    print(f"i = {i}")


@print_docstring
def example_gauss_seidel():
    """Run an example 'Iterative Linear Systems: Gauss-Seidel'."""
    a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
                  [2, -1, 10, -1], [0, 3, -1, 8]])
    b = np.array([6, 25, -11, 15])
    x0 = np.array([0, 0, 0, 0])
    toler = 10 ** -3
    iter_max = 10

    print("Inputs:")
    print(f"a =\n{a}")
    print(f"b = {b}")
    print(f"x0 = {x0}")
    print(f"toler = {toler}")
    print(f"iter_max = {iter_max}")

    [x, i] = linear_systems_iterative.gauss_seidel(a, b, x0, toler, iter_max)

    print("Output:")
    print(f"x = {x}")
    print(f"i = {i}")


def main():
    """Run the main function."""
    # Execute all examples

    # Solutions of equations
    example_solution_bisection()
    example_solution_secant()
    example_solution_regula_falsi()
    example_solution_pegasus()
    example_solution_muller()
    example_solution_newton()

    # Interpolation
    example_interpolation_lagrange()
    example_interpolation_newton()
    example_interpolation_gregory_newton()
    example_interpolation_neville()

    # Algorithms for polynomials
    example_polynomial_root_limits()
    example_polynomial_briot_ruffini()
    example_polynomial_newton_divided_difference()

    # Numerical differentiation
    example_differentiation_backward_difference()
    example_differentiation_three_point()
    example_differentiation_five_point()

    # Numerical integration
    example_trapezoidal_array()
    example_trapezoidal()
    example_simpson_array()
    example_simpson()

    # Initial-value problems for ordinary differential equations
    example_ode_euler()
    example_ode_taylor2()
    example_ode_taylor4()
    example_ode_rk4()

    # Systems of differential equations
    example_ode_rk4_system()

    # Methods for Linear Systems
    a = example_gauss_elimination_pp()
    example_backward_substitution(a)
    example_forward_substitution()

    # Iterative Methods for Linear Systems
    example_jacobi()
    example_gauss_seidel()


if __name__ == '__main__':
    main()
