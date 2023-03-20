"""
Numerical methods implementation in Python.

Author: Cristiano Fraga G. Nunes <cfgnunes@gmail.com>

Minimum required version of the Python: 3.0.
"""

import inspect
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


def print_function_docstring():
    """Print the docstring of a caller function."""
    outerframe = inspect.currentframe().f_back
    functionname = outerframe.f_code.co_name
    docstring = globals()[functionname].__doc__
    print("\n\n" + docstring)


def run_example_bisection():
    """Run an example 'Solutions: Bisection'."""
    print_function_docstring()

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
        return 4 * x ** 3 + x + math.cos(x) - 10

    a = 1.0
    b = 2.0
    toler = 10 ** -5
    iter_max = 100

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\ttoler = {toler}")
    print(f"\titer_max = {iter_max}")

    [root, i, converged] = solutions.bisection(f, a, b, toler, iter_max)

    print("Output:")
    print(f"\troot = {root:.4f}")
    print(f"\ti = {i}")
    print(f"\tconverged = {converged}")


def run_example_newton():
    """Run an example 'Solutions: Newton'."""
    print_function_docstring()

    # Newton method (find roots of an equation)
    #   Pros:
    #       It is a fast method.
    #    Cons:
    #       It may diverge;
    #       It is necessary to calculate the derivative of the function;
    #       It is necessary to give an initial x0 value where
    #           f'(x0) must be nonzero.

    def f(x):
        return 4 * x ** 3 + x + math.cos(x) - 10

    def df(x):
        return 12 * x ** 2 + 1 - math.sin(x)

    x0 = 1.0
    toler = 10 ** -5
    iter_max = 100

    print("Input:")
    print(f"\tx0 = {x0}")
    print(f"\ttoler = {toler}")
    print(f"\titer_max = {iter_max}")

    [root, i, converged] = solutions.newton(f, df, x0, toler, iter_max)

    print("Output:")
    print(f"\troot = {root:.4f}")
    print(f"\ti = {i}")
    print(f"\tconverged = {converged}")


def run_example_regula_falsi():
    """Run an example 'Solutions: Regula Falsi'."""
    print_function_docstring()

    def f(x):
        return 4 * x ** 3 + x + math.cos(x) - 10

    a = 1.0
    b = 2.0
    toler = 10 ** -5
    iter_max = 100

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\ttoler = {toler}")
    print(f"\titer_max = {iter_max}")

    [root, i, converged] = solutions.regula_falsi(f, a, b, toler, iter_max)

    print("Output:")
    print(f"\troot = {root:.4f}")
    print(f"\ti = {i}")
    print(f"\tconverged = {converged}")


def run_example_secant():
    """Run an example 'Solutions: Secant'."""
    print_function_docstring()

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
        return 4 * x ** 3 + x + math.cos(x) - 10

    a = 1.0
    b = 2.0
    toler = 10 ** -5
    iter_max = 100

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\ttoler = {toler}")
    print(f"\titer_max = {iter_max}")

    [root, i, converged] = solutions.secant(f, a, b, toler, iter_max)

    print("Output:")
    print(f"\troot = {root:.4f}")
    print(f"\ti = {i}")
    print(f"\tconverged = {converged}")


def run_example_lagrange():
    """Run an example 'Interpolation: Lagrange'."""
    print_function_docstring()

    x = np.array([2, 11 / 4, 4])
    y = np.array([1 / 2, 4 / 11, 1 / 4])
    x_int = 3

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")
    print(f"\tx_int = {x_int}")

    [y_int] = interpolation.lagrange(x, y, x_int)

    print("Output:")
    print(f"\ty_int = {y_int}")


def run_example_neville():
    """Run an example 'Interpolation: Neville'."""
    print_function_docstring()

    x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])
    x_int = 1.5

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")
    print(f"\tx_int = {x_int}")

    [y_int, q] = interpolation.neville(x, y, x_int)

    print("Output:")
    print(f"\ty_int = {y_int}")
    print(f"\tq = {q}")


def run_example_briot_ruffini():
    """Run an example 'Polynomials: Briot-Ruffini'."""
    print_function_docstring()

    a = np.array([2, 0, -3, 3, -4])
    root = -2

    print("Input:")
    print(f"\ta = {a}")
    print(f"\troot = {root:.4f}")

    [b, rest] = polynomials.briot_ruffini(a, root)

    print("Output:")
    print(f"\tb = {b}")
    print(f"\trest = {rest}")


def run_example_newton_divided_difference():
    """Run an example 'Polynomials: Newton's Divided-Difference'."""
    print_function_docstring()

    x = np.array([1.0, 1.3, 1.6, 1.9, 2.2])
    y = np.array([0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623])

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    [f] = polynomials.newton_divided_difference(x, y)

    print("Output:")
    print(f"\tf = {f}")


def run_example_derivative_backward_difference():
    """Run an example 'Differentiation: Backward-difference'."""
    print_function_docstring()

    x = np.array([0.0, 0.2, 0.4])
    y = np.array([0.00000, 0.74140, 1.3718])

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    [dy] = differentiation.derivative_backward_difference(x, y)

    print("Output:")
    print(f"\tdy = {dy}")


def run_example_derivative_three_point():
    """Run an example 'Differentiation: Three-Point'."""
    print_function_docstring()

    x = np.array([1.1, 1.2, 1.3, 1.4])
    y = np.array([9.025013, 11.02318, 13.46374, 16.44465])

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    [dy] = differentiation.derivative_three_point(x, y)

    print("Output:")
    print(f"\tdy = {dy}")


def run_example_derivative_five_point():
    """Run an example 'Differentiation: Five-Point'."""
    print_function_docstring()

    x = np.array([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
    y = np.array([-1.709847, -1.373823, -1.119214,
                  -0.9160143, -0.7470223, -0.6015966])

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    [dy] = differentiation.derivative_five_point(x, y)

    print("Output:")
    print(f"\tdy = {dy}")


def run_example_composite2_trapezoidal():
    """Run an example 'Integration: Trapezoidal Rule'."""
    print_function_docstring()

    x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    y = np.array([124, 134, 148, 156, 147, 133,
                  121, 109, 99, 85, 78, 89, 104, 116, 123])

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    [xi] = integration.composite2_trapezoidal(x, y)

    print("Output:")
    print(f"\txi = {xi}")


def run_example_composite_trapezoidal():
    """Run an example 'Integration: Trapezoidal Rule'."""
    print_function_docstring()

    def f(x):
        return x ** 2 * math.log(x ** 2 + 1)

    a = 0.0
    b = 2.0
    h = 0.25
    n = int((b - a) / h)

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tn = {n}")

    [xi] = integration.composite_trapezoidal(f, b, a, n)

    print("Output:")
    print(f"\txi = {xi}")


def run_example_composite2_simpson():
    """Run an example 'Integration: Composite 1/3 Simpsons Rule'."""
    print_function_docstring()

    x = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    y = np.array([124, 134, 148, 156, 147, 133,
                  121, 109, 99, 85, 78, 89, 104, 116, 123])

    print("Input:")
    print(f"\tx = {x}")
    print(f"\ty = {y}")

    [xi] = integration.composite2_simpson(x, y)

    print("Output:")
    print(f"\txi = {xi}")


def run_example_ccomposite_simpson():
    """Run an example 'Integration: Composite 1/3 Simpsons Rule'."""
    print_function_docstring()

    def f(x):
        return x ** 2 * math.log(x ** 2 + 1)

    a = 0.0
    b = 2.0
    h = 0.25
    n = int((b - a) / h)

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tn = {n}")

    [xi] = integration.composite_simpson(f, b, a, n)

    print("Output:")
    print(f"\txi = {xi}")


def run_example_composite_simpson():
    """Run an example 'ODE: Euler'."""
    print_function_docstring()

    def f(x, y):
        return y - x ** 2 + 1

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tn = {n}")
    print(f"\tya = {ya}")

    [vx, vy] = ode.euler(f, a, b, n, ya)

    print("Output:")
    print(f"\tvx = {vx}")
    print(f"\tvy = {vy}")


def run_example_taylor2():
    """Run an example 'ODE: Taylor (Order 2)'."""
    print_function_docstring()

    def f(x, y):
        return y - x ** 2 + 1

    def df1(x, y):
        return y - x ** 2 + 1 - 2 * x

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tn = {n}")
    print(f"\tya = {ya}")

    [vx, vy] = ode.taylor2(f, df1, a, b, n, ya)

    print("Output:")
    print(f"\tvx = {vx}")
    print(f"\tvy = {vy}")


def run_example_taylor4():
    """Run an example 'ODE: Taylor (Order 4)'."""
    print_function_docstring()

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

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tn = {n}")
    print(f"\tya = {ya}")

    [vx, vy] = ode.taylor4(f, df1, df2, df3, a, b, n, ya)

    print("Output:")
    print(f"\tvx = {vx}")
    print(f"\tvy = {vy}")


def run_example_rk4():
    """Run an example 'ODE: Runge-Kutta (Order 4)'."""
    print_function_docstring()

    def f(x, y):
        return y - x ** 2 + 1

    a = 0.0
    b = 2.0
    n = 10
    ya = 0.5

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tn = {n}")
    print(f"\tya = {ya}")

    [vx, vy] = ode.rk4(f, a, b, n, ya)

    print("Output:")
    print(f"\tvx = {vx}")
    print(f"\tvy = {vy}")


def run_example_rk4_system():
    """Run an example 'ODE: Runge-Kutta (Order 4) for systems of diff. eq.'."""
    print_function_docstring()

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

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tn = {n}")
    print(f"\tya = {ya}")

    [vx, vy] = ode.rk4_system(f, a, b, n, ya)

    print("Output:")
    print(f"\tvx = {vx}")
    print(f"\tvy = {vy}")


def run_example_gauss_elimination_pp():
    """Run an example 'Linear Systems: Gaussian Elimination'."""
    print_function_docstring()

    a = np.array([[1, -1, 2, -1], [2, -2, 3, -3], [1, 1, 1, 0], [1, -1, 4, 3]])
    b = np.array([-8, -20, -2, 4])

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")

    [a] = linear_systems.gauss_elimination_pp(a, b)

    print("Output:")
    print(f"\ta = {a}")

    return a


def run_example_backward_substitution(a):
    """Run an example 'Linear Systems: Backward Substitution'."""
    print_function_docstring()

    upper = a[:, 0:-1]
    d = a[:, -1]

    print("Input:")
    print(f"\tupper = {upper}")
    print(f"\td = {d}")

    [x] = linear_systems.backward_substitution(upper, d)

    print("Output:")
    print(f"\tx = {x}")


def run_example_forward_substitution():
    """Run an example 'Linear Systems: Forward Substitution'."""
    print_function_docstring()

    lower = np.array([[3, 0, 0, 0], [-1, 1, 0, 0],
                      [3, -2, -1, 0], [1, -2, 6, 2]])
    c = np.array([5, 6, 4, 2])

    print("Input:")
    print(f"\tlower = {lower}")
    print(f"\tc = {c}")

    [x] = linear_systems.forward_substitution(lower, c)

    print("Output:")
    print(f"\tx = {x}")


def run_example_jacobi():
    """Run an example 'Iterative Linear Systems: Jacobi'."""
    print_function_docstring()

    a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
                  [2, -1, 10, -1], [0, 3, -1, 8]])
    b = np.array([6, 25, -11, 15])
    x0 = np.array([0, 0, 0, 0])
    toler = 10 ** -3
    iter_max = 10

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tx0 = {x0}")
    print(f"\ttoler = {toler}")
    print(f"\titer_max = {iter_max}")

    [x, i] = linear_systems_iterative.jacobi(a, b, x0, toler, iter_max)

    print("Output:")
    print(f"\tx = {x}")
    print(f"\ti = {i}")


def run_example_gauss_seidel():
    """Run an example 'Iterative Linear Systems: Gauss-Seidel'."""
    print_function_docstring()

    a = np.array([[10, -1, 2, 0], [-1, 11, -1, 3],
                  [2, -1, 10, -1], [0, 3, -1, 8]])
    b = np.array([6, 25, -11, 15])
    x0 = np.array([0, 0, 0, 0])
    toler = 10 ** -3
    iter_max = 10

    print("Input:")
    print(f"\ta = {a}")
    print(f"\tb = {b}")
    print(f"\tx0 = {x0}")
    print(f"\ttoler = {toler}")
    print(f"\titer_max = {iter_max}")

    [x, i] = linear_systems_iterative.gauss_seidel(a, b, x0, toler, iter_max)

    print("Output:")
    print(f"\tx = {x}")
    print(f"\ti = {i}")


def main():
    """Run all examples."""
    run_example_bisection()
    run_example_newton()
    run_example_regula_falsi()
    run_example_secant()
    run_example_lagrange()
    run_example_neville()
    run_example_briot_ruffini()
    run_example_newton_divided_difference()
    run_example_derivative_backward_difference()
    run_example_derivative_three_point()
    run_example_derivative_five_point()
    run_example_composite2_trapezoidal()
    run_example_composite_trapezoidal()
    run_example_composite2_simpson()
    run_example_ccomposite_simpson()
    run_example_composite_simpson()
    run_example_taylor2()
    run_example_taylor4()
    run_example_rk4()
    run_example_rk4_system()
    a = run_example_gauss_elimination_pp()
    run_example_backward_substitution(a)
    run_example_forward_substitution()
    run_example_jacobi()
    run_example_gauss_seidel()


if __name__ == '__main__':
    main()
