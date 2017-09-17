import math


def bisection(f, a: float, b: float, tol: float, iter_max: int) -> [float, int, bool]:
    """
    Calculates the root of an equation by Bisection method
    Inputs:
            f: Function f(x)
            a: Lower limit
            b: Upper limit
          tol: Tolerance
     iter_max: Maximum number of iterations
    Outpus:
         root: Root value
         iter: Used iterations
    converged: Found the root
    """

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ("Error: The function does not change signal at the ends of the given interval.")

    delta_x = math.fabs(b - a) / 2

    x = 0
    converged = False
    iter = 0
    for iter in range(0, iter_max + 1):
        x = (a + b) / 2

        fx = f(x)
        print('iter: %.3d\t x: %+.4f\t fx: %+.4f\t delta_x: %+.4f\n' % (iter, x, fx, delta_x), end="")

        if delta_x <= tol and math.fabs(fx) <= tol:
            converged = True
            break

        if (fa * fx > 0):
            a = x
            fa = fx
        else:
            b = x

        delta_x = delta_x / 2
    else:
        print("Warning: The method not converged.")

    root = x

    return [root, iter, converged]


def secant(f, a: float, b: float, tol: float, iter_max: int) -> [float, int, bool]:
    """
    Calculates the root of an equation by Secant method
    Inputs:
            f: Function f(x)
            a: Lower limit
            b: Upper limit
          tol: Tolerance
     iter_max: Maximum number of iterations
    Outpus:
         root: Root value
         iter: Used iterations
    converged: Found the root
    """

    fa = f(a)
    fb = f(b)

    if math.fabs(fa) < math.fabs(fb):
        a, b = b, a
        fa, fb = fb, fa

    x = b
    fx = fb

    converged = False
    iter = 0
    for iter in range(0, iter_max + 1):
        delta_x = -fx / (fb - fa) * (b - a)
        x += delta_x

        fx = f(x)
        print("iter: %.3d\t x: %+.4f\t fx: %+.4f\t delta_x: %+.4f\n" % (iter, x, fx, delta_x), end="")
        if math.fabs(delta_x) <= tol and math.fabs(fx) <= tol:
            converged = True
            break

        a, b = b, x
        fa, fb = fb, fx
    else:
        print("Warning: The method not converged.")

    root = x

    return [root, iter, converged]
