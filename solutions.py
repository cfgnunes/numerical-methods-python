"""Methods for solutions of equations."""

import math


def bisection(f, a, b, toler, iter_max):
    """Calculate the root of an equation by the Bisection method.

    Args:
        f: function f(x).
        a: lower limit.
        b: upper limit.
        toler: tolerance (stopping criterion).
        iter_max: maximum number of iterations (stopping criterion).

    Returns:
        root: root value.
        iter: number of iterations used by the method.
        converged: flag to indicate if the root was found.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("The function does not change signal at \
              the ends of the given interval.")

    delta_x = math.fabs(b - a) / 2

    x = 0
    converged = False
    for i in range(0, iter_max + 1):
        x = (a + b) / 2
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if delta_x <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        if fa * fx > 0:
            a = x
            fa = fx
        else:
            b = x

        delta_x = delta_x / 2

    root = x
    return [root, i, converged]


def newton(f, df, x0, toler, iter_max):
    """Calculate the root of an equation by the Newton method.

    Args:
        f: function f(x).
        df: derivative of function f(x).
        x0: initial guess.
        toler: tolerance (stopping criterion).
        iter_max: maximum number of iterations (stopping criterion).

    Returns:
        root: root value.
        iter: number of iterations used by the method.
        converged: flag to indicate if the root was found.
    """
    fx = f(x0)
    dfx = df(x0)
    x = x0

    print(f"i = 000,\tx = {x:.4f},\tdfx = {dfx:.4f},\tfx = {fx:.4f}")

    converged = False
    for i in range(1, iter_max + 1):
        delta_x = -fx / dfx
        x += delta_x
        fx = f(x)
        dfx = df(x)

        print(f"i = {i:03d},\tx = {x:.4f},\tdfx = {dfx:.4f},\t", end="")
        print(f"fx = {fx:.4f},\tdx = {delta_x:.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler or dfx == 0:
            converged = True
            break

    root = x
    return [root, i, converged]


def regula_falsi(f, a, b, toler, iter_max):
    """Calculate the root of an equation by the Regula Falsi method.

    Args:
        f: function f(x).
        a: lower limit.
        b: upper limit.
        toler: tolerance (stopping criterion).
        iter_max: maximum number of iterations (stopping criterion).

    Returns:
        root: root value.
        iter: number of iterations used by the method.
        converged: flag to indicate if the root was found.
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("The function does not change signal at \
              the ends of the given interval.")

    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    x = b
    fx = fb

    converged = False
    for i in range(0, iter_max + 1):
        delta_x = -fx / (fb - fa) * (b - a)
        x += delta_x
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        if fx < 0:
            a = x
            fa = fx
        else:
            b = x
            fb = fx

    root = x
    return [root, i, converged]


def secant(f, a, b, toler, iter_max):
    """Calculate the root of an equation by the Secant method.

    Args:
        f: function f(x).
        a: lower limit.
        b: upper limit.
        toler: tolerance (stopping criterion).
        iter_max: maximum number of iterations (stopping criterion).

    Returns:
        root: root value.
        iter: number of iterations used by the method.
        converged: flag to indicate if the root was found.
    """
    fa = f(a)
    fb = f(b)

    if fb - fa == 0:
        raise ValueError("f(b)-f(a) must be nonzero.")

    if b - a == 0:
        raise ValueError("b-a must be nonzero.")

    if math.fabs(fa) < math.fabs(fb):
        a, b = b, a
        fa, fb = fb, fa

    x = b
    fx = fb

    converged = False
    for i in range(0, iter_max + 1):
        delta_x = -fx / (fb - fa) * (b - a)
        x += delta_x
        fx = f(x)

        print(f"i = {i:03d},\tx = {x:+.4f},\t", end="")
        print(f"fx = {fx:+.4f},\tdx = {delta_x:+.4f}")

        if math.fabs(delta_x) <= toler and math.fabs(fx) <= toler:
            converged = True
            break

        a, b = b, x
        fa, fb = fb, fx

    root = x
    return [root, i, converged]
