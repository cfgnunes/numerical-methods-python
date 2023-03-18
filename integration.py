"""Methods for numerical integration."""


def composite_simpson(f, b, a, n):
    """Calculate the integral from 1/3 Simpson's Rule.

    Args:
        f: function f(x).
        a: the initial point.
        b: the final point.
        n: number of intervals.

    Returns:
        xi: integral value.
    """
    h = (b - a) / n

    sum_odd = 0
    sum_even = 0

    for i in range(0, n - 1):
        x = a + (i + 1) * h
        if (i + 1) % 2 == 0:
            sum_even += f(x)
        else:
            sum_odd += f(x)

    xi = h / 3 * (f(a) + 2 * sum_even + 4 * sum_odd + f(b))
    return [xi]


def composite_trapezoidal(f, b, a, n):
    """Calculate the integral from the Trapezoidal Rule.

    Args:
        f: function f(x).
        a: the initial point.
        b: the final point.
        n: number of intervals.

    Returns:
        xi: integral value.
    """
    h = (b - a) / n

    sum_x = 0

    for i in range(0, n - 1):
        x = a + (i + 1) * h
        sum_x += f(x)

    xi = h / 2 * (f(a) + 2 * sum_x + f(b))
    return [xi]


def composite2_simpson(x, y):
    """Calculate the integral from 1/3 Simpson's Rule.

    Args:
        x: an array containing x values.
        y: an array containing y values.

    Returns:
        xi: integral value.
    """
    if y.size != y.size:
        raise ValueError("'x' and 'y' must have same size.")

    h = x[1] - x[0]
    n = x.size

    sum_odd = 0
    sum_even = 0

    for i in range(1, n - 1):
        if (i + 1) % 2 == 0:
            sum_even += y[i]
        else:
            sum_odd += y[i]

    xi = h / 3 * (y[0] + 2 * sum_even + 4 * sum_odd + y[n - 1])
    return [xi]


def composite2_trapezoidal(x, y):
    """Calculate the integral from the Trapezoidal Rule.

    Args:
        x: an array containing x values.
        y: an array containing y values.

    Returns:
        xi: integral value.
    """
    if y.size != y.size:
        raise ValueError("'x' and 'y' must have same size.")

    h = x[1] - x[0]
    n = x.size

    sum_x = 0

    for i in range(1, n - 1):
        sum_x += y[i]

    xi = h / 2 * (y[0] + 2 * sum_x + y[n - 1])
    return [xi]
