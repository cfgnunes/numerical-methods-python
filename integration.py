def composite_simpson(f, b, a, n):
    '''
    Calculate the integral from 1/3 Simpson's Rule
    Inputs:
            f: Function f(x)
            a: Initial point
            b: End point
            n: Number of intervals
    Outputs:
           xi: Integral value
    '''

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
    '''
    Calculate the integral from Trapezoidal Rule
    Inputs:
            f: Function f(x)
            a: Initial point
            b: End point
            n: Number of intervals
    Outputs:
           xi: Integral value
    '''

    h = (b - a) / n

    sum_x = 0

    for i in range(0, n - 1):
        x = a + (i + 1) * h
        sum_x += f(x)

    xi = h / 2 * (f(a) + 2 * sum_x + f(b))
    return [xi]


def composite2_simpson(x, y):
    '''
    Calculate the integral from 1/3 Simpson's Rule
    Inputs:
            x: Array containing x values
            y: Array containing y values
    Outputs:
           xi: Integral value
    '''

    if y.size != y.size:
        raise "Error: 'x' and 'y' must have same size."

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
    '''
    Calculate the integral from Trapezoidal Rule
    Inputs:
            x: Array containing x values
            y: Array containing y values
    Outputs:
           xi: Integral value
    '''

    if y.size != y.size:
        raise "Error: 'x' and 'y' must have same size."

    h = x[1] - x[0]
    n = x.size

    sum_x = 0

    for i in range(1, n - 1):
        sum_x += y[i]

    xi = h / 2 * (y[0] + 2 * sum_x + y[n - 1])
    return [xi]
