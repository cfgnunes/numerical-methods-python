import numpy as np


def euler(f, a: float, b: float, n: int, ya: float) -> [np.array, np.array]:
    '''
    Calculate the solution of the initial-value problem from Euler method
    Inputs:
            f: Function f(x)
            a: Initial point
            b: End point
            n: Number of intervals
           ya: Initial value
    Outputs:
          vx: Array containing x values
          vy: Array containing y values (solution of IVP)
    '''

    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    vx[0] = x
    vy[0] = y

    fxy = f(x, y)
    print('i: %.3d\t x: %.4f\t y: %.4f\n' % (0, x, y), end="")

    for i in range(0, n):
        x = a + (i + 1) * h
        y += h * fxy

        fxy = f(x, y)
        print('i: %.3d\t x: %.4f\t y: %.4f\n' % (i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]


def rk4(f, a: float, b: float, n: int, ya: float) -> [np.array, np.array]:
    '''
    Calculate the solution of the initial-value problem from Runge-Kutta (Order Four) method
    Inputs:
            f: Function f(x)
            a: Initial point
            b: End point
            n: Number of intervals
           ya: Initial value
    Outputs:
          vx: Array containing x values
          vy: Array containing y values (solution of IVP)
    '''

    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    vx[0] = x
    vy[0] = y

    print('i: %.3d\t x:%.4f\t y:%.4f\t\n' % (0, x, y), end="")

    for i in range(0, n):
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)

        x = a + (i + 1) * h
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        print('i: %.3d\t x:%.4f\t y:%.4f\t\n' % (i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]


def taylor2(f, df1, a: float, b: float, n: int, ya: float) -> [np.array, np.array]:
    '''
    Calculate the solution of the initial-value problem from Taylor (Order Two) method
    Inputs:
            f: Function f(x)
          df1: 1's derivative of function f(x)
            a: Initial point
            b: End point
            n: Number of intervals
           ya: Initial value
    Outputs:
          vx: Array containing x values
          vy: Array containing y values (solution of IVP)
    '''

    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    vx[0] = x
    vy[0] = y

    print('i: %.3d\t x:%.4f\t y:%.4f\t\n' % (0, x, y), end="")

    for i in range(0, n):
        y += h * (f(x, y) + 0.5 * h * df1(x, y))
        x = a + (i + 1) * h

        print('i: %.3d\t x:%.4f\t y:%.4f\t\n' % (i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]


def taylor4(f, df1, df2, df3, a: float, b: float, n: int, ya: float) -> [np.array, np.array]:
    '''
    Calculate the solution of the initial-value problem from Taylor (Order Four) method
    Inputs:
            f: Function f(x)
          df1: 1's derivative of function f(x)
          df2: 2's derivative of function f(x)
          df3: 3's derivative of function f(x)
            a: Initial point
            b: End point
            n: Number of intervals
           ya: Initial value
    Outputs:
          vx: Array containing x values
          vy: Array containing y values (solution of IVP)
    '''

    vx = np.zeros(n)
    vy = np.zeros(n)

    h = (b - a) / n
    x = a
    y = ya

    vx[0] = x
    vy[0] = y

    print('i: %.3d\t x:%.4f\t y:%.4f\t\n' % (0, x, y), end="")

    for i in range(0, n):
        y += h * (f(x, y) + 0.5 * h * df1(x, y) + (h ** 2 / 6) * df2(x, y) + (h ** 3 / 24) * df3(x, y))
        x = a + (i + 1) * h

        print('i: %.3d\t x:%.4f\t y:%.4f\t\n' % (i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]
