import numpy as np


def euler(f, a, b, n, ya):
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
    print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(0, x, y), end="")

    for i in range(0, n):
        x = a + (i + 1) * h
        y += h * fxy

        fxy = f(x, y)
        print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]


def taylor2(f, df1, a, b, n, ya):
    '''
    Calculate the solution of the initial-value problem from
    Taylor (Order Two) method.
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

    print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(0, x, y), end="")

    for i in range(0, n):
        y += h * (f(x, y) + 0.5 * h * df1(x, y))
        x = a + (i + 1) * h

        print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]


def taylor4(f, df1, df2, df3, a, b, n, ya):
    '''
    Calculate the solution of the initial-value problem from
    Taylor (Order Four) method.
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

    print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(0, x, y), end="")

    for i in range(0, n):
        y += h * (f(x, y) + 0.5 * h * df1(x, y) + (h ** 2 / 6) * df2(x, y) +
                  (h ** 3 / 24) * df3(x, y))
        x = a + (i + 1) * h

        print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]


def rk4(f, a, b, n, ya):
    '''
    Calculate the solution of the initial-value problem from
    Runge-Kutta (Order Four) method.
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

    print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(0, x, y), end="")

    for i in range(0, n):
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)

        x = a + (i + 1) * h
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        print("i: {:03d}\t x: {:.4f}\t y: {:.4f}\n".format(i + 1, x, y), end="")
        vx[i] = x
        vy[i] = y

    return [vx, vy]


def rk4_system(f, a, b, n, ya):
    '''
    Calculate the solution of systems of differential equations from
    Runge-Kutta (Order Four) method.
    Inputs:
            f: Array of functions f(x)
            a: Initial point
            b: End point
            n: Number of intervals
           ya: Array of initial values
    Outputs:
          vx: Array containing x values
          vy: Array containing y values (solution of IVP)
    '''

    m = len(f)

    k1 = np.zeros(m)
    k2 = np.zeros(m)
    k3 = np.zeros(m)
    k4 = np.zeros(m)

    vx = np.zeros(n + 1)
    vy = np.zeros((m, n + 1))

    h = (b - a) / n

    x = a
    y = ya

    vx[0] = x
    vy[:, 0] = y

    for i in range(0, n):

        for j in range(0, m):
            k1[j] = h * f[j](x, y)

        for j in range(0, m):
            k2[j] = h * f[j](x + h / 2, y + k1 / 2)

        for j in range(0, m):
            k3[j] = h * f[j](x + h / 2, y + k2 / 2)

        for j in range(0, m):
            k4[j] = h * f[j](x + h, y + k3)

        x = a + i * h
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        vx[i + 1] = x
        vy[:, i + 1] = y

    return [vx, vy]
