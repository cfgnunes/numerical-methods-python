"""Methods for compute limits."""

import math


def limit_epsilon_delta(f, x, toler, iter_max):
    """Calculate a limit using the epsilon-delta definition.

    Args:
        f (function): equation f(x).
        x (float): the value the independent variable is approaching.
        toler (float): tolerance (stopping criterion).
        iter_max (int): maximum number of iterations (stopping criterion).

    Returns:
        limit (float): the limit value.
    """
    delta = 0.5
    limit_low_prev = -math.inf
    limit_up_prev = math.inf

    converged = False
    for i in range(0, iter_max + 1):
        delta /= (i + 1)
        limit_low = f(x - delta)
        limit_up = f(x + delta)

        if math.fabs(limit_low - limit_low_prev) <= toler \
           and math.fabs(limit_up - limit_up_prev) <= toler \
           and math.fabs(limit_up - limit_low) <= toler:
            converged = True
            break

        limit_up_prev = limit_up
        limit_low_prev = limit_low

    if math.fabs(limit_up - limit_low) > 10 * toler:
        raise ValueError("Two sided limit does not exist.")

    return limit_low, i, converged
