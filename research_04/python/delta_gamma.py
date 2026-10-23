"""Second-order delta-gamma portfolio risk approximation."""

from __future__ import annotations

import math
from collections.abc import Sequence

try:
    from .autodiff import Jet
except ImportError:
    from autodiff import Jet


def delta_gamma_decomposition(
    result: Jet,
    shock: Sequence[float],
) -> dict[str, float]:
    """Decompose second-order portfolio value changes.

    The approximation is evaluated at the differentiation point
    represented by `result`. Shock coordinates must use the same
    factor units and ordering as the original variables.
    """
    dx = tuple(float(x) for x in shock)
    n = len(result.gradient)

    if len(dx) != n:
        raise ValueError("Shock dimension mismatch")

    if not all(math.isfinite(x) for x in dx):
        raise ValueError("Shock must contain finite values")

    delta = math.fsum(
        result.gradient[i] * dx[i]
        for i in range(n)
    )

    gamma_diagonal = 0.5 * math.fsum(
        result.hessian[i][i] * dx[i] * dx[i]
        for i in range(n)
    )

    gamma_cross = math.fsum(
        0.5
        * (result.hessian[i][j] + result.hessian[j][i])
        * dx[i]
        * dx[j]
        for i in range(n)
        for j in range(i + 1, n)
    )

    gamma = math.fsum((gamma_diagonal, gamma_cross))
    total = math.fsum((delta, gamma))

    if not all(math.isfinite(x) for x in (
        delta, gamma_diagonal, gamma_cross, gamma, total
    )):
        raise ValueError("Nonfinite delta-gamma approximation")

    return {
        "delta": delta,
        "gamma_diagonal": gamma_diagonal,
        "gamma_cross": gamma_cross,
        "gamma": gamma,
        "total": total,
    }


def delta_gamma_approximation(
    result: Jet,
    shock: Sequence[float],
) -> float:
    """Return the second-order approximation to the portfolio P&L."""
    return delta_gamma_decomposition(result, shock)["total"]
