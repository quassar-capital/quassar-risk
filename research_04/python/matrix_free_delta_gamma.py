"""Matrix-free second-order delta-gamma portfolio risk approximation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

try:
    from .matrix_free_hvp import DirectionalJet, directional_variables
except ImportError:
    from matrix_free_hvp import DirectionalJet, directional_variables


def matrix_free_delta_gamma(
    model: Callable,
    point: Sequence[float],
    shock: Sequence[float],
) -> dict[str, float]:
    """Evaluate a second-order portfolio P&L approximation.

    The model must accept a sequence of DirectionalJet variables
    and return one DirectionalJet scalar.

    The shock is also the direction used for the HVP.
    """
    x = tuple(float(v) for v in point)
    dx = tuple(float(v) for v in shock)

    if not x or len(x) != len(dx):
        raise ValueError("Point and shock dimensions must agree")

    if not all(math.isfinite(v) for v in (*x, *dx)):
        raise ValueError("Point and shock must contain finite values")

    variables = directional_variables(x, dx)
    result = model(variables)

    if not isinstance(result, DirectionalJet):
        raise TypeError("Model must return a DirectionalJet scalar")

    if len(result.gradient) != len(x):
        raise ValueError("Model output dimension mismatch")

    delta = math.fsum(
        g * v for g, v in zip(result.gradient, dx)
    )

    gamma = 0.5 * math.fsum(
        v * hv for v, hv in zip(dx, result.hvp)
    )

    total = math.fsum((delta, gamma))

    if not all(math.isfinite(v) for v in (delta, gamma, total)):
        raise ValueError("Nonfinite delta-gamma approximation")

    return {
        "delta": delta,
        "gamma": gamma,
        "total": total,
    }
