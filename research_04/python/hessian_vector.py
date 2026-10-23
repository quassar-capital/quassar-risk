"""Reference Hessian-vector products for second-order automatic differentiation."""

from __future__ import annotations

import math
from collections.abc import Sequence

try:
    from .autodiff import Jet
except ImportError:
    from autodiff import Jet


def hessian_vector_product(
    result: Jet,
    direction: Sequence[float],
) -> tuple[float, ...]:
    """Compute H @ v from an explicitly represented Hessian."""
    n = len(result.gradient)
    v = tuple(float(x) for x in direction)

    if len(v) != n:
        raise ValueError("Direction dimension mismatch")

    if not all(math.isfinite(x) for x in v):
        raise ValueError("Direction must contain finite values")

    return tuple(
        math.fsum(result.hessian[i][j] * v[j] for j in range(n))
        for i in range(n)
    )


def directional_curvature(
    result: Jet,
    direction: Sequence[float],
) -> float:
    """Compute the quadratic directional curvature v.T @ H @ v."""
    v = tuple(float(x) for x in direction)
    hv = hessian_vector_product(result, v)
    return math.fsum(v[i] * hv[i] for i in range(len(v)))
