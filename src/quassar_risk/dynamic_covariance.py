"""Exponentially weighted dynamic second-moment estimation.

The input observations are assumed to be centered returns.
The update preserves positive semidefiniteness when initialized
with a positive-semidefinite matrix and 0 <= decay <= 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _is_positive_semidefinite(
    matrix: list[list[float]],
) -> bool:
    """Check PSD using a pivot-free LDL decomposition with zero-pivot checks."""
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    diagonal = [0.0] * n
    scale = max(1.0, max(abs(v) for row in matrix for v in row))
    tolerance = 1e-12 * scale

    for i in range(n):
        lower[i][i] = 1.0
        pivot = matrix[i][i] - math.fsum(lower[i][k] ** 2 * diagonal[k] for k in range(i))
        if pivot < -tolerance:
            return False
        if abs(pivot) <= tolerance:
            diagonal[i] = 0.0
            for j in range(i + 1, n):
                residual = matrix[j][i] - math.fsum(
                    lower[j][k] * lower[i][k] * diagonal[k] for k in range(i)
                )
                if abs(residual) > tolerance:
                    return False
        else:
            diagonal[i] = pivot
            for j in range(i + 1, n):
                residual = matrix[j][i] - math.fsum(
                    lower[j][k] * lower[i][k] * diagonal[k] for k in range(i)
                )
                lower[j][i] = residual / pivot
    return True


def ewma_covariance(
    returns: Sequence[Sequence[float]],
    decay: float = 0.94,
    initial_covariance: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    """Compute an EWMA covariance estimate for centered returns."""
    if not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError("decay must lie between zero and one")
    if not returns:
        raise ValueError("returns must not be empty")

    observations = [list(row) for row in returns]
    dimension = len(observations[0])

    if dimension == 0:
        raise ValueError("returns must have at least one asset")

    for row in observations:
        if len(row) != dimension:
            raise ValueError("returns must be rectangular")
        if any(not math.isfinite(value) for value in row):
            raise ValueError("returns must be finite")

    if initial_covariance is None:
        covariance = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    else:
        covariance = [list(row) for row in initial_covariance]

        if len(covariance) != dimension or any(len(row) != dimension for row in covariance):
            raise ValueError("initial covariance dimensions mismatch")

        for i in range(dimension):
            for j in range(dimension):
                value = covariance[i][j]
                if not math.isfinite(value):
                    raise ValueError("initial covariance must be finite")
                if not math.isclose(
                    value,
                    covariance[j][i],
                    rel_tol=1e-12,
                    abs_tol=1e-14,
                ):
                    raise ValueError("initial covariance must be symmetric")

    if not _is_positive_semidefinite(covariance):
        raise ValueError("initial covariance must be positive semidefinite")

    for observation in observations:
        covariance = [
            [
                decay * covariance[i][j] + (1.0 - decay) * observation[i] * observation[j]
                for j in range(dimension)
            ]
            for i in range(dimension)
        ]

    return covariance
