"""Regression tests for dense and matrix-free delta-gamma risk."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "python")
)

from autodiff import variables
from delta_gamma import (
    delta_gamma_approximation,
    delta_gamma_decomposition,
)
from matrix_free_delta_gamma import matrix_free_delta_gamma


def close(actual, expected):
    assert math.isclose(
        actual, expected, rel_tol=1e-11, abs_tol=1e-11
    )


A = (
    (4.0, 1.0, -2.0),
    (1.0, 6.0, 0.5),
    (-2.0, 0.5, 8.0),
)
B = (1.5, -0.8, 2.0)
POINT = (0.3, -0.7, 1.2)


def quadratic(x):
    return (
        sum(B[i] * x[i] for i in range(3))
        + sum(
            0.5 * A[i][j] * x[i] * x[j]
            for i in range(3)
            for j in range(3)
        )
    )


def nonlinear(x):
    return (
        (x[0] * x[1]).exp()
        + x[2] * x[3] * x[4]
        + (x[5] * x[0]).sin()
        + (x[1] * x[1] + 2).log()
        + x[3] / (x[4] * x[4] + 3)
    )


@pytest.mark.parametrize(
    "shock",
    [
        (0.12, -0.09, 0.05),
        (0.0, 0.0, 0.0),
        (-0.2, 0.1, 0.3),
        (1.0, -1.0, 0.5),
    ],
)
def test_quadratic_exactness(shock):
    dense = delta_gamma_decomposition(
        quadratic(variables(POINT)), shock
    )
    free = matrix_free_delta_gamma(quadratic, POINT, shock)

    shifted = tuple(a + b for a, b in zip(POINT, shock))
    exact = quadratic(shifted) - quadratic(POINT)

    close(dense["total"], exact)
    close(free["total"], exact)

    for key in ("delta", "gamma", "total"):
        close(dense[key], free[key])


def test_analytical_delta_and_gamma():
    shock = (0.12, -0.09, 0.05)

    expected_delta = math.fsum(
        (
            B[i] + math.fsum(
                A[i][j] * POINT[j] for j in range(3)
            )
        ) * shock[i]
        for i in range(3)
    )

    expected_gamma = 0.5 * math.fsum(
        shock[i] * A[i][j] * shock[j]
        for i in range(3)
        for j in range(3)
    )

    result = matrix_free_delta_gamma(quadratic, POINT, shock)

    close(result["delta"], expected_delta)
    close(result["gamma"], expected_gamma)
    close(result["total"], expected_delta + expected_gamma)


def test_cross_factor_gamma():
    shock = (0.12, -0.09, 0.05)
    result = delta_gamma_decomposition(
        quadratic(variables(POINT)), shock
    )

    expected = math.fsum(
        A[i][j] * shock[i] * shock[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )

    close(result["gamma_cross"], expected)
    close(
        result["gamma"],
        result["gamma_diagonal"] + result["gamma_cross"],
    )


@pytest.mark.parametrize(
    "shock",
    [
        (0.1, -0.05, 0.02, 0.07, 0.0, -0.03),
        (-0.02, 0.08, 0.1, 0.0, 0.04, 0.06),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_nonlinear_dense_matrix_free_agreement(shock):
    point = (0.2, -0.4, 0.7, 1.1, -0.3, 0.8)

    dense = delta_gamma_decomposition(
        nonlinear(variables(point)), shock
    )
    free = matrix_free_delta_gamma(nonlinear, point, shock)

    for key in ("delta", "gamma", "total"):
        close(dense[key], free[key])


def test_approximation_matches_decomposition():
    shock = (0.12, -0.09, 0.05)
    result = quadratic(variables(POINT))

    close(
        delta_gamma_approximation(result, shock),
        delta_gamma_decomposition(result, shock)["total"],
    )


@pytest.mark.parametrize(
    "shock",
    [
        (0.1,),
        (0.1, 0.2),
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
    ],
)
def test_dense_invalid_shocks(shock):
    result = quadratic(variables(POINT))

    with pytest.raises(ValueError):
        delta_gamma_approximation(result, shock)


@pytest.mark.parametrize(
    "point,shock",
    [
        ((1.0, 2.0), (0.1,)),
        ((), ()),
        ((float("nan"),), (0.1,)),
        ((1.0,), (float("inf"),)),
    ],
)
def test_matrix_free_invalid_inputs(point, shock):
    with pytest.raises(ValueError):
        matrix_free_delta_gamma(quadratic, point, shock)
