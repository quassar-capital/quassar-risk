"""Regression tests for matrix-free Hessian-vector differentiation."""

import math
import sys
from pathlib import Path

import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

from autodiff import variables
from hessian_vector import hessian_vector_product
from matrix_free_hvp import directional_variables


def assert_close(actual, expected):
    assert math.isclose(
        actual, expected, rel_tol=1e-11, abs_tol=1e-11
    )


def nonlinear_model(x):
    return (
        (x[0] * x[1]).exp()
        + x[2] * x[3] * x[4]
        + (x[5] * x[0]).sin()
        + (x[1] * x[1] + 2).log()
        + x[3] / (x[4] * x[4] + 3)
    )


@pytest.mark.parametrize(
    "direction",
    [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, -0.5, 0.3, 0.0, 0.7),
        (0.2, -0.7, 1.1, 0.4, -0.3, 0.8),
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_matrix_free_matches_dense_reference(direction):
    values = (0.2, -0.4, 0.7, 1.1, -0.3, 0.8)

    dense = nonlinear_model(variables(values))
    free = nonlinear_model(directional_variables(values, direction))

    expected_hvp = hessian_vector_product(dense, direction)

    assert_close(free.value, dense.value)

    for actual, expected in zip(free.gradient, dense.gradient):
        assert_close(actual, expected)

    for actual, expected in zip(free.hvp, expected_hvp):
        assert_close(actual, expected)

    expected_tangent = math.fsum(
        g * v for g, v in zip(dense.gradient, direction)
    )
    assert_close(free.tangent, expected_tangent)


def test_independent_quadratic_analytical_reference():
    matrix = (
        (4.0, 1.0, -2.0),
        (1.0, 6.0, 0.5),
        (-2.0, 0.5, 8.0),
    )

    values = (0.3, -0.7, 1.2)
    direction = (2.0, -1.0, 0.5)

    x = directional_variables(values, direction)

    result = sum(
        0.5 * matrix[i][j] * x[i] * x[j]
        for i in range(3)
        for j in range(3)
    )

    expected_gradient = tuple(
        math.fsum(matrix[i][j] * values[j] for j in range(3))
        for i in range(3)
    )

    expected_hvp = tuple(
        math.fsum(matrix[i][j] * direction[j] for j in range(3))
        for i in range(3)
    )

    for actual, expected in zip(result.gradient, expected_gradient):
        assert_close(actual, expected)

    for actual, expected in zip(result.hvp, expected_hvp):
        assert_close(actual, expected)


def test_hessian_bilinear_symmetry():
    values = (0.2, -0.4, 0.7, 1.1, -0.3, 0.8)
    u = (1.0, -0.5, 0.2, 0.7, 0.0, -0.3)
    v = (-0.2, 0.8, 1.0, 0.0, 0.4, 0.6)

    hu = nonlinear_model(directional_variables(values, u)).hvp
    hv = nonlinear_model(directional_variables(values, v)).hvp

    u_h_v = math.fsum(a * b for a, b in zip(u, hv))
    v_h_u = math.fsum(a * b for a, b in zip(v, hu))

    assert_close(u_h_v, v_h_u)


def test_zero_direction_produces_zero_hvp():
    values = (0.2, -0.4, 0.7, 1.1, -0.3, 0.8)
    result = nonlinear_model(
        directional_variables(values, (0.0,) * 6)
    )

    assert result.tangent == 0.0
    assert all(value == 0.0 for value in result.hvp)


@pytest.mark.parametrize(
    "values,direction",
    [
        ((), ()),
        ((1.0, 2.0), (1.0,)),
        ((1.0,), (float("nan"),)),
        ((float("inf"),), (1.0,)),
    ],
)
def test_invalid_inputs_rejected(values, direction):
    with pytest.raises(ValueError):
        directional_variables(values, direction)


def test_log_domain_rejected():
    x, = directional_variables((-1.0,), (1.0,))

    with pytest.raises(ValueError):
        x.log()


def test_division_by_zero_rejected():
    x, = directional_variables((0.0,), (1.0,))

    with pytest.raises(ZeroDivisionError):
        x.reciprocal()
