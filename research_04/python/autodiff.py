"""Second-order forward-mode automatic differentiation.

Dense reference implementation for scalar functions of multiple variables.
Derivatives are propagated analytically through supported operations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Jet:
    value: float
    gradient: tuple[float, ...]
    hessian: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        n = len(self.gradient)
        if len(self.hessian) != n or any(len(row) != n for row in self.hessian):
            raise ValueError("Inconsistent derivative dimensions")
        if not all(math.isfinite(v) for v in (
            self.value,
            *self.gradient,
            *(v for row in self.hessian for v in row),
        )):
            raise ValueError("Nonfinite value or derivative")

    @classmethod
    def variable(cls, value: float, index: int, dimension: int) -> Jet:
        if dimension <= 0 or not 0 <= index < dimension:
            raise ValueError("Invalid variable index or dimension")
        return cls(
            float(value),
            tuple(float(i == index) for i in range(dimension)),
            tuple(tuple(0.0 for _ in range(dimension)) for _ in range(dimension)),
        )

    @classmethod
    def constant(cls, value: float, dimension: int) -> Jet:
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        return cls(
            float(value),
            (0.0,) * dimension,
            tuple((0.0,) * dimension for _ in range(dimension)),
        )

    def _coerce(self, other):
        if isinstance(other, Jet):
            if len(other.gradient) != len(self.gradient):
                raise ValueError("Derivative dimension mismatch")
            return other
        return Jet.constant(float(other), len(self.gradient))

    def __add__(self, other):
        other = self._coerce(other)
        n = len(self.gradient)
        return Jet(
            self.value + other.value,
            tuple(self.gradient[i] + other.gradient[i] for i in range(n)),
            tuple(tuple(
                self.hessian[i][j] + other.hessian[i][j]
                for j in range(n)
            ) for i in range(n)),
        )

    __radd__ = __add__

    def __neg__(self):
        return Jet(
            -self.value,
            tuple(-v for v in self.gradient),
            tuple(tuple(-v for v in row) for row in self.hessian),
        )

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) + (-self)

    def __mul__(self, other):
        other = self._coerce(other)
        n = len(self.gradient)
        return Jet(
            self.value * other.value,
            tuple(
                self.gradient[i] * other.value
                + self.value * other.gradient[i]
                for i in range(n)
            ),
            tuple(tuple(
                self.hessian[i][j] * other.value
                + self.value * other.hessian[i][j]
                + self.gradient[i] * other.gradient[j]
                + other.gradient[i] * self.gradient[j]
                for j in range(n)
            ) for i in range(n)),
        )

    __rmul__ = __mul__

    def compose(self, value: float, first: float, second: float) -> Jet:
        """Apply a scalar function using its first two derivatives."""
        n = len(self.gradient)
        return Jet(
            value,
            tuple(first * g for g in self.gradient),
            tuple(tuple(
                second * self.gradient[i] * self.gradient[j]
                + first * self.hessian[i][j]
                for j in range(n)
            ) for i in range(n)),
        )

    def reciprocal(self) -> Jet:
        if self.value == 0.0:
            raise ZeroDivisionError("Division by zero")
        x = self.value
        return self.compose(1.0 / x, -1.0 / x**2, 2.0 / x**3)

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) * self.reciprocal()

    def exp(self) -> Jet:
        y = math.exp(self.value)
        return self.compose(y, y, y)

    def log(self) -> Jet:
        if self.value <= 0:
            raise ValueError("Logarithm requires a positive argument")
        x = self.value
        return self.compose(math.log(x), 1.0 / x, -1.0 / x**2)

    def sin(self) -> Jet:
        return self.compose(
            math.sin(self.value),
            math.cos(self.value),
            -math.sin(self.value),
        )

    def cos(self) -> Jet:
        return self.compose(
            math.cos(self.value),
            -math.sin(self.value),
            -math.cos(self.value),
        )


def variables(values):
    """Create independent variables for a multivariate scalar function."""
    xs = tuple(float(v) for v in values)
    if not xs:
        raise ValueError("At least one variable is required")
    return tuple(Jet.variable(x, i, len(xs)) for i, x in enumerate(xs))
