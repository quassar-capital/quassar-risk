"""Matrix-free Hessian-vector products via directional second-order AD.

Each scalar carries:
    value: scalar primal value
    gradient: full first derivative
    hvp: directional derivative of the gradient

The Hessian is never explicitly constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DirectionalJet:
    value: float
    gradient: tuple[float, ...]
    hvp: tuple[float, ...]
    tangent: float

    def __post_init__(self):
        n = len(self.gradient)
        if n == 0 or len(self.hvp) != n:
            raise ValueError("Inconsistent derivative dimensions")
        if not all(math.isfinite(x) for x in (
            self.value, self.tangent, *self.gradient, *self.hvp
        )):
            raise ValueError("Nonfinite value or derivative")

    @classmethod
    def variable(cls, value, index, dimension, direction):
        if dimension <= 0 or not 0 <= index < dimension:
            raise ValueError("Invalid variable index or dimension")
        if len(direction) != dimension:
            raise ValueError("Direction dimension mismatch")
        v = tuple(float(x) for x in direction)
        if not all(math.isfinite(x) for x in v):
            raise ValueError("Nonfinite direction")
        return cls(
            float(value),
            tuple(float(i == index) for i in range(dimension)),
            (0.0,) * dimension,
            v[index],
        )

    @classmethod
    def constant(cls, value, dimension):
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        return cls(
            float(value),
            (0.0,) * dimension,
            (0.0,) * dimension,
            0.0,
        )

    def _coerce(self, other):
        if isinstance(other, DirectionalJet):
            if len(other.gradient) != len(self.gradient):
                raise ValueError("Derivative dimension mismatch")
            return other
        return DirectionalJet.constant(other, len(self.gradient))

    def __add__(self, other):
        b = self._coerce(other)
        return DirectionalJet(
            self.value + b.value,
            tuple(a + c for a, c in zip(self.gradient, b.gradient)),
            tuple(a + c for a, c in zip(self.hvp, b.hvp)),
            self.tangent + b.tangent,
        )

    __radd__ = __add__

    def __neg__(self):
        return DirectionalJet(
            -self.value,
            tuple(-x for x in self.gradient),
            tuple(-x for x in self.hvp),
            -self.tangent,
        )

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        b = self._coerce(other)
        return DirectionalJet(
            self.value * b.value,
            tuple(
                a * b.value + self.value * c
                for a, c in zip(self.gradient, b.gradient)
            ),
            tuple(
                h * b.value
                + self.value * k
                + a * b.tangent
                + self.tangent * c
                for h, k, a, c in zip(
                    self.hvp, b.hvp, self.gradient, b.gradient
                )
            ),
            self.tangent * b.value + self.value * b.tangent,
        )

    __rmul__ = __mul__

    def compose(self, value, first, second):
        return DirectionalJet(
            value,
            tuple(first * g for g in self.gradient),
            tuple(
                first * h + second * self.tangent * g
                for h, g in zip(self.hvp, self.gradient)
            ),
            first * self.tangent,
        )

    def reciprocal(self):
        if self.value == 0:
            raise ZeroDivisionError("Division by zero")
        x = self.value
        return self.compose(1 / x, -1 / x**2, 2 / x**3)

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) * self.reciprocal()

    def exp(self):
        y = math.exp(self.value)
        return self.compose(y, y, y)

    def log(self):
        if self.value <= 0:
            raise ValueError("Logarithm requires a positive argument")
        x = self.value
        return self.compose(math.log(x), 1 / x, -1 / x**2)

    def sin(self):
        return self.compose(
            math.sin(self.value),
            math.cos(self.value),
            -math.sin(self.value),
        )

    def cos(self):
        return self.compose(
            math.cos(self.value),
            -math.sin(self.value),
            -math.cos(self.value),
        )


def directional_variables(values, direction):
    xs = tuple(float(x) for x in values)
    v = tuple(float(x) for x in direction)

    if not xs or len(xs) != len(v):
        raise ValueError("Invalid variable or direction dimensions")
    if not all(math.isfinite(x) for x in (*xs, *v)):
        raise ValueError("Nonfinite variable or direction")

    return tuple(
        DirectionalJet.variable(x, i, len(xs), v)
        for i, x in enumerate(xs)
    )
