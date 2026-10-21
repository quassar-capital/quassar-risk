"""Peaks-over-threshold extreme-value risk calculations.

The GPD shape parameter xi is restricted to 0 <= xi < 1 for
finite expected shortfall. Losses are represented as positive
quantities, and exceedance probabilities refer to P(L > u).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GPDModel:
    threshold: float
    scale: float
    shape: float
    exceedance_probability: float

    def __post_init__(self) -> None:
        values = (
            self.threshold,
            self.scale,
            self.shape,
            self.exceedance_probability,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("parameters must be finite")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if not 0 <= self.shape < 1:
            raise ValueError("shape must satisfy 0 <= shape < 1")
        if not 0 < self.exceedance_probability <= 1:
            raise ValueError("invalid exceedance probability")

    def survival(self, loss: float) -> float:
        """Unconditional tail probability P(L > loss), for loss >= u."""
        if not math.isfinite(loss) or loss < self.threshold:
            raise ValueError("loss must be finite and >= threshold")
        excess = loss - self.threshold
        if self.shape == 0:
            conditional = math.exp(-excess / self.scale)
        else:
            conditional = (1 + self.shape * excess / self.scale) ** (-1 / self.shape)
        return self.exceedance_probability * conditional

    def value_at_risk(self, confidence: float) -> float:
        """Tail VaR at confidence above the threshold quantile."""
        if not math.isfinite(confidence):
            raise ValueError("confidence must be finite")
        tail_probability = 1 - confidence
        threshold_confidence = 1 - self.exceedance_probability
        if (
            not 0 < tail_probability
            or confidence <= threshold_confidence
            or math.isclose(
                confidence,
                threshold_confidence,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("confidence must lie in the modeled tail")
        log_ratio = math.log(self.exceedance_probability / tail_probability)
        if self.shape == 0:
            return self.threshold + self.scale * log_ratio
        return self.threshold + (self.scale * math.expm1(self.shape * log_ratio) / self.shape)

    def expected_shortfall(self, confidence: float) -> float:
        """Conditional tail expectation for xi < 1."""
        var = self.value_at_risk(confidence)
        return (var + self.scale - self.shape * self.threshold) / (1 - self.shape)


def fit_exponential_tail(losses: list[float], threshold: float) -> GPDModel:
    """Fit the exponential GPD submodel (xi=0) by maximum likelihood.

    This is a constrained baseline estimator, not a general GPD fit.
    """
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not losses:
        raise ValueError("losses must be nonempty")
    if any(not math.isfinite(x) for x in losses):
        raise ValueError("losses must be finite")

    exceedances = [x - threshold for x in losses if x > threshold]
    if not exceedances:
        raise ValueError("no observations exceed the threshold")

    scale = math.fsum(exceedances) / len(exceedances)
    if scale <= 0:
        raise ValueError("invalid tail scale")

    return GPDModel(
        threshold=threshold,
        scale=scale,
        shape=0.0,
        exceedance_probability=len(exceedances) / len(losses),
    )
