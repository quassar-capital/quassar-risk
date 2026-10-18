"""Portfolio loss risk primitives. Returns are arithmetic, dimensionless observations.

Conventions: portfolio loss L = -w^T R, where R is a vector of asset returns.
Risk values are in return units, not currency; weights may represent leverage.
"""
from __future__ import annotations
import math
import random
from statistics import NormalDist
from typing import Sequence


def _finite(x: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (float, int)):
        raise TypeError("inputs must be real numbers")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError("inputs must be finite")
    return v


def _vector(xs: Sequence[float]) -> list[float]:
    values = [_finite(x) for x in xs]
    if not values:
        raise ValueError("empty vector")
    return values


def _rows(rows: Sequence[Sequence[float]], *, min_rows: int = 1) -> list[list[float]]:
    data = [_vector(row) for row in rows]
    if len(data) < min_rows:
        raise ValueError("insufficient observations")
    n = len(data[0])
    if any(len(row) != n for row in data):
        raise ValueError("ragged observations")
    return data


def _confidence(alpha: float) -> float:
    a = _finite(alpha)
    if not 0 < a < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    return a


def covariance(returns: Sequence[Sequence[float]]) -> list[list[float]]:
    """Unbiased sample covariance; rows=dates, columns=assets; n >= 2."""
    data = _rows(returns, min_rows=2)
    n, d = len(data), len(data[0])
    means = [math.fsum(row[j] for row in data) / n for j in range(d)]
    centered = [[row[j] - means[j] for j in range(d)] for row in data]
    return [[math.fsum(row[i] * row[j] for row in centered) / (n - 1)
             for j in range(d)] for i in range(d)]


def portfolio_variance(weights: Sequence[float], cov: Sequence[Sequence[float]]) -> float:
    """w' Σ w with finite, symmetric covariance; small negative roundoff clipped."""
    w = _vector(weights)
    s = _rows(cov)
    d = len(w)
    if len(s) != d or any(len(row) != d for row in s):
        raise ValueError("covariance dimensions mismatch")
    for i in range(d):
        for j in range(d):
            if not math.isclose(s[i][j], s[j][i], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("covariance must be symmetric")
    v = math.fsum(w[i] * s[i][j] * w[j] for i in range(d) for j in range(d))
    scale = math.fsum(abs(w[i] * s[i][j] * w[j]) for i in range(d) for j in range(d))
    if v < -1e-12 * max(scale, 1e-300):
        raise ValueError("negative portfolio variance: covariance may not be PSD")
    return max(0.0, v)


def historical_var_es(losses: Sequence[float], confidence: float = 0.975) -> tuple[float, float]:
    """Empirical VaR and ES of losses using the discrete upper-tail convention.

    VaR = sorted loss at index ceil(n*alpha)-1; ES averages worst (1-alpha)
    fraction, allowing a fractional observation at the quantile boundary.
    This is an empirical quantile convention, not interpolated quantiles.
    """
    a = _confidence(confidence)
    x = sorted(_vector(losses))
    n = len(x)
    k = max(0, math.ceil(n * a) - 1)
    var = x[k]
    tail_count = n * (1 - a)
    full = int(math.floor(tail_count))
    fraction = tail_count - full
    tail_sum = math.fsum(x[n-full:]) if full else 0.0
    if fraction > 0:
        tail_sum += fraction * x[n-full-1]
    return var, tail_sum / tail_count


def gaussian_var_es(mean_loss: float, std_loss: float, confidence: float = 0.975) -> tuple[float, float]:
    """Analytical Gaussian loss VaR/ES; standard deviation must be nonnegative."""
    a = _confidence(confidence)
    mu, sigma = _finite(mean_loss), _finite(std_loss)
    if sigma < 0:
        raise ValueError("standard deviation cannot be negative")
    z = NormalDist().inv_cdf(a)
    return mu + sigma*z, mu + sigma*math.exp(-z*z/2)/math.sqrt(2*math.pi)/(1-a)


def _cholesky_psd(cov: list[list[float]]) -> list[list[float]]:
    """Semidefinite Cholesky with tolerance; rejects non-PSD matrices."""
    d = len(cov)
    scale = max(1.0, max(abs(x) for row in cov for x in row))
    tol = 1e-12 * scale
    lower = [[0.0]*d for _ in range(d)]
    for i in range(d):
        for j in range(i+1):
            rem = cov[i][j] - math.fsum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if rem < -tol:
                    raise ValueError("covariance is not positive semidefinite")
                lower[i][j] = math.sqrt(max(0.0, rem))
            elif lower[j][j] > math.sqrt(tol)*1e-6:
                lower[i][j] = rem/lower[j][j]
            elif abs(rem) > tol:
                raise ValueError("covariance is not positive semidefinite")
    return lower


def simulate_gaussian(mean: Sequence[float], cov: Sequence[Sequence[float]],
                      draws: int, seed: int) -> list[list[float]]:
    """Reproducible Gaussian draws (local PRNG, not cryptographically secure).

    Covariance is population covariance for simulated innovations. No temporal
    dynamics or volatility clustering are represented.
    """
    mu = _vector(mean)
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    s = _rows(cov)
    d = len(mu)
    if len(s) != d or any(len(row) != d for row in s):
        raise ValueError("covariance dimensions mismatch")
    for i in range(d):
        for j in range(d):
            if not math.isclose(s[i][j], s[j][i], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("covariance must be symmetric")
    lower = _cholesky_psd(s)
    rng = random.Random(seed)
    return [[mu[i] + math.fsum(lower[i][k]*z[k] for k in range(i+1))
             for i in range(d)] for z in ([rng.gauss(0, 1) for _ in range(d)] for _ in range(draws))]
