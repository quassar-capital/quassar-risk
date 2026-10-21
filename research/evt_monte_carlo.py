"""Reproducible Monte Carlo validation of exponential-tail risk measures."""

import math
import random

from quassar_risk.extreme_value import GPDModel


def main() -> None:
    rng = random.Random(20261020)

    sample_size = 100_000
    confidence = 0.99
    scale = 2.0

    model = GPDModel(
        threshold=0.0,
        scale=scale,
        shape=0.0,
        exceedance_probability=1.0,
    )

    observations = sorted(-scale * math.log1p(-rng.random()) for _ in range(sample_size))

    index = math.ceil(confidence * sample_size) - 1
    empirical_var = observations[index]

    tail = observations[index:]
    empirical_es = math.fsum(tail) / len(tail)

    theoretical_var = model.value_at_risk(confidence)
    theoretical_es = model.expected_shortfall(confidence)

    var_relative_error = abs(empirical_var - theoretical_var) / theoretical_var

    es_relative_error = abs(empirical_es - theoretical_es) / theoretical_es

    print("QUASSAR RISK — EVT MONTE CARLO VALIDATION")
    print(f"Sample size: {sample_size:,}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Theoretical VaR: {theoretical_var:.6f}")
    print(f"Empirical VaR:   {empirical_var:.6f}")
    print(f"VaR relative error: {var_relative_error:.4%}")
    print(f"Theoretical ES:  {theoretical_es:.6f}")
    print(f"Empirical ES:    {empirical_es:.6f}")
    print(f"ES relative error: {es_relative_error:.4%}")

    if var_relative_error >= 0.03:
        raise AssertionError("VaR Monte Carlo error exceeds 3%")

    if es_relative_error >= 0.03:
        raise AssertionError("ES Monte Carlo error exceeds 3%")

    print("MONTE CARLO VALIDATION PASSED")


if __name__ == "__main__":
    main()
