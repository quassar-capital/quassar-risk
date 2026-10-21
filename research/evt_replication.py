"""Independent Monte Carlo replication study for exponential EVT."""

import math
import random
import statistics

from quassar_risk.extreme_value import GPDModel


def main() -> None:
    model = GPDModel(
        threshold=0.0,
        scale=2.0,
        shape=0.0,
        exceedance_probability=1.0,
    )

    confidence = 0.99
    sample_size = 50_000
    replications = 30

    theoretical_var = model.value_at_risk(confidence)
    theoretical_es = model.expected_shortfall(confidence)

    var_errors = []
    es_errors = []

    for replication in range(replications):
        rng = random.Random(20261020 + replication)

        observations = sorted(-2.0 * math.log1p(-rng.random()) for _ in range(sample_size))

        index = math.ceil(confidence * sample_size) - 1
        empirical_var = observations[index]
        empirical_es = math.fsum(observations[index:]) / (sample_size - index)

        var_errors.append((empirical_var - theoretical_var) / theoretical_var)
        es_errors.append((empirical_es - theoretical_es) / theoretical_es)

    print("QUASSAR RISK — EVT REPLICATION STUDY")
    print(f"Independent replications: {replications}")
    print(f"Observations per replication: {sample_size:,}")

    for name, errors in (
        ("VaR", var_errors),
        ("Expected Shortfall", es_errors),
    ):
        mean_error = statistics.mean(errors)
        standard_error = statistics.stdev(errors) / math.sqrt(replications)
        maximum_error = max(abs(error) for error in errors)

        print(f"\n{name}")
        print(f"Mean relative error: {mean_error:.4%}")
        print(f"Standard error of mean: {standard_error:.4%}")
        print(f"Maximum absolute relative error: {maximum_error:.4%}")

        if abs(mean_error) >= 0.03:
            raise AssertionError(f"{name} mean relative error exceeds 3%")

    print("\nREPLICATION STUDY PASSED")


if __name__ == "__main__":
    main()
