"""Reproducible scaling experiment for dense and matrix-free HVP."""

import gc
import importlib.util
import math
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "python"

for name in ("autodiff", "hessian_vector", "matrix_free_hvp"):
    spec = importlib.util.spec_from_file_location(
        name, BASE / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

from autodiff import variables
from hessian_vector import hessian_vector_product
from matrix_free_hvp import directional_variables


def model(x):
    # Nonlinear, separable terms with nearest-neighbor interactions.
    result = sum(
        (xi * xi + 1.0).log() + 0.1 * xi.sin()
        for xi in x
    )
    result += sum(
        0.05 * x[i] * x[i + 1]
        for i in range(len(x) - 1)
    )
    return result


def measure(fn):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    output = fn()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return output, elapsed, peak


def dense_run(values, direction):
    result = model(variables(values))
    return hessian_vector_product(result, direction)


def matrix_free_run(values, direction):
    result = model(directional_variables(values, direction))
    return result.hvp


print("RESEARCH 04 — HVP SCALING EXPERIMENT")
print("Dimension | Dense median (s) | Matrix-free median (s) | Dense peak (MiB) | Matrix-free peak (MiB)")

for n in (8, 16, 32, 64):
    values = tuple(0.1 + 0.01 * i for i in range(n))
    direction = tuple((-1.0 if i % 2 else 1.0) / math.sqrt(n) for i in range(n))

    results = {}

    for name, fn in (
        ("dense", dense_run),
        ("matrix_free", matrix_free_run),
    ):
        runs = [measure(lambda: fn(values, direction)) for _ in range(3)]
        results[name] = (
            runs[0][0],
            statistics.median(run[1] for run in runs),
            max(run[2] for run in runs),
        )

    dense_output, dense_time, dense_memory = results["dense"]
    free_output, free_time, free_memory = results["matrix_free"]

    for a, b in zip(dense_output, free_output):
        if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(f"HVP mismatch at dimension {n}")

    print(
        f"{n:9d} | "
        f"{dense_time:16.6f} | "
        f"{free_time:22.6f} | "
        f"{dense_memory / 1048576:16.3f} | "
        f"{free_memory / 1048576:22.3f}"
    )

print("RESEARCH 04: HVP SCALING EXPERIMENT PASSED")
