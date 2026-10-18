# Quassar Risk — Numerical Research Notes

## Scope
This initial research kernel is **not** a production risk management system. It operates on finite, arithmetic, dimensionless return observations and portfolio losses defined by `L = -wᵀR`. It does not ingest market data, account for liquidity, estimate dynamic volatility, or perform regulatory capital calculations.

## Estimators and conventions
- `covariance`: unbiased sample covariance, divisor `n−1`, observations in rows.
- `portfolio_variance`: quadratic form `wᵀΣw`; checks symmetry and negative quadratic forms but does **not** establish PSD for all directions.
- `historical_var_es`: nearest-rank empirical VaR at `ceil(nα)`, and expected shortfall as the mean of the worst `n(1−α)` observations with a fractional boundary weight. Losses may be negative, as may VaR and ES.
- `gaussian_var_es`: for loss `L ~ N(μ, σ²)`, `VaRα = μ + σ Φ⁻¹(α)` and `ESα = μ + σ φ(Φ⁻¹(α))/(1−α)`.
- `simulate_gaussian`: seeded pseudorandom independent multivariate normal vectors with a semidefinite Cholesky factor. The PRNG is **not** suitable for security-sensitive applications. Empirical simulation moments converge statistically, not exactly.

## Limitations and validation
Inputs must be finite; rows must be rectangular. This version does not support missing values, weighting, robust covariance, covariance shrinkage, backtesting, portfolio rebalancing, confidence intervals, or tail-model calibration. Numerical PSD tolerance is heuristic and must not be used as a substitute for an eigenvalue-based PSD check for ill-conditioned matrices. Unit tests check representative analytical results and error cases; further property-based, cross-library, performance, and adversarial validation is required before deployment.

## Running
From the repository root: `PYTHONPATH=src python -m unittest discover -s tests -v`.
