#ifndef QUASSAR_RESEARCH04_HESSIAN_VECTOR_HPP
#define QUASSAR_RESEARCH04_HESSIAN_VECTOR_HPP

#include "autodiff.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace quassar::research04 {

inline Vector hessian_vector_product(
    const Jet& result,
    const Vector& direction
) {
    const auto n = result.dimension();

    if (direction.size() != n)
        throw std::invalid_argument("Direction dimension mismatch");

    for (double v : direction) {
        if (!std::isfinite(v))
            throw std::invalid_argument("Nonfinite direction");
    }

    Vector hv(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        double compensation = 0.0;

        for (std::size_t j = 0; j < n; ++j) {
            const double term = result.hessian[i][j] * direction[j];

            if (!std::isfinite(term))
                throw std::overflow_error("Nonfinite HVP product");

            const double next = sum + term;

            if (std::abs(sum) >= std::abs(term))
                compensation += (sum - next) + term;
            else
                compensation += (term - next) + sum;

            sum = next;
        }

        hv[i] = sum + compensation;

        if (!std::isfinite(hv[i]))
            throw std::overflow_error("Nonfinite HVP result");
    }

    return hv;
}

inline double directional_curvature(
    const Jet& result,
    const Vector& direction
) {
    const Vector hv = hessian_vector_product(result, direction);

    double sum = 0.0;
    double compensation = 0.0;

    for (std::size_t i = 0; i < direction.size(); ++i) {
        const double term = direction[i] * hv[i];

        if (!std::isfinite(term))
            throw std::overflow_error("Nonfinite curvature product");

        const double next = sum + term;

        if (std::abs(sum) >= std::abs(term))
            compensation += (sum - next) + term;
        else
            compensation += (term - next) + sum;

        sum = next;
    }

    const double result_value = sum + compensation;

    if (!std::isfinite(result_value))
        throw std::overflow_error("Nonfinite directional curvature");

    return result_value;
}

}  // namespace quassar::research04

#endif
