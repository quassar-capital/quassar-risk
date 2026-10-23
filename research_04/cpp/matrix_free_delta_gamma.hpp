#ifndef QUASSAR_RESEARCH04_MATRIX_FREE_DELTA_GAMMA_HPP
#define QUASSAR_RESEARCH04_MATRIX_FREE_DELTA_GAMMA_HPP

#include "matrix_free_hvp.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quassar::research04::matrix_free {

struct DeltaGammaResult {
    double delta;
    double gamma;
    double total;
};

template <typename Model>
DeltaGammaResult delta_gamma(
    Model model,
    const Vector& point,
    const Vector& shock
) {
    const std::size_t n = point.size();

    if (n == 0 || shock.size() != n)
        throw std::invalid_argument(
            "Point and shock dimensions must agree"
        );

    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(point[i]) ||
            !std::isfinite(shock[i])) {
            throw std::invalid_argument(
                "Nonfinite point or shock"
            );
        }
    }

    std::vector<DirectionalJet> x;
    x.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        x.push_back(
            DirectionalJet::variable(
                point[i], i, shock
            )
        );
    }

    const DirectionalJet result = model(x);

    if (result.dimension() != n)
        throw std::invalid_argument(
            "Model output dimension mismatch"
        );

    // Compensated summation for linear and quadratic terms.
    double delta_sum = 0.0;
    double delta_comp = 0.0;
    double gamma_sum = 0.0;
    double gamma_comp = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double delta_term =
            result.gradient[i] * shock[i];

        const double gamma_term =
            0.5 * shock[i] * result.hvp[i];

        if (!std::isfinite(delta_term) ||
            !std::isfinite(gamma_term)) {
            throw std::overflow_error(
                "Nonfinite delta-gamma contribution"
            );
        }

        // Neumaier compensated accumulation.
        const double next_delta = delta_sum + delta_term;

        if (std::abs(delta_sum) >= std::abs(delta_term))
            delta_comp +=
                (delta_sum - next_delta) + delta_term;
        else
            delta_comp +=
                (delta_term - next_delta) + delta_sum;

        delta_sum = next_delta;

        const double next_gamma = gamma_sum + gamma_term;

        if (std::abs(gamma_sum) >= std::abs(gamma_term))
            gamma_comp +=
                (gamma_sum - next_gamma) + gamma_term;
        else
            gamma_comp +=
                (gamma_term - next_gamma) + gamma_sum;

        gamma_sum = next_gamma;
    }

    const double delta = delta_sum + delta_comp;
    const double gamma = gamma_sum + gamma_comp;
    const double total = delta + gamma;

    if (!std::isfinite(delta) ||
        !std::isfinite(gamma) ||
        !std::isfinite(total)) {
        throw std::overflow_error(
            "Nonfinite delta-gamma approximation"
        );
    }

    return {delta, gamma, total};
}

}  // namespace quassar::research04::matrix_free

#endif
