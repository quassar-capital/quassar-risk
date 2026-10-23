#include "autodiff.hpp"
#include "hessian_vector.hpp"
#include "matrix_free_hvp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dense = quassar::research04;
namespace mf = quassar::research04::matrix_free;

void check(double actual, double expected, const char* label) {
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::abs(actual - expected) >
            1e-11 * (1.0 + std::abs(expected))) {
        throw std::runtime_error(label);
    }
}

template <typename T>
T nonlinear_model(const std::vector<T>& x) {
    using std::exp;
    using std::log;
    using std::sin;

    // AD overloads are explicitly available through argument-dependent lookup.
    return exp(x[0] * x[1])
         + x[2] * x[3] * x[4]
         + sin(x[5] * x[0])
         + log(x[1] * x[1] + 2.0)
         + x[3] / (x[4] * x[4] + 3.0);
}

int main() {
    const dense::Vector point = {
        0.2, -0.4, 0.7, 1.1, -0.3, 0.8
    };

    const std::vector<dense::Vector> directions = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, -0.5, 0.3, 0.0, 0.7},
        {0.2, -0.7, 1.1, 0.4, -0.3, 0.8},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };

    std::vector<dense::Jet> dense_x;

    for (std::size_t i = 0; i < point.size(); ++i) {
        dense_x.push_back(
            dense::Jet::variable(point[i], i, point.size())
        );
    }

    const dense::Jet reference = nonlinear_model(dense_x);

    for (const auto& direction : directions) {
        std::vector<mf::DirectionalJet> free_x;

        for (std::size_t i = 0; i < point.size(); ++i) {
            free_x.push_back(
                mf::DirectionalJet::variable(
                    point[i], i, direction
                )
            );
        }

        const auto result = nonlinear_model(free_x);
        const auto expected_hvp =
            dense::hessian_vector_product(reference, direction);

        check(result.value, reference.value, "Value mismatch");

        double expected_tangent = 0.0;

        for (std::size_t i = 0; i < point.size(); ++i) {
            check(
                result.gradient[i],
                reference.gradient[i],
                "Gradient mismatch"
            );

            check(
                result.hvp[i],
                expected_hvp[i],
                "HVP mismatch"
            );

            expected_tangent +=
                reference.gradient[i] * direction[i];
        }

        check(
            result.tangent,
            expected_tangent,
            "Directional derivative mismatch"
        );
    }

    // Independent analytical quadratic reference.
    const dense::Matrix A = {
        {4.0, 1.0, -2.0},
        {1.0, 6.0, 0.5},
        {-2.0, 0.5, 8.0}
    };

    const dense::Vector qpoint = {0.3, -0.7, 1.2};
    const dense::Vector direction = {2.0, -1.0, 0.5};

    std::vector<mf::DirectionalJet> qx;

    for (std::size_t i = 0; i < qpoint.size(); ++i) {
        qx.push_back(
            mf::DirectionalJet::variable(
                qpoint[i], i, direction
            )
        );
    }

    auto quadratic =
        mf::DirectionalJet::constant(0.0, qpoint.size());

    for (std::size_t i = 0; i < qpoint.size(); ++i) {
        for (std::size_t j = 0; j < qpoint.size(); ++j) {
            quadratic = quadratic
                + 0.5 * A[i][j] * qx[i] * qx[j];
        }
    }

    for (std::size_t i = 0; i < qpoint.size(); ++i) {
        double expected_gradient = 0.0;
        double expected_hvp = 0.0;

        for (std::size_t j = 0; j < qpoint.size(); ++j) {
            expected_gradient += A[i][j] * qpoint[j];
            expected_hvp += A[i][j] * direction[j];
        }

        check(
            quadratic.gradient[i],
            expected_gradient,
            "Analytical gradient mismatch"
        );

        check(
            quadratic.hvp[i],
            expected_hvp,
            "Analytical HVP mismatch"
        );
    }

    // Invalid direction validation.
    bool rejected = false;

    try {
        mf::DirectionalJet::variable(
            1.0, 2, dense::Vector{1.0, 2.0}
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected) {
        throw std::runtime_error("Invalid index accepted");
    }

    std::cout
        << "RESEARCH 04: C++17 MATRIX-FREE HVP VALIDATION PASSED\n"
        << "Dense cross-validation, analytical derivatives, "
        << "five directions and input checks verified.\n";

    return 0;
}
