#include "matrix_free_delta_gamma.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mf = quassar::research04::matrix_free;

using quassar::research04::Matrix;
using quassar::research04::Vector;

void check(double actual, double expected, const char* label) {
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::abs(actual - expected) >
            1e-11 * (1.0 + std::abs(expected))) {
        throw std::runtime_error(label);
    }
}

int main() {
    // TEST 1: Exact analytical quadratic reference.
    const Matrix A = {
        {4.0, 1.0, -2.0},
        {1.0, 6.0, 0.5},
        {-2.0, 0.5, 8.0}
    };

    const Vector b = {1.5, -0.8, 2.0};
    const Vector point = {0.3, -0.7, 1.2};
    const Vector shock = {0.12, -0.09, 0.05};

    auto quadratic = [&](const auto& x) {
        auto result = mf::DirectionalJet::constant(
            0.0, x.size()
        );

        for (std::size_t i = 0; i < x.size(); ++i) {
            result = result + b[i] * x[i];

            for (std::size_t j = 0; j < x.size(); ++j) {
                result = result
                    + 0.5 * A[i][j] * x[i] * x[j];
            }
        }

        return result;
    };

    const auto result = mf::delta_gamma(
        quadratic, point, shock
    );

    double expected_delta = 0.0;
    double expected_gamma = 0.0;

    for (std::size_t i = 0; i < point.size(); ++i) {
        double gradient = b[i];

        for (std::size_t j = 0; j < point.size(); ++j) {
            gradient += A[i][j] * point[j];

            expected_gamma +=
                0.5 * shock[i] * A[i][j] * shock[j];
        }

        expected_delta += gradient * shock[i];
    }

    check(
        result.delta,
        expected_delta,
        "Analytical delta mismatch"
    );

    check(
        result.gamma,
        expected_gamma,
        "Analytical gamma mismatch"
    );

    check(
        result.total,
        expected_delta + expected_gamma,
        "Analytical total mismatch"
    );

    std::cout
        << "Quadratic analytical reference: PASSED\n";

    // TEST 2: Nonlinear model with independent derivatives.
    //
    // f(x,y) = exp(x*y) + sin(x) + log(y*y+2)
    //
    // Evaluate delta and gamma analytically.

    const Vector p = {0.4, -0.2};
    const Vector dx = {0.07, -0.05};

    auto nonlinear = [](const auto& x) {
        using mf::exp;
        using mf::log;
        using mf::sin;

        return exp(x[0] * x[1])
            + sin(x[0])
            + log(x[1] * x[1] + 2.0);
    };

    const auto nonlinear_result =
        mf::delta_gamma(nonlinear, p, dx);

    const double x = p[0];
    const double y = p[1];
    const double e = std::exp(x * y);
    const double denominator = y * y + 2.0;

    const double gx = y * e + std::cos(x);
    const double gy = x * e + 2.0 * y / denominator;

    const double hxx = y * y * e - std::sin(x);
    const double hxy = (1.0 + x * y) * e;
    const double hyy =
        x * x * e
        + 2.0 * (2.0 - y * y)
          / (denominator * denominator);

    const double analytical_delta =
        gx * dx[0] + gy * dx[1];

    const double analytical_gamma = 0.5 * (
        hxx * dx[0] * dx[0]
        + 2.0 * hxy * dx[0] * dx[1]
        + hyy * dx[1] * dx[1]
    );

    check(
        nonlinear_result.delta,
        analytical_delta,
        "Nonlinear analytical delta mismatch"
    );

    check(
        nonlinear_result.gamma,
        analytical_gamma,
        "Nonlinear analytical gamma mismatch"
    );

    check(
        nonlinear_result.total,
        analytical_delta + analytical_gamma,
        "Nonlinear analytical total mismatch"
    );

    std::cout
        << "Nonlinear analytical reference: PASSED\n";

    // TEST 3: Zero shock.
    const auto zero = mf::delta_gamma(
        quadratic, point, Vector{0.0, 0.0, 0.0}
    );

    check(zero.delta, 0.0, "Zero-shock delta");
    check(zero.gamma, 0.0, "Zero-shock gamma");
    check(zero.total, 0.0, "Zero-shock total");

    // TEST 4: Invalid dimensions.
    bool rejected = false;

    try {
        mf::delta_gamma(
            quadratic,
            point,
            Vector{0.1, 0.2}
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error(
            "Invalid shock dimensions accepted"
        );

    std::cout
        << "Zero shock and input validation: PASSED\n";

    std::cout
        << "RESEARCH 04: C++17 MATRIX-FREE DELTA-GAMMA VALIDATION PASSED\n";

    return 0;
}
