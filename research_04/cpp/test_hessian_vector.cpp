#include "hessian_vector.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace quassar::research04;

void check(double actual, double expected, const char* label) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > 1e-11) {
        throw std::runtime_error(label);
    }
}

int main() {
    const Matrix A = {
        {4.0, 1.0, -2.0},
        {1.0, 6.0, 0.5},
        {-2.0, 0.5, 8.0}
    };

    const Vector point = {0.3, -0.7, 1.2};
    const Vector direction = {2.0, -1.0, 0.5};

    std::vector<Jet> x;

    for (std::size_t i = 0; i < point.size(); ++i)
        x.push_back(Jet::variable(point[i], i, point.size()));

    Jet f = Jet::constant(0.0, point.size());

    for (std::size_t i = 0; i < point.size(); ++i) {
        for (std::size_t j = 0; j < point.size(); ++j) {
            f = f + 0.5 * A[i][j] * x[i] * x[j];
        }
    }

    const Vector hv = hessian_vector_product(f, direction);

    double expected_curvature = 0.0;

    for (std::size_t i = 0; i < point.size(); ++i) {
        double expected = 0.0;

        for (std::size_t j = 0; j < point.size(); ++j)
            expected += A[i][j] * direction[j];

        check(hv[i], expected, "HVP analytical mismatch");
        expected_curvature += direction[i] * expected;
    }

    check(
        directional_curvature(f, direction),
        expected_curvature,
        "Directional curvature mismatch"
    );

    // Verify rejection of incompatible dimensions.
    bool rejected = false;

    try {
        hessian_vector_product(f, Vector{1.0, 2.0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("Dimension mismatch accepted");

    // Verify rejection of nonfinite directions.
    rejected = false;

    try {
        hessian_vector_product(
            f,
            Vector{1.0, std::numeric_limits<double>::infinity(), 0.0}
        );
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("Nonfinite direction accepted");

    // Verify that the zero direction produces zero curvature.
    const Vector zero = {0.0, 0.0, 0.0};

    for (double value : hessian_vector_product(f, zero))
        check(value, 0.0, "Zero-direction HVP mismatch");

    check(
        directional_curvature(f, zero),
        0.0,
        "Zero-direction curvature mismatch"
    );

    std::cout
        << "RESEARCH 04: C++17 HVP VALIDATION PASSED\n"
        << "Analytical HVP, directional curvature, "
        << "zero direction and input checks verified.\n";

    return 0;
}
