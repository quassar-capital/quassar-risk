#include "autodiff.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace quassar::research04;

void check(double actual, double expected) {
    if (std::abs(actual - expected) > 1e-11) {
        throw std::runtime_error("Analytical derivative mismatch");
    }
}

int main() {
    const double a = 1.2;
    const double b = 0.7;

    Jet x = Jet::variable(a, 0, 2);
    Jet y = Jet::variable(b, 1, 2);

    // f(x,y) = x²y + exp(x) + sin(y)
    Jet f = x * x * y + exp(x) + sin(y);

    check(f.value, a*a*b + std::exp(a) + std::sin(b));

    check(f.gradient[0], 2*a*b + std::exp(a));
    check(f.gradient[1], a*a + std::cos(b));

    check(f.hessian[0][0], 2*b + std::exp(a));
    check(f.hessian[0][1], 2*a);
    check(f.hessian[1][0], 2*a);
    check(f.hessian[1][1], -std::sin(b));

    // Independent logarithmic derivative reference.
    Jet z = Jet::variable(2.0, 0, 1);
    Jet g = log(z*z + 1.0);

    check(g.gradient[0], 4.0 / 5.0);
    check(g.hessian[0][0], -6.0 / 25.0);

    // Reciprocal derivatives.
    Jet h = 1.0 / z;

    check(h.value, 0.5);
    check(h.gradient[0], -0.25);
    check(h.hessian[0][0], 0.25);

    // Domain validation.
    bool rejected = false;

    try {
        log(Jet::variable(-1.0, 0, 1));
    } catch (const std::domain_error&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("Invalid logarithm accepted");

    // Derivative dimension validation.
    rejected = false;

    try {
        Jet::variable(1.0, 0, 2)
            + Jet::variable(1.0, 0, 3);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }

    if (!rejected)
        throw std::runtime_error("Dimension mismatch accepted");

    std::cout
        << "RESEARCH 04: C++17 AUTODIFF VALIDATION PASSED\n"
        << "Analytical gradients, Hessians, mixed derivatives "
        << "and domain checks verified.\n";

    return 0;
}
