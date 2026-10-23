#ifndef QUASSAR_RESEARCH04_MATRIX_FREE_HVP_HPP
#define QUASSAR_RESEARCH04_MATRIX_FREE_HVP_HPP

#include "autodiff.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quassar::research04::matrix_free {

struct DirectionalJet {
    double value;
    Vector gradient;
    Vector hvp;
    double tangent;

    DirectionalJet(
        double value_,
        Vector gradient_,
        Vector hvp_,
        double tangent_
    )
        : value(value_),
          gradient(std::move(gradient_)),
          hvp(std::move(hvp_)),
          tangent(tangent_) {

        if (gradient.empty() || gradient.size() != hvp.size())
            throw std::invalid_argument("Invalid derivative dimensions");

        if (!std::isfinite(value) || !std::isfinite(tangent))
            throw std::invalid_argument("Nonfinite scalar");

        for (std::size_t i = 0; i < gradient.size(); ++i) {
            if (!std::isfinite(gradient[i]) || !std::isfinite(hvp[i]))
                throw std::invalid_argument("Nonfinite derivative");
        }
    }

    std::size_t dimension() const {
        return gradient.size();
    }

    static DirectionalJet constant(double value, std::size_t n) {
        if (n == 0)
            throw std::invalid_argument("Dimension must be positive");

        return DirectionalJet(
            value, Vector(n, 0.0), Vector(n, 0.0), 0.0
        );
    }

    static DirectionalJet variable(
        double value,
        std::size_t index,
        const Vector& direction
    ) {
        const auto n = direction.size();

        if (n == 0 || index >= n)
            throw std::invalid_argument("Invalid variable index");

        for (double v : direction)
            if (!std::isfinite(v))
                throw std::invalid_argument("Nonfinite direction");

        Vector gradient(n, 0.0);
        gradient[index] = 1.0;

        return DirectionalJet(
            value, gradient, Vector(n, 0.0), direction[index]
        );
    }
};

inline void check_dimensions(
    const DirectionalJet& a,
    const DirectionalJet& b
) {
    if (a.dimension() != b.dimension())
        throw std::invalid_argument("Derivative dimension mismatch");
}

inline DirectionalJet operator+(
    const DirectionalJet& a,
    const DirectionalJet& b
) {
    check_dimensions(a, b);

    const auto n = a.dimension();
    Vector g(n), hv(n);

    for (std::size_t i = 0; i < n; ++i) {
        g[i] = a.gradient[i] + b.gradient[i];
        hv[i] = a.hvp[i] + b.hvp[i];
    }

    return DirectionalJet(
        a.value + b.value, g, hv, a.tangent + b.tangent
    );
}

inline DirectionalJet operator-(const DirectionalJet& a) {
    Vector g = a.gradient;
    Vector hv = a.hvp;

    for (double& x : g) x = -x;
    for (double& x : hv) x = -x;

    return DirectionalJet(-a.value, g, hv, -a.tangent);
}

inline DirectionalJet operator-(
    const DirectionalJet& a,
    const DirectionalJet& b
) {
    return a + (-b);
}

inline DirectionalJet operator*(
    const DirectionalJet& a,
    const DirectionalJet& b
) {
    check_dimensions(a, b);

    const auto n = a.dimension();
    Vector g(n), hv(n);

    for (std::size_t i = 0; i < n; ++i) {
        g[i] =
            a.gradient[i] * b.value +
            a.value * b.gradient[i];

        hv[i] =
            a.hvp[i] * b.value +
            a.value * b.hvp[i] +
            a.gradient[i] * b.tangent +
            a.tangent * b.gradient[i];
    }

    return DirectionalJet(
        a.value * b.value,
        g,
        hv,
        a.tangent * b.value + a.value * b.tangent
    );
}

inline DirectionalJet compose(
    const DirectionalJet& x,
    double value,
    double first,
    double second
) {
    const auto n = x.dimension();
    Vector g(n), hv(n);

    for (std::size_t i = 0; i < n; ++i) {
        g[i] = first * x.gradient[i];

        hv[i] =
            first * x.hvp[i] +
            second * x.tangent * x.gradient[i];
    }

    return DirectionalJet(
        value, g, hv, first * x.tangent
    );
}

inline DirectionalJet reciprocal(const DirectionalJet& x) {
    if (x.value == 0.0)
        throw std::domain_error("Division by zero");

    const double v = x.value;

    return compose(
        x, 1.0 / v, -1.0 / (v * v), 2.0 / (v * v * v)
    );
}

inline DirectionalJet operator/(
    const DirectionalJet& a,
    const DirectionalJet& b
) {
    return a * reciprocal(b);
}

inline DirectionalJet exp(const DirectionalJet& x) {
    const double y = std::exp(x.value);
    return compose(x, y, y, y);
}

inline DirectionalJet log(const DirectionalJet& x) {
    if (x.value <= 0.0)
        throw std::domain_error("Logarithm requires positive argument");

    return compose(
        x,
        std::log(x.value),
        1.0 / x.value,
        -1.0 / (x.value * x.value)
    );
}

inline DirectionalJet sin(const DirectionalJet& x) {
    return compose(
        x,
        std::sin(x.value),
        std::cos(x.value),
        -std::sin(x.value)
    );
}

inline DirectionalJet cos(const DirectionalJet& x) {
    return compose(
        x,
        std::cos(x.value),
        -std::sin(x.value),
        -std::cos(x.value)
    );
}

inline DirectionalJet operator+(
    const DirectionalJet& a, double b
) {
    return a + DirectionalJet::constant(b, a.dimension());
}

inline DirectionalJet operator+(
    double a, const DirectionalJet& b
) {
    return DirectionalJet::constant(a, b.dimension()) + b;
}

inline DirectionalJet operator-(
    const DirectionalJet& a, double b
) {
    return a - DirectionalJet::constant(b, a.dimension());
}

inline DirectionalJet operator-(
    double a, const DirectionalJet& b
) {
    return DirectionalJet::constant(a, b.dimension()) - b;
}

inline DirectionalJet operator*(
    const DirectionalJet& a, double b
) {
    return a * DirectionalJet::constant(b, a.dimension());
}

inline DirectionalJet operator*(
    double a, const DirectionalJet& b
) {
    return DirectionalJet::constant(a, b.dimension()) * b;
}

inline DirectionalJet operator/(
    const DirectionalJet& a, double b
) {
    return a / DirectionalJet::constant(b, a.dimension());
}

inline DirectionalJet operator/(
    double a, const DirectionalJet& b
) {
    return DirectionalJet::constant(a, b.dimension()) / b;
}

}  // namespace quassar::research04::matrix_free

#endif
