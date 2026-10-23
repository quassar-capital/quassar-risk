#ifndef QUASSAR_RESEARCH04_AUTODIFF_HPP
#define QUASSAR_RESEARCH04_AUTODIFF_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quassar::research04 {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

struct Jet {
    double value;
    Vector gradient;
    Matrix hessian;

    Jet(double v, Vector g, Matrix h)
        : value(v), gradient(std::move(g)), hessian(std::move(h)) {
        const auto n = gradient.size();

        if (n == 0 || hessian.size() != n)
            throw std::invalid_argument("Invalid derivative dimensions");

        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite value");

        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(gradient[i]) || hessian[i].size() != n)
                throw std::invalid_argument("Invalid derivative structure");

            for (double x : hessian[i])
                if (!std::isfinite(x))
                    throw std::invalid_argument("Nonfinite Hessian");
        }
    }

    std::size_t dimension() const {
        return gradient.size();
    }

    static Jet constant(double value, std::size_t n) {
        if (n == 0)
            throw std::invalid_argument("Dimension must be positive");

        return Jet(value, Vector(n, 0.0), Matrix(n, Vector(n, 0.0)));
    }

    static Jet variable(double value, std::size_t index, std::size_t n) {
        if (n == 0 || index >= n)
            throw std::invalid_argument("Invalid variable index");

        Jet result = constant(value, n);
        result.gradient[index] = 1.0;
        return result;
    }
};

inline void check_dimensions(const Jet& a, const Jet& b) {
    if (a.dimension() != b.dimension())
        throw std::invalid_argument("Derivative dimension mismatch");
}

inline Jet operator+(const Jet& a, const Jet& b) {
    check_dimensions(a, b);
    const auto n = a.dimension();

    Vector g(n);
    Matrix h(n, Vector(n));

    for (std::size_t i = 0; i < n; ++i) {
        g[i] = a.gradient[i] + b.gradient[i];

        for (std::size_t j = 0; j < n; ++j)
            h[i][j] = a.hessian[i][j] + b.hessian[i][j];
    }

    return Jet(a.value + b.value, g, h);
}

inline Jet operator-(const Jet& a) {
    const auto n = a.dimension();

    Vector g(n);
    Matrix h(n, Vector(n));

    for (std::size_t i = 0; i < n; ++i) {
        g[i] = -a.gradient[i];

        for (std::size_t j = 0; j < n; ++j)
            h[i][j] = -a.hessian[i][j];
    }

    return Jet(-a.value, g, h);
}

inline Jet operator-(const Jet& a, const Jet& b) {
    return a + (-b);
}

inline Jet operator*(const Jet& a, const Jet& b) {
    check_dimensions(a, b);
    const auto n = a.dimension();

    Vector g(n);
    Matrix h(n, Vector(n));

    for (std::size_t i = 0; i < n; ++i) {
        g[i] =
            a.gradient[i] * b.value +
            a.value * b.gradient[i];

        for (std::size_t j = 0; j < n; ++j) {
            h[i][j] =
                a.hessian[i][j] * b.value +
                a.value * b.hessian[i][j] +
                a.gradient[i] * b.gradient[j] +
                b.gradient[i] * a.gradient[j];
        }
    }

    return Jet(a.value * b.value, g, h);
}

inline Jet compose(
    const Jet& x,
    double value,
    double first,
    double second
) {
    const auto n = x.dimension();

    Vector g(n);
    Matrix h(n, Vector(n));

    for (std::size_t i = 0; i < n; ++i) {
        g[i] = first * x.gradient[i];

        for (std::size_t j = 0; j < n; ++j) {
            h[i][j] =
                second * x.gradient[i] * x.gradient[j] +
                first * x.hessian[i][j];
        }
    }

    return Jet(value, g, h);
}

inline Jet reciprocal(const Jet& x) {
    if (x.value == 0.0)
        throw std::domain_error("Division by zero");

    const double v = x.value;

    return compose(
        x,
        1.0 / v,
        -1.0 / (v * v),
        2.0 / (v * v * v)
    );
}

inline Jet operator/(const Jet& a, const Jet& b) {
    return a * reciprocal(b);
}

inline Jet exp(const Jet& x) {
    const double y = std::exp(x.value);
    return compose(x, y, y, y);
}

inline Jet log(const Jet& x) {
    if (x.value <= 0.0)
        throw std::domain_error("Logarithm requires positive argument");

    return compose(
        x,
        std::log(x.value),
        1.0 / x.value,
        -1.0 / (x.value * x.value)
    );
}

inline Jet sin(const Jet& x) {
    return compose(
        x,
        std::sin(x.value),
        std::cos(x.value),
        -std::sin(x.value)
    );
}

inline Jet cos(const Jet& x) {
    return compose(
        x,
        std::cos(x.value),
        -std::sin(x.value),
        -std::cos(x.value)
    );
}

inline Jet operator+(const Jet& a, double b) {
    return a + Jet::constant(b, a.dimension());
}

inline Jet operator+(double a, const Jet& b) {
    return Jet::constant(a, b.dimension()) + b;
}

inline Jet operator-(const Jet& a, double b) {
    return a - Jet::constant(b, a.dimension());
}

inline Jet operator-(double a, const Jet& b) {
    return Jet::constant(a, b.dimension()) - b;
}

inline Jet operator*(const Jet& a, double b) {
    return a * Jet::constant(b, a.dimension());
}

inline Jet operator*(double a, const Jet& b) {
    return Jet::constant(a, b.dimension()) * b;
}

inline Jet operator/(const Jet& a, double b) {
    return a / Jet::constant(b, a.dimension());
}

inline Jet operator/(double a, const Jet& b) {
    return Jet::constant(a, b.dimension()) / b;
}

}  // namespace quassar::research04

#endif
