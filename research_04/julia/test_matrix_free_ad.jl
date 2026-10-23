using Test

include("MatrixFreeAD.jl")
using .MatrixFreeAD

function nonlinear(x)
    exp(x[1] * x[2]) +
    sin(x[1]) +
    log(x[2] * x[2] + 2.0)
end

@testset "Research 04 — Julia Matrix-Free AD" begin

    @testset "Nonlinear analytical derivatives" begin
        point = [0.4, -0.2]

        directions = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, -0.3],
            [0.0, 0.0]
        ]

        x, y = point
        e = exp(x * y)
        d = y^2 + 2.0

        expected_value = e + sin(x) + log(d)

        gradient = [
            y * e + cos(x),
            x * e + 2.0 * y / d
        ]

        hessian = [
            y^2 * e - sin(x)       (1.0 + x*y) * e;
            (1.0 + x*y) * e       x^2 * e + 2.0*(2.0-y^2)/d^2
        ]

        for direction in directions
            result = nonlinear(variables(point, direction))

            @test isapprox(
                result.value, expected_value;
                atol=1e-12, rtol=1e-12
            )

            @test all(isapprox.(
                result.gradient, gradient;
                atol=1e-12, rtol=1e-12
            ))

            @test all(isapprox.(
                hvp(result), hessian * direction;
                atol=1e-12, rtol=1e-12
            ))

            @test isapprox(
                result.tangent,
                sum(gradient .* direction);
                atol=1e-12, rtol=1e-12
            )
        end
    end

    @testset "Exact quadratic Delta-Gamma" begin
        A = [
             4.0  1.0 -2.0;
             1.0  6.0  0.5;
            -2.0  0.5  8.0
        ]

        b = [1.5, -0.8, 2.0]
        point = [0.3, -0.7, 1.2]
        shock = [0.12, -0.09, 0.05]

        function model(x)
            result = constant(0.0, length(x))

            for i in eachindex(x)
                result = result + b[i] * x[i]

                for j in eachindex(x)
                    result = result +
                        0.5 * A[i,j] * x[i] * x[j]
                end
            end

            result
        end

        result = delta_gamma(model, point, shock)

        expected_delta = sum((A * point + b) .* shock)
        expected_gamma = 0.5 * sum(shock .* (A * shock))

        valuation(p) = sum(b .* p) + 0.5 * sum(p .* (A * p))

        @test isapprox(
            result.delta, expected_delta;
            atol=1e-12
        )

        @test isapprox(
            result.gamma, expected_gamma;
            atol=1e-12
        )

        @test isapprox(
            result.total,
            valuation(point + shock) - valuation(point);
            atol=1e-12
        )

        zero = delta_gamma(model, point, zeros(3))

        @test zero.delta == 0.0
        @test zero.gamma == 0.0
        @test zero.total == 0.0
    end

    @testset "Input validation" begin
        @test_throws DimensionMismatch variables(
            [1.0, 2.0], [1.0]
        )

        @test_throws BoundsError variable(
            1.0, 3, [1.0, 0.0]
        )

        @test_throws DomainError log(
            variable(-1.0, 1, [1.0])
        )

        @test_throws DomainError (
            variable(1.0, 1, [1.0]) /
            variable(0.0, 1, [1.0])
        )

        @test_throws ArgumentError variable(
            NaN, 1, [1.0]
        )
    end
end

println("RESEARCH 04: JULIA MATRIX-FREE AD VALIDATION PASSED")
