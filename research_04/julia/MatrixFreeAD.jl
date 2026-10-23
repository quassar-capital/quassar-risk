module MatrixFreeAD

export DirectionalJet, variable, constant, variables
export hvp, delta_gamma

struct DirectionalJet
    value::Float64
    gradient::Vector{Float64}
    hvp::Vector{Float64}
    tangent::Float64

    function DirectionalJet(value, gradient, hvp, tangent)
        g = Float64.(gradient)
        h = Float64.(hvp)
        v = Float64(value)
        t = Float64(tangent)

        isempty(g) && throw(ArgumentError("Empty derivative dimension"))
        length(g) == length(h) ||
            throw(DimensionMismatch("Gradient/HVP mismatch"))

        all(isfinite, g) && all(isfinite, h) &&
        isfinite(v) && isfinite(t) ||
            throw(ArgumentError("Nonfinite value or derivative"))

        new(v, g, h, t)
    end
end

function constant(value::Real, n::Integer)
    n > 0 || throw(ArgumentError("Dimension must be positive"))
    DirectionalJet(value, zeros(n), zeros(n), 0.0)
end

function variable(value::Real, index::Integer, direction::AbstractVector)
    n = length(direction)
    n > 0 || throw(ArgumentError("Empty direction"))
    1 <= index <= n || throw(BoundsError(direction, index))
    all(isfinite, direction) ||
        throw(ArgumentError("Nonfinite direction"))

    g = zeros(n)
    g[index] = 1.0

    DirectionalJet(value, g, zeros(n), direction[index])
end

function variables(point::AbstractVector, direction::AbstractVector)
    length(point) == length(direction) ||
        throw(DimensionMismatch("Point/direction mismatch"))

    [variable(point[i], i, direction) for i in eachindex(point)]
end

function checkdims(a::DirectionalJet, b::DirectionalJet)
    length(a.gradient) == length(b.gradient) ||
        throw(DimensionMismatch("Derivative dimension mismatch"))
end

function compose(x::DirectionalJet, value, first, second)
    g = first .* x.gradient
    h = first .* x.hvp .+
        (second * x.tangent) .* x.gradient

    DirectionalJet(value, g, h, first * x.tangent)
end

import Base: +, -, *, /, exp, log, sin, cos

function +(a::DirectionalJet, b::DirectionalJet)
    checkdims(a, b)
    DirectionalJet(
        a.value + b.value,
        a.gradient .+ b.gradient,
        a.hvp .+ b.hvp,
        a.tangent + b.tangent
    )
end

-(a::DirectionalJet) =
    DirectionalJet(-a.value, -a.gradient, -a.hvp, -a.tangent)

-(a::DirectionalJet, b::DirectionalJet) = a + (-b)

function *(a::DirectionalJet, b::DirectionalJet)
    checkdims(a, b)

    g = a.gradient .* b.value .+
        b.gradient .* a.value

    h = a.hvp .* b.value .+
        b.hvp .* a.value .+
        a.gradient .* b.tangent .+
        b.gradient .* a.tangent

    DirectionalJet(
        a.value * b.value,
        g,
        h,
        a.tangent * b.value + a.value * b.tangent
    )
end

function reciprocal(x::DirectionalJet)
    x.value != 0.0 || throw(DomainError(x.value, "Division by zero"))

    v = x.value
    compose(x, inv(v), -inv(v^2), 2.0 / v^3)
end

/(a::DirectionalJet, b::DirectionalJet) = a * reciprocal(b)

for op in (:+, :-, :*, :/)
    @eval begin
        function ($op)(a::DirectionalJet, b::Real)
            ($op)(a, constant(b, length(a.gradient)))
        end

        function ($op)(a::Real, b::DirectionalJet)
            ($op)(constant(a, length(b.gradient)), b)
        end
    end
end

function exp(x::DirectionalJet)
    v = Base.exp(x.value)
    compose(x, v, v, v)
end

function log(x::DirectionalJet)
    x.value > 0.0 ||
        throw(DomainError(x.value, "Logarithm requires positive input"))

    v = x.value
    compose(x, Base.log(v), inv(v), -inv(v^2))
end

sin(x::DirectionalJet) =
    compose(x, Base.sin(x.value), Base.cos(x.value), -Base.sin(x.value))

cos(x::DirectionalJet) =
    compose(x, Base.cos(x.value), -Base.sin(x.value), -Base.cos(x.value))

function hvp(result::DirectionalJet)
    copy(result.hvp)
end

function compensated_sum(values)
    total = 0.0
    correction = 0.0

    for value in values
        isfinite(value) ||
            throw(ArgumentError("Nonfinite contribution"))

        next = total + value

        if abs(total) >= abs(value)
            correction += (total - next) + value
        else
            correction += (value - next) + total
        end

        total = next
    end

    result = total + correction
    isfinite(result) ||
        throw(OverflowError("Nonfinite accumulated result"))

    result
end

function delta_gamma(model, point::AbstractVector, shock::AbstractVector)
    x = variables(point, shock)
    result = model(x)

    result isa DirectionalJet ||
        throw(ArgumentError("Model must return DirectionalJet"))

    length(result.gradient) == length(shock) ||
        throw(DimensionMismatch("Model output dimension mismatch"))

    delta = compensated_sum(result.gradient .* shock)
    gamma = 0.5 * compensated_sum(result.hvp .* shock)
    total = delta + gamma

    isfinite(total) ||
        throw(OverflowError("Nonfinite Delta-Gamma approximation"))

    (; delta, gamma, total)
end

end
