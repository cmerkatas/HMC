function l(x::Array{Float64, 1}; σ=3.0)
    # x[2] is the normal with std σ
    lpdf = -x[2] / 2 + x[2]^2 / (2σ^2) + x[1]^2 / (2*exp(x[2]))
    return lpdf
end

v_range = range(-5, stop=5, length=400);
x_range = range(-5, stop=5, length=400);
F = [l([x,y], σ=3.0) for x in x_range, y in x_range]';

heatmap(x_range, x_range, exp.(-F), color=:deep)

# analytical gradient
function ∇l(x::Array{Float64, 1}; σ=3.0)
    ∂x₁ = x[1] / exp(x[2])
    ∂x₂ = -1.0 / 2.0 + x[2]/σ^2 - x[1]^2 / (2*exp(x[2]))
    return [∂x₁, ∂x₂]
end

# zygote gradient
function dldx(x)
    q, lambda = Zygote.pullback(l, x)
    grad = first(lambda(1))
    return q, grad
end

# l(ones(2)), ∇l(ones(2))
# q, grad = dldx(ones(2))
