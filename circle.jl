using Distributions, Plots, Zygote
using LinearAlgebra, Random
using Turing: MvNormal
using StatsBase

# run it for the funnel distribution
include("hmc.jl")
# example of leapfrog on h(q,p) = q^2/2 + p^2/2
H(q, p) = q.^2 / 2  + p.^2 / 2
U(q) = q^2 / 2
K(p) = p^2 / 2
function ∂U(q)
    u, lambda = Zygote.pullback(U, q)
    grad = first(lambda(1))
    return grad
end

function ∂K(p)
    k, lambda = Zygote.pullback(K, p)
    grad = first(lambda(1))
    return grad
end


q = range(-1, 1, length=100);
p = range(-1, 1, length=100);

contour(p,q, (p,q) -> H.(q,p), levels=1, color=:blue); xlims!((-2,2)); ylims!((-2,2))

function test_leapfrog(ϵ, L)
    z = HMCState(0.0, ϵ, L)
    z.p = 1.
    ∂U(q) = q

    P = zeros(L)
    Q = zeros(L)
    for i in 1:1:L
        z.L = i
        leapfrog!(z, ∂U, ∂K)
        P[i]=copy(z.p)
        Q[i]=copy(z.q)
    end
    return P,Q
end

@time P,Q = test_leapfrog(0.5,20)
scatter!(P,Q,legend=nothing)
