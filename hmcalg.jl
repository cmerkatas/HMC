using LinearAlgebra
using Turing: MvNormal
abstract type SamplerState end

mutable struct HMCState{Q, P, E, LL, MM} <: SamplerState
    q::Q    # position, variables of interest
    p::P    # momentum, auxiliary variables
    ϵ::E    # step size
    L::LL   # num of steps
    M::MM   # mass matrix
    # maybe add hamiltonian, potential and kinetic energy

    function HMCState(q, ϵ, L; M=nothing)
        if M === nothing
            dq = length(q)
            M = I(dq)
        end
        p = rand(MvNormal(zeros(dq), M))
        new{typeof(q), typeof(p), typeof(ϵ), typeof(L), typeof(M)}(q, p, ϵ, L, M)
    end
end


function leapfrog!(z::HMCState, ∂U::Function, ∂K::Function)
    step_size = z.ϵ
    num_steps = z.L
    q = z.q
    p = z.p

    # new state according to leapfrom for hamilton dynamics
    p -= step_size .* ∂U(q) / 2
    for jump in 1:1:num_steps-1
        q += step_size .* ∂K(p)
        p -= step_size .* ∂U(q)
    end
    q += step_size .* ∂K(p)
    p -= step_size .* ∂U(q) / 2 # the second half for the kinetic variables

    z.q = q
    z.p = -p
end

function hmcstep!(z::HMCState, U::Function, ∂U::Function, K::Function, ∂K::Function)

    current_q = z.q
    z.p = rand(MvNormal(zeros(length(current_q)), z.M))
    current_p = z.p

    # new state according to leapfrom for hamilton dynamics
    leapfrog!(z, ∂U, ∂K)

    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = K(current_p)
    proposed_U = U(z.q)
    proposed_K = K(z.p)

    if  rand(Float64) .< exp(current_U - proposed_U + current_K - proposed_K)
        z.q = z.q
    else
        z.q = current_q
    end
end


# run it for the funnel distribution
include("funnel.jl")
# define the state
z = HMCState(ones(2), 0.025, 4)
M = z.M
# The potential U and ∂U are given from l and ∇l
# define the kinetic energy K = p^⊤ Minv p / 2 i.e a multivariate Normal
K(p) = p' * M * p / 2
∂K(p) = M \ p
hmcstep!(z, l, ∇l, K, ∂K)

ns = 500
Random.seed!(1)
samples=zeros(2, ns)
for s in 1:1:ns
    hmcstep!(z, l, ∇l, K, ∂K)
    samples[:,s] = z.q
end

using GeometryBasics
v_range = range(-5, stop=5, length=400)
x_range = range(-5, stop=5, length=400)
Z = [l([x,y], σ=3.0) for x in x_range, y in x_range]'

heatmap(x_range, x_range, exp.(-Z), color=:deep)
plot!(Point2.(eachcol(samples)), color="red", label="samples")
