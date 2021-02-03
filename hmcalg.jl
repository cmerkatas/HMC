using LinearAlgebra, Random
using Turing: MvNormal
abstract type SamplerState end

mutable struct HMCState{Q, P, E, LL, MM, UU, KK, HH} <: SamplerState
    q::Q    # position, variables of interest
    p::P    # momentum, auxiliary variables
    ϵ::E    # step size
    L::LL   # num of steps
    M::MM   # mass matrix
    u::UU
    k::KK
    h::HH

    function HMCState(q, ϵ, L; M=nothing)
        if M === nothing
            M = I(length(q))
        end
        p = rand(MvNormal(zeros(length(q)), M))
        u, k, h = 0.0, 0.0, 0.0
        new{typeof(q), typeof(p), typeof(ϵ), typeof(L), typeof(M), typeof(u), typeof(k), typeof(h)}(q, p, ϵ, L, M, u, k, h)
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
        z.u = proposed_U
        z.k = proposed_K
        z.h = proposed_U + proposed_K
    else
        z.q = current_q
        z.u = current_U
        z.k = current_K
        z.h = current_U + current_K
    end
end


# run it for the funnel distribution
# define the state
z = HMCState(zeros(2), 0.05, 3; M=[0.9 0.1;0.1 0.9])
M = z.M
# The potential U and ∂U are given from l and ∇l
# define the kinetic energy K = p^⊤ Minv p / 2 i.e a multivariate Normal
K(p) = p' * M * p / 2
∂K(p) = M \ p
hmcstep!(z, l, ∇l, K, ∂K)

ns = 500
Random.seed!(1)
sampled_states = Array{SamplerState}(undef, ns)
samples = zeros(2, ns)
energies = (U=zeros(ns), K=zeros(ns), H=zeros(ns))
for s in 1:1:ns
    hmcstep!(z, l, ∇l, K, ∂K)
    sampled_states[s] = z
    samples[:,s] = z.q
    energies.U[s] = z.u
    energies.K[s] = z.k
    energies.H[s] = z.u + z.k
end

v_range = range(-5, stop=5, length=400)
x_range = range(-5, stop=5, length=400)
Z = [l([x,y], σ=3.0) for x in x_range, y in x_range]'

heatmap(x_range, x_range, exp.(-Z), color=:deep)
plot!(samples[1,:], samples[2,:], color="red")
#
# plot(1:ns, energies.U, label="U(q)", lw=1.5, color=:blue)
# plot!(energies.K, label="K(p)", lw=1.5, color=:red)
# plot!(energies.H, label="H(q,p)", lw=1.5, color=:green)
