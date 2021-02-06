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
        if length(p) == 1
            M = convert(Float64, M...)
            p = convert(Float64, p...)
        end
        u, k, h = 0.0, 0.0, 0.0
        new{typeof(q), typeof(p), typeof(ϵ), typeof(L), typeof(M), typeof(u), typeof(k), typeof(h)}(q, p, ϵ, L, M, u, k, h)
    end
end;


function leapfrog!(z::HMCState, ∂U::Function, ∂K::Function)
    ϵ = z.ϵ
    L = z.L
    q = z.q
    p = z.p

    # new state according to leapfrom for hamilton dynamics
    p -= ϵ .* ∂U(q) / 2
    for l in 1:1:L-1
        q += ϵ .* ∂K(p)
        p -= ϵ .* ∂U(q)
    end
    q += ϵ .* ∂K(p)
    p -= ϵ .* ∂U(q) / 2 # the second half for the kinetic variables

    z.q = q
    z.p = -p
end;

function hmcstep!(z::HMCState, U::Function, ∂U::Function, K::Function, ∂K::Function)

    current_q = copy(z.q)
    z.p = rand(MvNormal(zeros(length(current_q)), z.M))
    current_p = copy(z.p)

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
end;
