using Distributions
using Flux
using Plots

mutable struct HMCState
    position::Union{Number, Array{Float64,1}}
    momentum::Union{Number, Array{Float64,1}}
    step_size::Float64
    num_steps::Int64
end

function HMCState(q, p, ssize, nsteps)
    position = q
    momentum = p
    step_size = ssize
    num_steps = nsteps
    HMCState(position, momentum, step_size, num_steps)
end

function leapfrog!(z::HMCState, gradU::Function, gradK::Function)
    step_size = z.step_size
    num_steps = z.num_steps
    q = z.position
    p = z.momentum

    # new state according to leapfrom for hamilton dynamics
    p -= step_size .* gradU.(q) / 2
    for jump in 1:1:num_steps-1
        q += step_size .* gradK.(p)
        p -= step_size .* gradU.(q)
    end
    q += step_size .* gradK.(p)
    p -= step_size .* gradU.(q) / 2 # the second half for the kinetic variables

    z.position = q
    z.momentum = -p
    return nothing
end

#q = zeros(2)
#p = ones(2)
q, p, step_size, num_steps = 0.0,1.0, 0.1, 10
x = HMCState(q, p, step_size, num_steps)

U(x) = x^2 / 2
K(x) = x^2 / 2
∇U(x) = gradient(U, x)[1]     # numerical gradient
∇K(x) = gradient(K, x)[1]     # numerical gradient

H(x, y) = U(x) + K(y)

leapfrog!(x, ∇U, ∇K)

function sg_hmc!(state::HMCState, U::Function, gradU::Function,
                K::Function, gradK::Function; masses=Float64[])

    current_q = state.position
    d = length(current_q)
    masses = isempty(masses) ? 1.0 : masses
    state.momentum =  d > 1 ? rand(MvNormal(zeros(d), Diagonal(masses))) : rand(Normal(0, sqrt(masses)))
    current_p = state.momentum

    # new state according to leapfrom for hamilton dynamics
    leapfrog!(state, gradU, gradK)
    p = state.momentum
    q = state.position
    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = sum(current_p.^2 ./ (2*masses))
    proposed_U = U(q)
    proposed_K = sum(p.^2 / (2*masses))

    if  rand(Float64) .< exp(current_U - proposed_U + current_K - proposed_K)
        state.position = q
    else
        state.position = current_q
    end
    return nothing
end


function test_hmc(n=100000)
    samples = zeros(n)
    q, p, step_size, num_steps = 0.0, 1.0, 0.1, 10
    x = HMCState(q, p, step_size, num_steps)
    metric = 1.0
    U(x) = -logpdf(Normal(12.5,2.2), x)
    K(x) = x' * inv(metric) * x
    ∇U(x) = gradient(U, x)[1]     # numerical gradient
    ∇K(x) = gradient(K, x)[1]     # numerical gradient
    for i in 1:n
        sg_hmc!(x, U, ∇U, K, ∇K, masses=metric)
        samples[i] = x.position
    end
    return samples
end

@time samples = test_hmc()

histogram(samples)


mean(samples)
std(samples)




U(x) = x^2 / 2
∇U(x) = gradient(U, x)[1]     # numerical gradient

H(x, y) = U(x) + K(y)


x = range(-1, 1, length=100)
y = range(-1, 1, length=100)

contour(x,y, (x,y) -> H.(x,y), levels=1, color=:blue); xlims!((-2,2)); ylims!((-2,2))


function test_leapfrog(ϵ, L)
    q,p=0.0,1.0
    z = HMCState(q, p)

    K(x) = x^2 / 2
    ∇K(x) = gradient(K, x)[1]      # numerical gradient

    U(x) = x^2 / 2
    ∇U(x) = gradient(U, x)[1]
    P = zeros(L)
    Q = zeros(L)
    for i in 1:1:L
        leapfrog!(z, ∇U, ∇K, ϵ, L)
        P[i]=z.momentum
        Q[i]=z.position
    end
    return P,Q
end

P,Q = test_leapfrog(0.5,20)
