using Plots
using Flux
using Distributions
using LinearAlgebra
using Random
# using ForwardDiff
# using StaticArrays
# using CSV
# using DataFrames
# using Base.Iterators: repeated
# using Statistics

function leapfrog(q, p, step_size, num_steps; masses = [])
    d = length(q)
    if d != length(p)
        error("Position vector and momentum vector must have equal length.")
    end
    # if d == 1
    #     p = [p]
    #     q = [q]
    # end

    # define masses of kinetic energy
    if isempty(masses)
        masses = d > 1 ? ones(d) : 1
    else
        if length(masses)!= d
            error("Length of masses does not equal number of parameters")
        end
    end

    # new state according to leapfrom for hamilton dynamics
    p -= step_size .* ∇U(q)[1] / 2
    for jump in 1:1:num_steps-1
        q += step_size .* p ./ masses
        p -= step_size .* ∇U(q)[1]
    end
    q += step_size .* p ./ masses
    p -= step_size .* ∇U(q)[1] / 2 # the second half for the kinetic variables
    # println("p=",-p)
    # println("q=",q)
    return q, -p

end


## example of leapfrog on h(q,p) = q^2/2 + p^2/2
H(q, p) = q^2 / 2  + p^2 / 2
U(q) = q^2 / 2
gradU(q) = q                # analytical gradient
∇U(q) = gradient(U, q)      # numerical gradient

q = range(-1, 1, length=100)
p = range(-1, 1, length=100)

H.(q,p)
g = ∇U.(q)

contour(p,q, (p,q) -> H.(q,p), levels=1, color=:blue); xlims!((-2,2)); ylims!((-2,2))

function test_leapfrog(ϵ, L)
    gradU(q) = q
    ∇U(q) = q

    q = 0.0
    p = 1.0
    P = zeros(L)
    Q = zeros(L)
    for i in 1:1:L
        q, p = leapfrog.(q,p,ϵ,i)
        P[i]=p
        Q[i]=q
    end
    return P,Q
end

P,Q = test_leapfrog(0.5,20)
scatter!(P,Q,legend=nothing)

function hmc(current_q, step_size, num_steps; masses=[])
    q = current_q
    d = length(q)
    # if d != length(p)
    #     error("Position vector and momentum vector must have equal length.")
    # end
    # if d == 1
    #     p = [p]
    #     q = [q]
    # end

    # define masses of kinetic energy
    if isempty(masses)
        masses = d > 1 ? ones(d) : 1
    else
        if length(masses) != d
            error("Length of masses does not equal number of parameters")
        end
    end
    p = rand(MvNormal(zeros(d), Diagonal(masses)))
    current_p = p

    # new state according to leapfrom for hamilton dynamics
    q, p = leapfrog(q, p, step_size, num_steps; masses = [])

    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(current_q)
    current_K = sum(current_p.^2 ./ (2*masses))
    proposed_U = U(q)
    proposed_K = sum(p.^2 / (2*masses))

    if  rand(Uniform(0,1)) < exp(current_U - proposed_U + current_K - proposed_K)
        return q
    else
        return current_q
    end
end

μ = zeros(2)
Σ = [1 0.8;0.8 1]
# μ, Σ = zeros(d), Diagonal(ones(d))
U(q) = -logpdf(MvNormal(μ,Σ), q)
∇U(q) = gradient(q->logpdf(MvNormal(μ,Σ), q), q)



function main(n=1000)
    sampled_values = zeros(n,2)
    sampled_values[1,:] = [0, -0.0]
    for i in 2:n
        sampled_values[i,:] = hmc(sampled_values[i-1,:], 0.1, 10)
    end
    return sampled_values
end
Random.seed!(123)
samples = main()

histogram(samples[:,2])

x_range=range(-3, stop=3, length=100)
y_range=range(-3, stop=3, length=100)
Z = [pdf(MvNormal(μ, Σ), [x,y]) for x in x_range, y in y_range]
contour(x_range, y_range, Z)
plot!(samples[:,1], samples[:,2], color="red")

using GeometryBasics
heatmap(x_range, y_range, Z, color=:deep)
plot!(Point2.(eachcol(samples')), color="red", label="samples")

anim = @gif for i=1:10:1000
    scatter!(samples[i,1], samples legend=nothing, color="blue")
end every 5
