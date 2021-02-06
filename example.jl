using Distributions, Plots, Zygote
using LinearAlgebra, Random
using Turing: MvNormal
using StatsBase

# run it for the funnel distribution
include("hmc.jl")
include("funnel.jl")

# define the state
z = HMCState(ones(2), 0.01, 5)
M = z.M
# The potential U and ∂U are given from l and ∇l
# define the kinetic energy K = p^⊤ Minv p / 2 i.e a multivariate Normal
K(p) = p' * M * p / 2
∂K(p) = M \ p
hmcstep!(z, l, ∇l, K, ∂K)

nsamples, burnin = 10000, 5000
Random.seed!(12345)
samples=zeros(2, nsamples-burnin)
for s in 1:1:nsamples
    hmcstep!(z, l, ∇l, K, ∂K)
    if s > burnin
        samples[:,s-burnin] = z.q
    end
end

v_range = range(-50, stop=50, length=500)
x_range = range(-25, stop=25, length=500)
Z = [l([x,y], σ=3.0) for x in x_range, y in x_range]'

heatmap(x_range, x_range, exp.(-Z), color=:deep)
plot!(samples[1,:], samples[2,:], color=:red)
acf1 = autocor(samples[1,:], 1:20);
pltacf1 = plot(acf1, title="Autocorrelation function", line=:stem);
acf2 = autocor(samples[2,:], 1:20);
pltacf2 = plot(acf2, title="Autocorrelation function", line=:stem);
plot(pltacf1, pltacf2, layout=(1,2))
