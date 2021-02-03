using Random, Statistics, StatsBase
struct FunnelDistribution{T<:Real,U<:Integer} <: Distribution{Multivariate,Continuous}
    "Variance of the dominant normal distribution."
    a::T

    "Variance of the supporting normal distributions."
    b::T

    "Number of dimensions."
    n::U
end

function FunnelDistribution(; a::Real = 1.0, b::Real = 0.5, n::Integer = 3)
    a_pr, b_pr = promote(a, b)
    FunnelDistribution(a_pr, b_pr, n)
end


Base.length(d::FunnelDistribution) = d.n
Base.eltype(d::FunnelDistribution) = Base.eltype(d.a)

Statistics.mean(d::FunnelDistribution) = zeros(d.n)

function Statistics.cov(dist::FunnelDistribution)
    cov(nestedview(rand(rng, sampler(dist), 10^5)))
end

StatsBase.params(d::FunnelDistribution) = (d.a, d.b, d.n)


function Distributions._logpdf(d::FunnelDistribution, x::AbstractArray)
    idxs = eachindex(x)
    s = logpdf(Normal(0, d.a^2), x[idxs[1]])
    @inbounds for i in idxs[2:end]
        s += logpdf(Normal(0, exp(2 * d.b * x[1])), x[i])
    end
    s
end

function Distributions.gradlogpdf(d::FunnelDistribution{T, U}, x::AbstractVector{T}) where {T<:Real, U<:Integer}
    FillArrays.Zeros(length(d))
end

function Distributions._rand!(rng::AbstractRNG, d::FunnelDistribution, x::AbstractVector)
    idxs = eachindex(x)
    x[idxs[1]] = rand(rng, Normal(0, d.a^2))
    @inbounds for i in idxs[2:end]
        x[i] = rand(rng, Normal(0, exp(2 * d.b * x[1])))
    end
    return x
end
#
#
# dist = FunnelDistribution(1.0, 0.5, 2)
#
# s=rand(dist,200)
#
# scatter(s[1,:],s[2,:])
