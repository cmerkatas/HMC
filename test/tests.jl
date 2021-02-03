using Test
include("funnel.jl")

# analytical vs autodiff.
@test (l(ones(2)), ∇l(ones(2))) == dldx(ones(2))
