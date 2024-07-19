include("toy_normal.jl")
include("sais.jl")

function test_large_t(elt_type) 
    a = ais(NormalPath(1), SAIS(14); elt_type)
    μ = a.particles
    @show ess(μ), n_particles(μ)
    @assert isapprox(ess(μ), n_particles(μ), atol=10)
    @assert isapprox(a.barriers.globalbarrier, 1.2, atol=0.05)
end
test_large_t(Float64)
test_large_t(Float32)