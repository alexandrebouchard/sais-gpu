include("zja.jl")
include("sais.jl")
include("simple_mixture.jl")

Λ = 7.
T = 256
div = (Λ / T)^2

backend = CUDABackend()
target = SimpleMixture(backend) 

@show a1 = ais(target, SAIS(8); backend) 
@show a2 = ais(target, ZJA(div); backend)

@show a2.full_timing.time / a1.full_timing.time