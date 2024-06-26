include("sais.jl") 
include("toy_normal.jl")
include("simple_mixture.jl")

backend = CUDABackend()
target = #NormalPath(1)
    SimpleMixture(backend)
ais(target, SAIS(10); elt_type = Float64, backend)