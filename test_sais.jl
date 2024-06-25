include("sais.jl") 
include("toy_normal.jl")
include("simple_mixture.jl")

backend = CUDABackend()
ais(SimpleMixture(backend); elt_type = Float64, backend)