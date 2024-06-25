include("sais.jl") 
include("toy_normal.jl")
include("simple_mixture.jl")

backend = CPU()
ais(SimpleMixture(backend); elt_type = Float32)