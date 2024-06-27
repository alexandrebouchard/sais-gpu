include("sais.jl") 
include("toy_normal.jl")
include("simple_mixture.jl")


for backend in backends 
    for target in [NormalPath(2), SimpleMixture(backend)]
        for seed in [1, SplittableRandom(1)]
            for multi_threaded in (backend isa CPU ? [true, false] : [true])
                for elt_type in [Float32, Float64]
                    ais(target, SAIS(3); backend, seed, multi_threaded, elt_type)
                end
            end
        end
    end
end
