include("sais.jl")
include("simple_mixture.jl")

function test_profile()
    for backend in backends()
        target = SimpleMixture(backend) 
        a = ais(target, SAIS(5); backend) 
        @assert percent_time_in_kernel(a) > (backend isa CPU ? 0.99 : 0.9) 
    end
end
test_profile()