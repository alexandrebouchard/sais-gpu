using CUDA
using BenchmarkTools

include("SplitRandom.jl") 

@kernel function test!(rngs, out) 
    i = @index(Global)
    rng = rngs[i]
    out[i] = randn(rng, Float32)
end

function test_split_repro()
    N = 100
    prev = nothing
    for backend in backends()
        @show backend
        for i in 1:5
            rngs = SplitRandomArray(N; backend)
            out = KernelAbstractions.zeros(backend, Float32, N)

            test!(backend)(rngs, out, ndrange=size(out))
            KernelAbstractions.synchronize(backend)
            
            check = zeros(Float32, N)
            copyto!(check, out)
            if !isnothing(prev)
                @assert check ≈ prev
            end 
            prev = check
        end
    end
end

function test_split_moments() 
    N = 2^20
    for backend in backends()
        rngs = SplitRandomArray(N; backend)
        out = KernelAbstractions.zeros(backend, Float32, N)
        #@btime begin 
            test!(backend)(rngs, out, ndrange=size(out))
            KernelAbstractions.synchronize(backend)
        #end
        @assert isapprox(mean(out), 0, atol = 2.0/sqrt(N))
        @assert isapprox(std(out), 1, atol = 2.0/sqrt(N))
    end
end

test_split_moments() 
test_split_repro()