#using CUDA

include("SplitRandom.jl") 

@kernel function test!(rngs, out) 
    i = @index(Global)
    rng = rngs[i]
    out[i] = randn(rng, Float32)
end


N = 10
backend = 
    CPU()
    #CUDABackend()
    #MetalBackend()

rngs = SplitRandomArray(N; backend)
out = KernelAbstractions.zeros(backend, Float32, N)

using BenchmarkTools
@btime begin 
    test!(backend)(rngs, out, ndrange=size(out))
    KernelAbstractions.synchronize(backend)
end

@show out

# include("SplitRandom.jl") 

# # simple(rng) = rand(rng, UInt64)

# # try 
# #     map(simple, SplitRandom(1, gpu = true))
# # catch err 
# #     global saved_err = err 
# # end

# # using Cthulhu
# # code_typed(saved_err; interactive = true)


# direct_test(sr::SplitRandom{A}) where {A} = SplittableRandoms.mix64(next_seed!(sr))



# function test(rngs, samples) 
#     i = threadIdx().x 
#     #samples[i] = direct_test(rngs[i])
#     sr = rngs[i]
#     #sr.array[1, sr.task_index] += sr.array[2, sr.task_index]
#     samples[i] = 2.0
#     return 
# end

# N = 2
# rngs = SplitRandomArray(N, gpu = true) 
# samples = cu(ones(N))

# # try 
# @cuda threads=N test(rngs, samples)
# # catch err
# #     code_typed(err; interactive = true)
# # end



# @show samples