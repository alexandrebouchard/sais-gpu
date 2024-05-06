N = 2^25

using CUDA 
using BenchmarkTools 

println("On GPU (CUDA)")
gpu_vector = CUDA.ones(N)
function add_cuda!(x)
    CUDA.@sync mapreduce(exp, +, x)
    return
end
@btime add_cuda!(gpu_vector)

println("On CPU")
cpu_vector = ones(Float32, N) 
function add_cpu!(x)
    mapreduce(exp, +, x)
    return
end
@btime add_cpu!(cpu_vector)
