N = 2^25

using Metal 
using BenchmarkTools 

println("On GPU (Metal)")
gpu_vector = Metal.ones(N)
function add_metal!(x)
    Metal.@sync sum(exp, x)
    return
end
@btime add_metal!(gpu_vector)

println("On CPU")
cpu_vector = ones(Float32, N) 
function add_cpu!(x)
    sum(exp, x)
    return
end
@btime add_cpu!(cpu_vector)
