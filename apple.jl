N = 2^25

using Metal 
using BenchmarkTools 

println("On GPU (Metal)")
gpu_vector = Metal.ones(N)
function add_metal!(x)
    Metal.@sync mapreduce(exp, +, x)
    return
end
@btime add_metal!(gpu_vector)

println("On CPU")
cpu_vector = ones(N) 
function add_cpu!(x)
    mapreduce(exp, +, x)
    return
end
@btime add_cpu!(cpu_vector)
