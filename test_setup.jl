using KernelAbstractions 
using CUDA
include("utils.jl")

backends = gpu_available() ? [CPU(), CUDABackend()] : [CPU()]

if !gpu_available() 
    @warn "Skipping GPU tests"
end