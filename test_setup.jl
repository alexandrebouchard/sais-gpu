using KernelAbstractions 
using CUDA
include("utils.jl")



if !gpu_available() 
    @warn "Skipping GPU tests"
end