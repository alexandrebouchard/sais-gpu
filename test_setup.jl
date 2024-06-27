using KernelAbstractions 
using CUDA

gpu_available = try 
    CUDA.driver_version()
    true
catch 
    false
end

backends = gpu_available ? [CPU(), CUDABackend()] : [CPU()]

if !gpu_available 
    @warn "Skipping GPU tests"
end