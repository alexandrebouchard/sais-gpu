using CUDA 
using KernelAbstractions 

backend = CUDABackend()

mtx = KernelAbstractions.zeros(backend, Float32, 2, 10)

CUDA.randn!(mtx)

reduce(+,  mtx, dims = 2)