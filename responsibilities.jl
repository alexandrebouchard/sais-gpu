# CPU back-end
responsibilities(array::Array) = eachindex(array)

# GPU back-end 
function responsibilities(array::CuDeviceArray)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    return index:stride:length(array)
end
