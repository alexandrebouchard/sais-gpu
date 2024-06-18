using CUDA

struct MyStruct{T} 
    array::T 
end 

function operation(s::MyStruct) 
    s.array[1] += exp(s.array[1])
end

function test(array) 
    i = threadIdx().x 
    m = MyStruct(@view array[:, i])
    operation(m)
    return
end

N = 5
array = cu(ones(2, N)) 

@cuda threads=N test(array)

@show array