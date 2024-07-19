using KernelAbstractions 
using CUDA
using Adapt 

struct TestStruct{T}
    t::T 
end
Adapt.@adapt_structure TestStruct

@kernel function test_(structs, outs) 
    n = @index(Global)

    state = structs[n]
    outs[n] = do_something(state)
end


function do_something(state) 
    return state.t[1]
end



backend = #CPU() 
    CUDABackend()

test = test_(backend) 

N = 2
outputs = KernelAbstractions.zeros(backend, Int, N)

to_device(x, ::CPU) = x 
to_device(x, ::CUDABackend) = cu(x)

raw_inputs = [TestStruct(to_device([i, i+1], backend)) for i in 1:N]
inputs = to_device(raw_inputs, backend)
@show typeof(inputs)

#inputs = KernelAbstractions.allocate(backend, TestStruct{})

test(inputs, outputs; ndrange = N)

@show outputs