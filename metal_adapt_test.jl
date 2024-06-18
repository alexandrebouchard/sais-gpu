using Adapt
using Metal 


struct Test{T}
    array::T 
end 

struct Another{T} 
    array::T 
    loc::Int
end

Adapt.@adapt_structure Test 

@kernel function adapt_test(t) 
    i = @index(Global)
    test = Another(t.array, 1)
    test.array[1] = 2
end


backend = MetalBackend()
vector = KernelAbstractions.zeros(backend, Float32, 2)
t = Test(vector)

adapt_test(backend)(t, ndrange = size(vector))