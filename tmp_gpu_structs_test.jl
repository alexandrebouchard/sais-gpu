using KernelAbstractions 



@kernel function test(structs, outs) 
    n = @index(Global)

    state = structs[n]
    out = outs[n]
    do_something(state, out)

end


function do_something(state, out) 


end

backend = CUDABackend()