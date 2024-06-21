include("toy_normal.jl") 


T = 1000 
D = 1
target = NormalPath(D)


for backend in [CUDABackend(), CPU()]
    @show backend
    # warm-up 
    ais(target; T, N=10, backend) 

    # measure
    for N in map(i -> 2^i, 5:15)
        @show N 

        timings = map(1:2) do _
            _, t = ais(target; T, N, backend) 
            t.time 
        end
        @show timings
    end
end