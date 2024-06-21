include("toy_normal.jl") 
using DataFrames
using AlgebraOfGraphics
using CairoMakie 

T = 1000 
D = 1
target = NormalPath(D)

function run_bench()
    result = DataFrame(
                N=Int[], 
                time=Float64[], 
                backend=Symbol[])
    for backend in [CUDABackend(), CPU()]
        # warm-up - do not include this one in results
        ais(target; T, N=10, backend) 
        # measure
        for N in map(i -> 2^i, 0:15)
            _, t = ais(target; T, N, backend) 
            push!(result, (; N, backend = backend_label(backend), time = t.time))
        end
    end
    return result
end
backend_label(::CPU) = :CPU 
backend_label(_) = :GPU

plot(result) =
    data(result) * 
        visual(Lines) *
        mapping(
            :N => "Number of particles", 
            :time => "Wallclock time (s)", 
            color = :backend) 


results = run_bench()
p = plot(results)
axis = (width = 225, height = 225, xscale = log2, yscale = log2)
fg = draw(p; axis)
save("bench_gpu_particles.png", fg)