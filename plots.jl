include("ais.jl")
using CairoMakie

plot_barrier(ais) =
    lines(0.01..1, x -> ais.barriers.localbarrier(x))


plot_cumulative(ais) =  
    lines(0..1, x -> ais.barriers.cumulativebarrier(x))



# using CairoMakie 
# p = test_mix_barrier() 
# save("test_mix_barrier.png", p)
