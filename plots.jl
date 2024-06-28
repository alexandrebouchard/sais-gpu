include("ais.jl")
using CairoMakie

function plot_barrier(ais)
    plot = lines(0.01..1, x -> ais.barriers.localbarrier(x))
    return plot
end


#      
#      barriers = a.barriers 
#      return 
# end

# using CairoMakie 
# p = test_mix_barrier() 
# save("test_mix_barrier.png", p)
