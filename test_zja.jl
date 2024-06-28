include("zja.jl")
include("sais.jl")
include("simple_mixture.jl")
using DataFrames
using AlgebraOfGraphics
using CairoMakie 

backend = CPU() 
target = SimpleMixture(backend) 

result = DataFrame(
    scheme = Symbol[],
    normalized_index = Float64[], 
    beta = Float64[]
)

scheduler_label(::ZJA) = :ZJA 
scheduler_label(::SAIS) = :SAIS 
for scheduler in [ZJA(0.00001), SAIS(12)]
    a = ais(target, scheduler; backend, elt_type = Float64) 
    @show a
    for i in eachindex(a.schedule) 
        push!(result, (; 
            scheme = scheduler_label(scheduler), 
            normalized_index = Float64(i) / length(a.schedule), 
            beta = a.schedule[i]))
    end
end

p = data(result) * 
    visual(Lines) *
    mapping(
        :normalized_index, 
        :beta, 
        color = :scheme
    )
axis = (width = 225, height = 225)
fg = draw(p; axis)
save("test_zja.png", fg)


### TODO: check the path is linear (i.e. exclude the Normal example)
# maybe OK, ToyNormal not providing the log_prior

# target = SimpleMixture(CPU()) 
# a = ais(target, ZJA(0.01))
# @show a.schedule 

# TODO:
#     - test it crashes with toy normals 
#     - test schedule agree (number of points and locations)

# test will be: compare the learned schedules!

# function test_agreement()
#     target = SimpleMixture(CPU()) 
#     schedule = [0.0, 0.2, 0.3, 1.0]

#     # compute divergences off line
#     a = ais(target, schedule; N = 5, explorer = RWMH(0), compute_barriers = true) 
#     divergences = (a.intensity) .^ 2

#     # check agreement of the online code
#     log_weights = cumsum(log_increments, dims = 1)
# end
# test_agreement()