include("zja.jl")
include("sais.jl")
include("simple_mixture.jl")
using DataFrames
using AlgebraOfGraphics
using CairoMakie 

function predict_t() 
    backend = CPU() 
    target = SimpleMixture(backend) 
    Λ = 7.1 # from SAIS run 
    # formula at equi-divergence (section 5.1)
    #   Λ = T √div 
    #   => T = Λ / √div 
    div = 0.00001 
    scheduler = ZJA(div)
    a = ais(target, scheduler; backend, elt_type = Float64) 
    @show empirical = length(a.schedule)
    @show prediction = Λ / sqrt(div)
    @assert isapprox(empirical, prediction; rtol=0.01)
end
predict_t()

function compute_both()
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
    return result 
end

function plot(result)
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
end


# result = compute_both() 
# plot(result) 

