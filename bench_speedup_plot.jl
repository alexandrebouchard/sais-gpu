include("bench_variance_utils.jl")

using CairoMakie 
using AlgebraOfGraphics
using Statistics

## Variances/time fig

function plot_speedup(result) 
    groups = groupby(result, [:backend, :model, :type, :N, :elt_type])
    summaries = combine(groups, 
                    :time => mean => :mean_time,
                )
    normalized = copy(summaries)
    for row in eachrow(normalized)
        # get time for corresponding CPU backend 
        selected = subset(summaries, 
            :backend => ByRow(==("CPU")),
            :model => ByRow(==(row[:model])),
            :type => ByRow(==(row[:type])),
            :N => ByRow(==(row[:N])),
            :elt_type => ByRow(==(row[:elt_type])))
        normalization = selected[1, :mean_time]
        # normalize by that
        row[:mean_time] = row[:mean_time]/normalization
    end
    normalized = subset(normalized, :backend => ByRow(==("GPU")))
    sort!(normalized)
    @show normalized
    data(normalized) * 
        (visual(Lines) + visual(Scatter)) *
        mapping(
            :N => "Number of particles", 
            :mean_time => "Speed-up", 
            color = :type,
            col = :model,
            row = :elt_type
        )
end

function create_speedup_fig(result) 
    p = plot_speedup(result)
    axis = (width = 225, height = 225, yscale = log10, xscale = log10)
    fg = draw(p; axis)
    return fg
end

result = DataFrame(CSV.File("nextflow/deliverables/bench_speedup_2024-07-12T17:05:01.785723-07:00/bench_speedup.csv"))