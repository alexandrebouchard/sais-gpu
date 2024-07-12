include("bench_variance_utils.jl")

using CairoMakie 
using AlgebraOfGraphics
using Statistics

function plot_vars(result) 
    groups = groupby(result, [:type, :backend, :scheme, :model])
    @show summaries = combine(groups, 
                    :relnorm => var => :variance,
                    :time => mean => :mean_time,
                    :time => std => :std_time,
                    :T => mean => :mean_T,
                    :T => std => :std_T
                )
    data(summaries) * 
        (visual(Lines) + visual(Scatter)) *
        mapping(
            :mean_time => "Mean time (s)", 
            :variance => "Relative variance", 
            color = :type,
            row = :backend,
            col = :model,
        )
end

function create_fig(csv_path) 
    result = DataFrame(CSV.File(csv_path))
    p = plot_vars(result)
    axis = (width = 225, height = 225, yscale = log10, xscale = log10)
    fg = draw(p; axis)
    return fg
end

# fg = create_fig("nextflow/deliverables/bench_variance/aggregated/bench_variance_tmp.csv")
# save("bench_variance.png", fg)
##
