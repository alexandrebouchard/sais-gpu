include("bench_variance_utils.jl")

using CairoMakie 
using AlgebraOfGraphics
using Statistics

## Variances/time fig

function drop_lowest_round(result)
    for model in unique(result.model)
        m = minimum(subset(result, :model => ByRow(==(model))).n_rounds)
        result = filter([:model, :n_rounds] => (mo, nr) -> mo != model || nr > m, result)
    end
    return result
end

function plot_vars(result) 
    result = drop_lowest_round(result)
    groups = groupby(result, [:backend, :model, :type, :n_rounds, :scheme])
    summaries = combine(groups, 
                    :relnorm => var => :variance,
                    :time => mean => :mean_time,
                    :time => std => :std_time,
                    :T => mean => :mean_T,
                    :T => std => :std_T
                )
    sort!(summaries)
    @show summaries
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

function create_vars_fig(result) 
    p = plot_vars(result)
    axis = (width = 225, height = 225, yscale = log10, xscale = log10)
    fg = draw(p; axis)
    return fg
end

# Not quite what we want: we would need for a single round, not the full cost:
# function plot_times(result) 
#     data(result) * 
#         visual(BoxPlot) * 
#         mapping(
#             :n_rounds => "Effort", 
#             :time => "Time (s)", 
#             color = :type,
#             row = :backend, 
#             col = :model
#         )
# end

# function create_times_fig(result) 
#     p = plot_times(result)
#     axis = (width = 225, height = 225, yscale = log10)
#     fg = draw(p; axis)
#     return fg
# end

# fg = create_times_fig(result) 
# save("temp.png", fg)