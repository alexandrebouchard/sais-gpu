include("bench_variance_utils.jl")

using CairoMakie 
using AlgebraOfGraphics

result = DataFrame(CSV.File("bench_variance.csv"; delim = ";"))

# function plot(result) 
#     result = subset(result, :type => t -> t .!= :FixedSchedule)
#     groups = groupby(result, [:T, :type, :backend])
#     transformed = combine(groups, :lognorm => rmse)
#     data(transformed) *
#         visual() * 
#         mapping(
#             :T, #:time, 
#             :lognorm_rmse, 
#             #marker = :scheme,
#             color = :type,
#             row = :backend,
#         )

# end

function plot_vars(result) 
    #result = subset(result, :type => t -> t .!= :FixedSchedule)
    groups = groupby(result, [:type, :backend, :scheme])
    summaries = combine(groups, 
                    :lognorm => (x -> var(relative_z_hat.(x))) => :variance,
                    :time => mean => :mean_time
                )
    data(summaries) * 
        mapping(
            :mean_time, 
            :variance, 
            color = :type,
            row = :backend,
        )
end

p = plot_vars(result)
axis = (width = 225, height = 225, yscale = log10, xscale = log10)
fg = draw(p; axis)
save("bench_variance.png", fg)