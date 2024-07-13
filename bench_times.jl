##### temp: investigate weird behaviour of ZHA

# function create_zha_fig(result) 
#     sub = subset(result, :type => ByRow(!=("FixedSchedule")))

#     p = data(sub) * 
#         AlgebraOfGraphics.density() *
#         mapping(
#             :time, 
#             color = :type,
#             row = :scheme,
#             col = :backend
#         )
#     axis = (width = 225, height = 225, xscale = log10)
#     fg = draw(p; axis)
#     save("zha.png", fg)
# end
# create_zha_fig(result) 