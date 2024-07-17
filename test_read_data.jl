include("logistic_regression_data.jl")


function test_read_data(file)
    @show file

    # test subsetting
    mat_x, vec_y = load_and_preprocess_data(; file, requested_size = 30, requested_dim = 50, standardize = false)
    @assert size(mat_x) == (30, 50+1)
    sub_x, sub_y = load_and_preprocess_data(; file, requested_size = 20, requested_dim = 30, standardize = false)
    @assert mat_x[1:20, 1:31] == sub_x

    load_and_preprocess_data(; file)
end

# test with synthetic
test_read_data(nothing)

# test all datasets
foreach_mat_file("data") do file
    test_read_data(file)
end

