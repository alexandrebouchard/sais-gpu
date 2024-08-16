using MAT
using SplittableRandoms
using Random
using Statistics
using StatsBase


# description of data: https://jundongl.github.io/scikit-feature/datasets.html
function describe_datasets(directory = "data")
   foreach_mat_file(directory) do file
      data = matread(file)
      x = data["X"] 
      y = data["Y"]
      n, p = size(x)
      n_zeros = sum(x -> x == 0 ? 1 : 0, x)
      sparsity = n_zeros/n/p
      println(
         """
         $file
            n, p: $((n, p))
            labels: $(sort(unique(Int.(y))))
            label counts: $(counts(Int.(y)))
            sparsity: $sparsity

         """
      )
   end
end

is_one(x) = Int(x == 1 ? 1 : 0)
function load_and_preprocess_data(;
      file::Union{String,Nothing} = nothing, # nothing for synthetic data
      generated_dim = 2^10,
      generated_size = 2^10,
      requested_dim = nothing, 
      requested_size = nothing, 
      rng = SplittableRandom(1), 
      add_intercept = true,
      binarize = is_one,
      standardize = true
   )

   data_sim_rng = SplittableRandoms.split(rng)
   if isnothing(file)
      data = create_logistic_data(data_sim_rng, generated_size, generated_dim)
   else
      data = matread(file)
   end

   mat_x = data["X"]
   vec_y = data["Y"][:, begin]
   original_data_size, original_data_dim = size(mat_x)

   dim_rng = SplittableRandoms.split(rng)
   if !isnothing(requested_dim)
      @assert requested_dim ≤ original_data_dim
      d_idx = randperm(dim_rng, original_data_dim)[1:requested_dim]
      mat_x = mat_x[:, d_idx]
   end

   instances_rng = SplittableRandoms.split(rng)
   if !isnothing(requested_size)
      @assert requested_size ≤ original_data_size
      n_idx = randperm(instances_rng, original_data_size)[1:requested_size]
      mat_x = mat_x[n_idx, :]
      vec_y = vec_y[n_idx]
   end
   
   data_size, data_dim = size(mat_x)

   if standardize 
      for c in eachcol(mat_x)
         s = std(c)
         if s > 0 
            c .= (c .- mean(c)) ./ s
         else
            @warn "Trying to standardize a column with zero standard deviation" maxlog=1
         end
      end
   end
   if add_intercept # after standardization!
      mat_x = hcat(ones(data_size), mat_x) 
   end

   converted_y = isnothing(binarize) ? Int.(vec_y) : binarize.(vec_y)

   return mat_x, converted_y
 end

function create_logistic_data(rng, n, d)
   β = randn(rng, d)
   x_mat = randn(rng, n, d)
   p_vec = 1 ./ (1 .+ exp.( .- x_mat*β))
   y_vec = rand(rng, n) .< p_vec
   # conform with real dataset format:
   return Dict("X" => x_mat, "Y" => y_vec)
end

foreach_mat_file(f, directory) = 
   for file in readdir(directory)
      if endswith(file, "mat")
         f(joinpath(directory, file))
      end
   end