# Like SplittableRandom, but with all the splitting only allowed at construction. 
# This restriction makes it suitable to GPUs.

using SplittableRandoms
using KernelAbstractions 
using Random
using Base: rand
using Random: Random, AbstractRNG, RandomDevice, rng_native_52, SamplerUnion

using Adapt 

struct SplitRandomArray{T}
    array::T 
end 
Adapt.@adapt_structure SplitRandomArray

# Pass the array to the kernel
function SplitRandomArray(size::Int; backend::Backend = CPU(), seed = 1) 
    base_rng = splittable(seed)
    rng_array = [SplittableRandoms.split(base_rng) for _ in 1:size] 
    init = Array{UInt64}(undef, 2, size)
    for i in 1:size 
        init[1, i] = rng_array[i].seed 
        init[2, i] = rng_array[i].gamma
    end 
    result = KernelAbstractions.zeros(backend, UInt64, 2, size) 
    copyto!(result, init)
    return SplitRandomArray(result)
end

Base.getindex(sra::SplitRandomArray, index::Int) = SplitRandom(index, sra.array)


# support functions

Base.length(sra::SplitRandomArray) = size(sra.array)[2]

# essentially a view into an array 
struct SplitRandom{T} <: AbstractRNG
    task::Int
    # entry 1 is SplittableRandom.seed
    # entry 2 is SplittableRandom.gamma
    array::T 
end

splittable(seed::Int) = SplittableRandom(seed) 
splittable(rng::SplittableRandom) = rng

next_seed!(sr::SplitRandom) = sr.array[1, sr.task] += sr.array[2, sr.task]

Base.rand(sr::SplitRandom{A}, ::Type{UInt64}) where {A} = SplittableRandoms.mix64(next_seed!(sr))
# GPU friendly box-muller transform
Base.randn(sr::SplitRandom{A}, ::Type{Float32}) where {A} = sqrt(-2log(rand(sr, Float32))) * cos(rand(sr, Float32) * 2f0*pi)

Random.rng_native_52(::SplitRandom{A}) where {A} = UInt64

@inline function Base.rand(
    rng::SplitRandom{A},
    T::Random.SamplerUnion(Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
    ) where {A}
    rand(rng, UInt64) % T[]
end
