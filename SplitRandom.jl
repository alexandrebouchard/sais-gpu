# Like SplittableRandom, but with all the splitting only allowed at construction. 
# This restriction makes it suitable to GPUs.

using SplittableRandoms
using CUDA 
using Random
using Base: rand
using Random: Random, AbstractRNG, RandomDevice, rng_native_52, SamplerUnion

using Adapt 

struct SplitRandomArray{T}
    array::T 
end 
Adapt.@adapt_structure SplitRandomArray

# Pass this to the kernel
function SplitRandomArray(size::Int; gpu::Bool = false, seed = 1) 
    base_rng = splittable(seed)
    rng_array = [SplittableRandoms.split(base_rng) for _ in 1:size] 
    storage = Array{UInt64}(undef, 2, size)
    for i in 1:size 
        storage[1, i] = rng_array[i].seed 
        storage[2, i] = rng_array[i].gamma
    end 
    return SplitRandomArray(gpu ? cu(storage) : storage)
end

# Create those as stack-allocated
struct SplitRandom{T} <: AbstractRNG
    # view into the array
    # entry 1 is SplittableRandom.seed
    # entry 2 is SplittableRandom.gamma
    view::T 
end
SplitRandom(sra::SplitRandomArray, task::Int) = SplitRandom(@view sra.array[:, task])


# support functions

splittable(seed::Int) = SplittableRandom(seed) 
splittable(rng::SplittableRandom) = rng

next_seed!(sr::SplitRandom) = sr.view[1] += sr.view[2]

Base.rand(sr::SplitRandom{A}, ::Type{UInt64}) where {A} = SplittableRandoms.mix64(next_seed!(sr))
Random.rng_native_52(::SplitRandom{A}) where {A} = UInt64

@inline function Base.rand(
    rng::SplitRandom{A},
    T::Random.SamplerUnion(Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
    ) where {A}
    rand(rng, UInt64) % T[]
end

