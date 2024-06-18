using SplittableRandoms
using Adapt

r = SplittableRandom(12)


N = 2^20

y_d = CUDA.fill(2.0f0, N)

function my_f(x)
    exp(x), x
end

map(my_f, y_d)




cu([SplittableRandom(1),SplittableRandom(2)])