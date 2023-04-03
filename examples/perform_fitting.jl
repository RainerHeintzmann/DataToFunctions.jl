using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff,  LineSearches


true_vals = [0.0, 0.0, 1.0, 0.95]


sample_data = rand(11,12)

f = get_function(sample_data; super_sampling=2);
fitting_data = f([0.0, 0.0], [1.0, 0.95])
#f(p2[1], p2[2]) = f(p2::Vector{Tuple{Float64, Float64}})
loss(p, z) = sum(abs2.(f(p, z) .- fitting_data))
loss(p2::Vector{Tuple{Float64, Float64}}) = loss(p2[1], p2[2])
loss(p3) = loss([p3[1], p3[2]], [p3[3], p3[4]])

initial_x = [(0.0, 0.0), (1.0, 1.0)]

loss(initial_x)

init_x = [0.0, 0.0, 1.1, 0.8]

#Zygote.forwarddiff(loss, init_x)

ForwardDiff.gradient(loss, init_x)#true_vals .+ [0.0001, 0.0, 0.0, 0.0])
#conf = ForwardDiff.GradientConfig(f, init_x, chunk::Chunk = Chunk(init_x))


"""
BFGS(; alphaguess = Optim.LineSearches.InitialStatic(),
       linesearch = Optim.LineSearches.HagerZhang(),
       initial_invH = nothing,
       initial_stepnorm = 0.001,
       manifold = Optim.Flat()
       )

GradientDescent(; alphaguess = 0.01,
       linesearch = Optim.LineSearches.HagerZhang(),
       P = nothing,
       precondprep = (P, x) -> nothing
)

lower = [-10.0, -10.0, 0.00001, 0.00001]
upper = [10.0, 10.0, 10.0, 10.0]
#initial_x = [2.0, 2.0]
# requires using LineSearches
inner_optimizer = LBFGS(; linesearch=LineSearches.BackTracking(order=2))
res = optimize(loss, lower, upper, init_x, Fminbox(inner_optimizer), Optim.Options(store_trace = true, extended_trace = true, iterations=500), autodiff = :forward)
"""

res = optimize(loss, init_x, LBFGS(), Optim.Options(store_trace=true, extended_trace=true, iterations=500), autodiff = :forward)
trace = Optim.trace(res)





Optim.minimizer(res)
Optim.f_trace(res)
Optim.x_trace(res)[end]
Optim.converged(res)
Optim.g_norm_trace(res)
Optim.g_calls(res)

loss(Optim.x_trace(res)[end])
ForwardDiff.gradient(loss, Optim.x_trace(res)[end])

Optim.trace(res)

