using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff,  LineSearches, Plots, Printf


Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f)

true_vals =   [1.0, 2.0, 1.0, 1.0]
init_x =      [0.2, 0.5, 1.0, 1.0]

sample_data = rand(11,12)

f = get_function(sample_data; super_sampling=2);
#f(p0::Vector{Float64}) = f([p0[1], p0[2], p0[3], p0[4]])

fitting_data = f(true_vals) .+ rand(11, 12)./10.0
#f(p2[1], p2[2]) = f(p2::Vector{Tuple{Float64, Float64}})
loss(p, z) = sum(abs2.(f(p, z) .- fitting_data))
loss(p2::Vector{Tuple{Float64, Float64}}) = loss(p2[1], p2[2])
loss(p3) = loss([p3[1], p3[2]], [p3[3], p3[4]])





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
"""
lower = [-1*size(fitting_data)[1], -1*size(fitting_data)[2], 0.0, 0.0]
upper = [size(fitting_data)[1], size(fitting_data)[2], size(fitting_data)[1], size(fitting_data)[2]]
#initial_x = [2.0, 2.0]
# requires using LineSearches
inner_optimizer = LBFGS(; m=1, linesearch=LineSearches.BackTracking(order=2))
res = optimize(
       loss, 
       lower, upper, 
       init_x, 
       Fminbox(inner_optimizer), 
       #Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
       autodiff = :forward
)




heatmap(fitting_data .- f(Optim.x_trace(res)[end]), aspect_ratio=1, clim=(0.0, 1.0))

"""

res = optimize(
       loss, init_x, 
       LBFGS(), 
       Optim.Options(store_trace=true, extended_trace=true, iterations=500), 
       autodiff = :forward
       )
"""

trace = Optim.trace(res);
trace




Optim.minimizer(res)
Optim.f_trace(res)
Optim.x_trace(res)
Optim.converged(res)
Optim.g_norm_trace(res)
Optim.g_calls(res)

loss(Optim.x_trace(res)[end])
ForwardDiff.gradient(loss, Optim.x_trace(res)[end])



anim = @animate for i1 in 1:length(Optim.x_trace(res))

       heatmap(fitting_data .- f(Optim.x_trace(res)[i1]), 
              aspect_ratio=1, 
              clim=(0.0, 1.0),
              dpi=300
       )
       title!("iteration: $(Int(i1))/$(length(Optim.x_trace(res))),
              estimation: $(Optim.x_trace(res)[i1])
              true vals : $(true_vals)")

end;

gif(anim, "anim1.mp4", fps=5)
