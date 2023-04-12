using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff,  LineSearches, Plots, Printf

using Distributions
using Plots

Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f)


μ = [0, 0]
Σ = [1  0.0;
     0.0 1]

p = MvNormal(μ, Σ)

size_arr = 12.0
X = -1*size_arr/2.0:1*size_arr/2.0
Y = -1*size_arr/2.0:1*size_arr/2.0 

z = [pdf(p, [x,y]) for y in Y, x in X]

heatmap(z, aspect_ratio=1)


true_vals =   [0.0, 0.0, 5.0, 1.0]

sample_data = z./maximum(z) #rand(11,12)

f = get_function(sample_data; super_sampling=2, extrapolation_bc=0.0);
#f(p0::Vector{Float64}) = f([p0[1], p0[2], p0[3], p0[4]])

fitting_data = f(true_vals) .+ rand(13, 13)./100.0
#f(p2[1], p2[2]) = f(p2::Vector{Tuple{Float64, Float64}})
loss(p, z) = sum(abs2.(f(p, z) .- fitting_data))
#loss(p2::Vector{Tuple{Float64, Float64}}) = loss(p2[1], p2[2])
loss(p3) = loss([p3[1], p3[2]], [p3[3], p3[4]])

heatmap(fitting_data, aspect_ratio=1)


a, b = Tuple(argmax(fitting_data)) .- 7
init_x =      [a, b, 1.0, 1.0]

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
       Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
       autodiff = :forward
)

begin
    p00 = heatmap(sample_data, aspect_ratio=1.0, clim=(0.0, 1.0), title="Sample data", legend = :none);
    p01 = heatmap(fitting_data, aspect_ratio=1.0, clim=(0.0,1.0), title="Fitting data", legend = :none);
    p02 = heatmap(f(Optim.minimizer(res)), aspect_ratio=1.0, clim=(0.0,1.0), title="estimated fit", legend = :none);
    p03 = heatmap(fitting_data .- f(Optim.minimizer(res)), aspect_ratio=1.0, clim=(0.0, 0.3), title="discrepancy", legend = :none);

    plot(p00, p01, p02, p03, layout=@layout([A B C D]), 
        framestyle=nothing, showaxis=false, 
        xticks=false, yticks=false, 
        size=(700, 300),  
        plot_title="True vals: $(true_vals)
  est vals: $(Optim.minimizer(res))",
        plot_titlevspan=0.25
    )
end


println(string(true_vals) * "\n" * string(Optim.minimizer(res)))
