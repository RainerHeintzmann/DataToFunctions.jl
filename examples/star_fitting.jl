using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff, LineSearches, Plots, Printf
using View5D
using Distributions
using Plots

using Rotations
using CoordinateTransformations

Base.show(io::IO, f::Float64) = @printf(io, "%.3f", f)

# defining the mean and the varixance of the test normal (Gaussian) distribution
μ = [0, 0]
Σ = [2  0.0;
     0.0 2]

Σ_d = [2  1.5;
     1.5 2]

# initializing the multivariate normal distribution
p = MvNormal(μ, Σ)
p_d = MvNormal(μ, Σ_d)

# size of the test array to fit
size_arr = 12.0

# this part of the code is to define the sample array based on a 2D normal distribution
X = -1*size_arr/2.0:1*size_arr/2.0
Y = -1*size_arr/2.0:1*size_arr/2.0 

z = [pdf(p, [x,y]) for y in Y, x in X]
z_d = [pdf(p_d, [x,y]) for y in Y, x in X]

# @vv z
heatmap(z, aspect_ratio=1)
heatmap(z_d, aspect_ratio=1)

# setting a typical values for the shift (1:2) and scale (3:4)
true_vals =   [1.1, -1.5, 0.75, 1.5, pi/2]

# normalizing the sample data
sample_data = z./maximum(z) 
sample_data_d = z_d./maximum(z_d)

# converting the data to function (DataToFunctions.get_function)
f = get_function(sample_data; super_sampling=2, extrapolation_bc=0.0);
f_d = get_function(sample_data_d; super_sampling=2, extrapolation_bc=0.0);

# adding some scaled random noise to the fitting data
fitting_data = f_d(true_vals) .+ rand(13, 13)./40.0

# @vv fitting_data

# defining the loss function based on the gaussian noise
loss(p) = sum(abs2.(f(p) .- fitting_data))

#loss(p3) = loss([p3[1], p3[2]], [p3[3], p3[4]])

heatmap(fitting_data, aspect_ratio=1)

# perform the main fit to the fitting data by minimizing the loss function
output = perform_fit(loss, fitting_data)

# plotting the output of the fitting pocedure for further illustration
begin
    p00 = heatmap(sample_data, aspect_ratio=1.0, clim=(0.0, 1.0), title="Sample data", legend = :none);
    p01 = heatmap(fitting_data, aspect_ratio=1.0, clim=(0.0,1.0), title="Fitting data", legend = :none);
    p02 = heatmap(f(output), aspect_ratio=1.0, clim=(0.0,1.0), title="estimated fit", legend = :none);
    p03 = heatmap(fitting_data .- f(output), aspect_ratio=1.0, clim=(0.0, 1.0), title="discrepancy", legend = :none);

    plot(p00, p01, p02, p03, layout=@layout([A B C D]), 
        framestyle=nothing, showaxis=false, 
        xticks=false, yticks=false, 
        size=(700, 300),  
        plot_title="True vals: $(true_vals)
  est vals: $(output)",
        plot_titlevspan=0.25
    )
end

# comparing the true values to the best fitting parameters
println(string(true_vals) * "\n" * string(output))
