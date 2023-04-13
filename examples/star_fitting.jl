using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff, LineSearches, Plots, Printf
using View5D
using Distributions
using Plots

Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f)

# defining the mean and the varixance of the test normal (Gaussian) distribution
μ = [0, 0]
Σ = [2  0.0;
     0.0 2]

# initializing the multivariate normal distribution
p = MvNormal(μ, Σ)

# size of the test array to fit
size_arr = 12.0

# this part of the code is to define the sample array based on a 2D normal distribution
X = -1*size_arr/2.0:1*size_arr/2.0
Y = -1*size_arr/2.0:1*size_arr/2.0 

z = [pdf(p, [x,y]) for y in Y, x in X]

@vv z

# setting a typical values for the shift (1:2) and scale (3:4)
true_vals =   [1.1, -1.5, 0.75, 1.5]

# normalizing the sample data
sample_data = z./maximum(z) 

# converting the data to function (DataToFunctions.get_function)
f = get_function(sample_data; super_sampling=2, extrapolation_bc=0.0);

# adding some scaled random noise to the fitting data
fitting_data = f(true_vals) .+ rand(13, 13)./10.0

@vv fitting_data

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
