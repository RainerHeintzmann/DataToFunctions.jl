using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff, LineSearches, Plots, Printf
using View5D
using Distributions
using Plots


Base.show(io::IO, f::Float64) = @printf(io, "%.3f", f)

# defining the mean and the varixance of the test normal (Gaussian) distribution
μ = [0, 0]
Σ = [2  0.0;
     0.0 2]

Σ_d = [2  1.5;
     1.5 2]

# initializing the multivariate normal distribution
p = MvNormal(μ, Σ)

# size of the test array to fit
size_arr = 12.0

# this part of the code is to define the sample array based on a 2D normal distribution
X = -1*size_arr/2.0:1*size_arr/2.0
Y = -1*size_arr/2.0:1*size_arr/2.0 

z = [pdf(p, [x,y]) for y in Y, x in X]

@vv z

heatmap(z, aspect_ratio=1)

# setting a typical values for the shift (1:2) and scale (3:4)
# true_vals =   [1.15, -0.73, 2.1, 0.9, 0.0, 0.0, pi/6]
true_vals =   [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, pi/6]

# normalizing the sample data
sample_data = z./maximum(z) 

# converting the data to function (DataToFunctions.get_function)
f_general = get_function_general(sample_data; super_sampling=1);#, extrapolation_bc=0.0);
# f_d = get_function_loop(sample_data_d; super_sampling=1);#, extrapolation_bc=0.0);

# adding some scaled random noise to the fitting data
fitting_data = f_general(true_vals) .+ rand(13, 13)./10.0

# @vv fitting_data

# defining the loss function based on the gaussian noise
loss(p) = sum(abs2.(f_general(p) .- fitting_data))
# loss(x) = loss(x::AbstractVector{T} where T)

# loss(p3) = loss([p3[1], p3[2], p3[3], p3[4], p3[5], p3[6], p3[7]])

heatmap(fitting_data, aspect_ratio=1)

# perform the main fit to the fitting data by minimizing the loss function
output, res = perform_fit_general(loss, fitting_data)

# plotting the output of the fitting pocedure for further illustration
begin
    p00 = heatmap(sample_data, aspect_ratio=1.0, clim=(0.0, 1.0), title="Sample data", legend = :none);
    p01 = heatmap(fitting_data, aspect_ratio=1.0, clim=(0.0,1.0), title="Fitting data", legend = :none);
    p02 = heatmap(f_general(output), aspect_ratio=1.0, clim=(0.0,1.0), title="estimated fit", legend = :none);
    p03 = heatmap(fitting_data .- f_general(output), aspect_ratio=1.0, clim=(0.0, 0.3), title="discrepancy", legend = :none);

    plot(p00, p01, p02, p03, layout=@layout([A B C D]), 
        framestyle=nothing, showaxis=false, 
        xticks=false, yticks=false, 
        size=(1200, 500),  
        plot_title="True vals: $(true_vals)
  est vals: $(output)",
        plot_titlevspan=0.25
    )
end

# comparing the true values to the best fitting parameters 
println(string(true_vals) * "\n" * string(output))



anim = @animate for i1 in 1:length(Optim.x_trace(res))

    begin
        p00 = heatmap(sample_data, aspect_ratio=1.0, clim=(0.0, 1.0), title="Sample data", legend = :none);
        p01 = heatmap(fitting_data, aspect_ratio=1.0, clim=(0.0,1.0), title="Fitting data", legend = :none);
        p02 = heatmap(f_general(Optim.x_trace(res)[i1]), aspect_ratio=1.0, clim=(0.0,1.0), title="estimated fit", legend = :none);
        p03 = heatmap(fitting_data .- f_general(Optim.x_trace(res)[i1]), aspect_ratio=1.0, clim=(0.0, 0.3), title="discrepancy", legend = :none);
    
        plot(p00, p01, p02, p03, layout=@layout([A B C D]), 
            framestyle=nothing, showaxis=false, 
            xticks=false, yticks=false, 
            size=(1200, 500),  
            plot_title="iteration: $(Int(i1))/$(length(Optim.x_trace(res))),
            estimation: $(Optim.x_trace(res)[i1])
            true vals : $(true_vals)",
            plot_titlevspan=0.25
        )
    end


end;

gif(anim, "examples/anim_general_1.mp4", fps=20)