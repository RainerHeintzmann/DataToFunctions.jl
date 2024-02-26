using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff, LineSearches, Plots, Printf
using View5D
using Distributions, Rotations
using Plots
using TestImages

Base.show(io::IO, f::Float64) = @printf(io, "%.3f", f)



"""
    perform_fit_general(loss_function, fitting_data::AbstractArray)

Performs a fit to the fitting data using a loss function defined by the user

# Arguments
`loss_function`: User-defined loss function which is minimized
`fitting_data`: The data which is being fitted 

# Returns
a vector of 7 parameters: 2 for the shift, 2 for the scaling, 2 for shear, and 1 for rotation angle

# Example
there is an example of this function in the `examples/star_fitting_genaral.jl`

"""
function perform_fit_general(loss_function, fitting_data::AbstractArray)
    # guess the shift parameters by taking the maximum values of the array and
    # centering the positions
    ##a, b = Tuple(argmax(fitting_data)) .- size(fitting_data) ./2.0 .- 1.0
    #print("INSIDE!!! hehe")
    # assigning the initial parameter estimates
    init_x = vec([1.0, -2.0, 1.0, 1.0, 0.0, 0.0, 0.0]) #ndims(fitting_data)+1, ndims(fitting_data)+1))
    # reshape(Matrix(1.0*I, ndims(fitting_data)+1, ndims(fitting_data)+1), 1, 9)) #[a, b, 1.0, 1.0, 0.001, 0.001, 0.001]

    # setting the lower and upper boundary of the parameter values based on their limits
    lower = [-1*size(fitting_data)[1], -1*size(fitting_data)[2], 0.0, 0.0, 0.0, 0.0, 0.0]
    upper = [size(fitting_data)[1], size(fitting_data)[2], size(fitting_data)[1], size(fitting_data)[2], 2.0, 2.0, pi]

    # initializing the LBFGS optimizer
    inner_optimizer = LBFGS(; m=1, linesearch=LineSearches.BackTracking(order=2))
    
    # Computer, Optimize! :D
    res = optimize(
            loss_function, 
            
            #BFGS(),
            lower, upper, init_x,
            Fminbox(inner_optimizer), 
            Optim.Options(store_trace = true, extended_trace = true, iterations=5000), 
            #autodiff = :forward
        )
    
    # return the estimated parameters
    return Optim.minimizer(res), res
end


"""
    perform_fit(loss_function, fitting_data::AbstractArray)

Performs a fit to the fitting data using a loss function defined by the user

# Arguments
`loss_function`: User-defined loss function which is minimized
`fitting_data`: The data which is being fitted 

# Returns
a vector of 4 parameters: first 2 for the shift and other 2 for the scaling factors

# Example
there is an example of this function in the `examples/star_fitting.jl`

"""
function perform_fit(loss_function, fitting_data::AbstractArray)
    # guess the shift parameters by taking the maximum values of the array and
    # centering the positions
    a, b = Tuple(argmax(fitting_data)) .- size(fitting_data) ./2.0 .- 1.0
    
    # assigning the initial parameter estimates
    init_x = [a, b, 1.0, 1.0]

    # setting the lower and upper boundary of the parameter values based on the limits of the shift and scaling
    lower = [-1*size(fitting_data)[1], -1*size(fitting_data)[2], 0.0, 0.0]
    upper = [size(fitting_data)[1], size(fitting_data)[2], size(fitting_data)[1], size(fitting_data)[2]]

    # initializing the LBFGS optimizer
    inner_optimizer = LBFGS(; m=10, linesearch=LineSearches.BackTracking(order=2))
    
    # Computer, Optimize! :D
    res = optimize(
            loss_function, 
            lower, upper, 
             init_x, 
            Fminbox(inner_optimizer), 
            Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
             autodiff = :forward
        )
    
    # return the estimated parameters
    return Optim.minimizer(res)
end




# size of the test array to fit
size_arr = 22

noise_level = 10.0;

# defining the mean and the varixance of the test normal (Gaussian) distribution
μ = [0, 0]
Σ = [size_arr/10  0.0;
     0.0 size_arr/10]

Σ_d = [2  1.5;
     1.5 2]

# initializing the multivariate normal distribution
p = MvNormal(μ, Σ)



# this part of the code is to define the sample array based on a 2D normal distribution
X = -1*size_arr/2.0:1*size_arr/2.0
Y = -1*size_arr/2.0:1*size_arr/2.0 

z = [pdf(p, [x,y]) for y in Y, x in X]

# @vv z



# setting a typical values for the shift (1:2) and scale (3:4)
true_vals =   [0.5, -1.5, 1.0, 1.0, 0.0, 0.0, pi/4]#pi/6]
#true_vals =   [2.3, -1.2, 0.9, 2.1, 0.1, 0.05, pi/2, 2.0, 3.0]

# normalizing the sample data
#sample_data = Float32.(z./maximum(z)) 
#sample_data = TestImages.shepp_logan(32);
sample_data = rand.(size_arr, size_arr);

sample_data .+= rand(size(sample_data)...)./noise_level;



x_cen, y_cen = (size(sample_data) .÷ 2.0 .+1)

shear_mat = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0];
scale_mat = [1/1.2 0.0 0.0; 0.0 1/1.8 0.0; 0.0 0.0 1.0];

t_to_origin = [1.0 0.0 1*x_cen; 0.0 1.0 y_cen; 0.0 0.0 1.0];
t_to_center = [1.0 0.0 -1.0*x_cen; 0.0 1.0 -1.0*y_cen; 0.0 0.0 1.0];

# converting the data to function (DataToFunctions.get_function)
f_general = get_function_affine(sample_data);#; super_sampling=1);#, extrapolation_bc=0.0);


ang = rand((0.0:0.1:pi))
rot_mat =  [cos(ang)  -1.0*sin(ang) 0.0; sin(ang)  cos(ang) 0.0; 0.0 0.0 1.0];

matrix_c = Float32.(t_to_origin * scale_mat * shear_mat * rot_mat * t_to_center )

# f_d = get_function_loop(sample_data_d; super_sampling=1);#, extrapolation_bc=0.0);

# adding some scaled random noise to the fitting data
# fitting_data = f_general(SMatrix{3,3}(matrix_c)); #.+ rand(size(sample_data)...)./100.0;
fitting_data = f_general(true_vals) .+ rand(size(sample_data)...)./10.0;


plot(heatmap(sample_data, aspect_ratio=1), heatmap(fitting_data, aspect_ratio=1))


# defining the loss function based on the gaussian noise
loss(p1::AbstractVector) = sum(abs2.(f_general(p1::AbstractVector) .- fitting_data))
# loss(x) = loss(x::AbstractVector{T} where T)

# loss(p3) = loss([p3[1], p3[2], p3[3], p3[4], p3[5], p3[6], p3[7]])


# perform the main fit to the fitting data by minimizing the loss function
@time output, res = perform_fit_general(loss, fitting_data)


# plotting the output of the fitting pocedure for further illustration
begin
    p00 = heatmap(sample_data, aspect_ratio=1.0, clim=(0.0, 1.0), title="Sample data", legend = :none);
    p01 = heatmap(fitting_data, aspect_ratio=1.0, clim=(0.0,1.0), title="Fitting data", legend = :none);
    p02 = heatmap(f_general(output), aspect_ratio=1.0, clim=(0.0,1.0), title="estimated fit", legend = :none);
    p03 = heatmap(fitting_data .- f_general(output), aspect_ratio=1.0, clim=(0.0, 1.0), title="discrepancy", legend = :none);

    plot(p00, p01, p02, p03, layout=@layout([A B C D]), 
        framestyle=nothing, showaxis=false, 
        xticks=false, yticks=false, 
        size=(1200, 500),  
        plot_title="True vals: $(true_vals)
  fitted vals: $(output)",
        plot_titlevspan=0.2
    )
end


matrix_c
# comparing the true values to the best fitting parameters 

#println(matrix_c)  * reshape(output, ndims(fitting_data)+1, ndims(fitting_data)+1)



#anim = @animate for i1 in 1:length(Optim.x_trace(res))
#
#    begin
#        p00 = heatmap(sample_data, aspect_ratio=1.0, clim=(0.0, 1.0), title="Sample data", legend = :none);
#        p01 = heatmap(fitting_data, aspect_ratio=1.0, clim=(0.0,1.0), title="Fitting data", legend = :none);
#        p02 = heatmap(f_general(Optim.x_trace(res)[i1]), aspect_ratio=1.0, clim=(0.0,1.0), title="estimated fit", legend = :none);
#        p03 = heatmap(fitting_data .- f_general(Optim.x_trace(res)[i1]), aspect_ratio=1.0, clim=(0.0, 0.3), title="discrepancy", legend = :none);
#    
#        plot(p00, p01, p02, p03, layout=@layout([A B C D]), 
#            framestyle=nothing, showaxis=false, 
#            xticks=false, yticks=false, 
#            size=(1200, 500),  
#            plot_title="iteration: $(Int(i1))/$(length(Optim.x_trace(res))),
#            estimation: $(Optim.x_trace(res)[i1])
#            true vals : $(true_vals)",
#            plot_titlevspan=0.25
#        )
#    end
#
#
#end;
#
#gif(anim, "DataToFunctions.jl/examples/anim_general_generalized.mp4", fps=2)
#
