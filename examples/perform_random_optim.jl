using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff,  LineSearches, Plots, Printf
using ProfileView, Profile

# to show all the numbers in 2 decimals format
Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f)

###
# creating the random matrix in size = (11, 12)
sample_data = rand(11, 12)

# creating the lower and upper bounds of the fitting variables (shift and scaling)
#   shift can not be higher than the size of the array,
#   scale can not be lower than zero or higher than size of the data, the latter causes the resulting array to be just one pixel
lower = [-1*size(sample_data)[1], -1*size(sample_data)[2], 0.0, 0.0]
upper = [size(sample_data)[1], size(sample_data)[2], size(sample_data)[1], size(sample_data)[2]]

# preparing the function to fit
f = get_function(sample_data; super_sampling=2);

# assigning a scale (multiplier) to the range of true values, wedo not want that the parameters of 
# function to be near the limits and cause strange behavior of the optimization
scale_range = 4.0

# true values of the fitting are random for the repeatibility
true_vals =  (rand(4) .* (upper .- lower)/scale_range ) .+ (lower/scale_range) #[3.0, 5.5, 1.85, 0.6]

# create the fitting data and adding random noise with scale of 1/10
noise = rand(11, 12)./10.0
fitting_data = f(true_vals) .+ noise

# create the loss function and using the Julia's multiple dispatch
# because the gradients function is required just one input (can be vector)
loss(p, z) = sum(abs2.(f(p, z) .- fitting_data))
loss(p2::Vector{Tuple{Float64, Float64}}) = loss(p2[1], p2[2])
loss(p3) = loss([p3[1], p3[2]], [p3[3], p3[4]])




# initialization of the LBFGS optimizer
inner_optimizer = LBFGS(; m=1, linesearch=LineSearches.BackTracking(order=2))



function perform_optim_mthr(loss, n_walkers)
    """
        defining a function to survey the parameter space to neglect the local 
        minima and find the global maxima

        loss: the loss function
        n_walkers: number of random initial parameter estimation
    """
    # allocating the estimation array: 
    # consists of four parameters [1:4] and the minimum loss function of them [5]
    est_m = zeros(n_walkers, 5)

    #x_tr = Array{Any}
    #res = Array{Optim.MultivariateOptimizationResults{}}(undef, n_walkers, 1)
    #res = Optim.MultivariateOptimizationResults{}
    
    # defining the random initial parameter values
    walkers = (rand(4, n_walkers) .* (upper .- lower)/scale_range ) .+ (lower/scale_range)
    
    # main loop to do the optimization for each of the initial parameter values
    # it uses the Threads to distribute the for loop to each thread
    # note that in the settings.json the Julia is started with 16 threads
    Threads.@threads for i in 1:size(walkers)[2]
        res = optimize(
            loss,
            lower, upper, # the limits (simple box constraints)
            walkers[:, i], 
            Fminbox(inner_optimizer),  # assigning the limits of fitting (simple constraints) along with the LBFGS
            Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
            autodiff = :forward
        );
        #x_tr = Optim.x_trace(res)

        # saving each 4 parameters of the fit and the minimum loss function
        est_m[i, 1:4] .= Optim.minimizer(res)#x_tr[end]
        est_m[i, 5] = minimum(res)#loss(x_tr[end])
    end
    # saving the parameters of the  minimum of the loss function 
    ans_m = est_m[argmin(est_m[:, 5]), :]
    return est_m, ans_m
end

#@profile est_m, ans_m = perform_optim_mthr(loss, 10000)


# first time:   4.749537 seconds (31.85 M allocations: 3.468 GiB, 8.50% gc time, 88.55% compilation time)
# 2nd time:     0.645307 seconds (12.68 M allocations: 2.525 GiB, 56.30% gc time)
# changing the fitting_data (noise values) : 0.448227 seconds (9.90 M allocations: 1.984 GiB, 27.55% compilation time: 38% of which was recompilation)
@time est_m, ans_m = perform_optim_mthr(loss, 1000); 


println(string(true_vals) * "\n" * string(ans_m))

# [2.23, -1.16, 1.26, 2.59]
# [2.26, -1.15, 1.27, 2.59, 0.41]


p00 = heatmap(sample_data, aspect_ratio=1.0, clim=(0.0, 1.0), title="Sample data", legend = :none);
p01 = heatmap(fitting_data, aspect_ratio=1.0, clim=(0.0,1.0), title="Fitting data", legend = :none);
p02 = heatmap(f(ans_m[1:4]), aspect_ratio=1.0, clim=(0.0,1.0), title="estimated fit", legend = :none);
p03 = heatmap(fitting_data .- f(ans_m[1:4]), aspect_ratio=1.0, clim=(0.0,1.0), title="discrepancy", legend = :none);

plot(p00, p01, p02, p03, layout=@layout([A B C D]), 
    framestyle=nothing, showaxis=false, 
    xticks=false, yticks=false, 
    size=(700, 300),  
    plot_title="True vals: $(true_vals)", plot_titlevspan=0.2
)

savefig("Output_mth_1.png")




function perform_optim_sthr(loss, n_walkers=1000)
    est = Array{Float64, 2}(undef, n_walkers, 5)

    walkers = (rand(4, n_walkers) .* (upper .- lower)/scale_range ) .+ (lower/scale_range)
    for i in 1:size(walkers)[2]
        res = optimize(
            loss, 
            lower, upper, 
            walkers[:, i], 
            Fminbox(inner_optimizer), 
            Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
            autodiff = :forward
        );
        x_tr = Optim.x_trace(res)
        est[i, 1:4] .= x_tr[end]
        est[i, 5] = loss(x_tr[end])
    end
    ans1 = est[argmin(est[:, 5]), :]
    return est, ans1
end


@time est_s, ans_s = perform_optim_sthr(loss, 20000); 

println(string(true_vals) * "\n" * string(ans_s))
# [1.07, -2.63, 0.78, 1.65]
# [1.02, -2.60, 0.77, 1.66, 0.46]

