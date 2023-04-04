using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using Zygote
using ForwardDiff,  LineSearches, Plots, Printf
using ProfileView, Profile


Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f)

###

sample_data = rand(11,12)
lower = [-1*size(sample_data)[1], -1*size(sample_data)[2], 0.0, 0.0]
upper = [size(sample_data)[1], size(sample_data)[2], size(sample_data)[1], size(sample_data)[2]]

f = get_function(sample_data; super_sampling=2);
scale_range = 4.0
true_vals =  (rand(4) .* (upper .- lower)/scale_range ) .+ (lower/scale_range) #[3.0, 5.5, 1.85, 0.6]
fitting_data = f(true_vals) .+ rand(11, 12)./10.0

loss(p, z) = sum(abs2.(f(p, z) .- fitting_data))
loss(p2::Vector{Tuple{Float64, Float64}}) = loss(p2[1], p2[2])
loss(p3) = loss([p3[1], p3[2]], [p3[3], p3[4]])




#initial_x = [2.0, 2.0]
# requires using LineSearches
inner_optimizer = LBFGS(; m=1, linesearch=LineSearches.BackTracking(order=2))



function perform_optim_mthr(loss, n_walkers)
    est = zeros(n_walkers, 5)
    #x_tr = Array{Any}
    #res = Array{Optim.MultivariateOptimizationResults{}}(undef, n_walkers, 1)
    #res = Optim.MultivariateOptimizationResults{}

    walkers = (rand(4, n_walkers) .* (upper .- lower)/scale_range ) .+ (lower/scale_range)
    Threads.@threads for i in 1:size(walkers)[2]
        res = optimize(
            loss,
            lower, upper, 
            walkers[:, i], 
            Fminbox(inner_optimizer), 
            #Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
            autodiff = :forward
        );
        #x_tr = Optim.x_trace(res)

        est[i, 1:4] .= Optim.minimizer(res)#x_tr[end]
        est[i, 5] = minimum(res)#loss(x_tr[end])
    end
    ans1 = est[argmin(est[:, 5]), :]
    return est, ans1
end

#@profile est_m, ans_m = perform_optim_mthr(loss, 10000)


@time est_m, ans_m = perform_optim_mthr(loss, 20000); #  5.191172 seconds (126.96 M allocations: 26.015 GiB, 54.49% gc time, 0.90% compilation time)

println(string(true_vals) * "\n" * string(ans_m))
# [1.07, -2.63, 0.78, 1.65]
# [1.02, -2.60, 0.77, 1.66, 0.46]


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


@time est_s, ans_s = perform_optim_sthr(loss, 20000); # 17.400975 seconds (105.36 M allocations: 21.475 GiB, 7.54% gc time)

println(string(true_vals) * "\n" * string(ans_s))
# [1.07, -2.63, 0.78, 1.65]
# [1.02, -2.60, 0.77, 1.66, 0.46]

