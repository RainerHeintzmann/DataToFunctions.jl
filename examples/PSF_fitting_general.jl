using DataToFunctions
using Optim, StaticArrays, LinearAlgebra
using PointSpreadFunctions
using Zygote
using ForwardDiff, LineSearches, Plots, Printf
using View5D
using Distributions, Rotations
using Plots
using TestImages
using BenchmarkTools
#using InverseModeling
import Random
using Noise, Images, CSV, TiffImages
using ProgressBars

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
function perform_fit_general(loss_function, fitting_data::AbstractArray, init_x::AbstractArray{T}) where T
    # guess the shift parameters by taking the maximum values of the array and
    # centering the positions
    ##a, b = Tuple(argmax(fitting_data)) .- size(fitting_data) ./2.0 .- 1.0
    #print("INSIDE!!! hehe")
    # assigning the initial parameter estimates
    # init_x = vec([0.5, -1.5, 1.0, 1.0, 0.0, 0.0, pi/5]) #ndims(fitting_data)+1, ndims(fitting_data)+1))
    # reshape(Matrix(1.0*I, ndims(fitting_data)+1, ndims(fitting_data)+1), 1, 9)) #[a, b, 1.0, 1.0, 0.001, 0.001, 0.001]

    # setting the lower and upper boundary of the parameter values based on their limits
    lower = T[-1*size(fitting_data)[1], -1*size(fitting_data)[2], 0.0, 0.0, -0.01, -0.01, 0.0]
    upper = T[size(fitting_data)[1], size(fitting_data)[2], size(fitting_data)[1], size(fitting_data)[2], 0.01, 0.01, pi/2.0]

    # initializing the LBFGS optimizer
    inner_optimizer = BFGS()#; m=3, linesearch=LineSearches.BackTracking(order=3))
    
    # Computer, Optimize! :D
    res = optimize(
            loss_function, 
            
            #LBFGS(),
            lower, upper, 
            init_x,
            Fminbox(inner_optimizer), 
            Optim.Options(store_trace = true, extended_trace = true, iterations=5000), 
            autodiff = :forward
        )
    
    # return the estimated parameters
    return Optim.minimizer(res), res
end

function perform_fit(loss_function, init_x::AbstractArray{T}) where T
    
    # Computer, Optimize! :D
    res = optimize(
            loss_function, 
            init_x,
            #Newton(),
            #BFGS(),#; linesearch=LineSearches.BackTracking(order=3)),
            LBFGS(),#; linesearch=LineSearches.BackTracking(order=3)),
            #lower, upper, 
            #init_x,
            #Fminbox(inner_optimizer), 
            Optim.Options(store_trace = true, extended_trace = true, iterations=5000), 
            autodiff = :forward
        )
    
    # return the estimated parameters
    return Optim.minimizer(res), res
end

"""
    apply_transform(;matrix=true, sz=64, dtype=Float32, noise_level=1/20.0)

This function is designed for applying a transformation to a sample data using either matrix transformations or
parametric transformations based on the provided arguments.

# Arguments
`matrix`: if true, the fitting is done using a matrix transformation, otherwise, the fitting is done using a parametric transformation
`sz`: the size of the sample data
`dtype`: the data type of the sample data
`noise_level`: the noise level to add to the sample data

# Returns
a tuple of two arrays: the first array is the fitting data, and the second array is the estimated fitting data

# Example
apply_transform(matrix=true, sz=64, dtype=Float32, noise_level=1/20.0)

"""
function apply_transform(;matrix=false, sz=64, dtype=Float32, n_photons=1000, pure_rand=false, from_params=true, plotting=false)
    Random.seed!(14)

    # defining the mean and the varixance of the test normal (Gaussian) distribution
    μ = [0, 0]
    Σ = [sz/10  0.0;
        0.0 sz/10]

    # initializing the multivariate normal distribution
    p = MvNormal(μ, Σ)

    # to define the sample array based on a 2D normal distribution
    X = -1*sz/2.0:1*sz/2.0
    Y = -1*sz/2.0:1*sz/2.0 

    z = [pdf(p, [x,y]) for y in Y, x in X]


    # creating a PSF for the widefield microscope
    sz_psf = (sz, sz, 100)
    sampling = (0.040, 0.040, 0.050)
    # simulate a confocal PSF
    aberrations = Aberrations([Zernike_HorizontalComa],[0.8]);
    pp = PSFParams(0.5,1.4,1.52, method=MethodPropagateIterative, aberrations=aberrations);

    #pp_ex = PSFParams(pp_em; λ=0.488);#, method=MethodPropagateIterative, aplanatic=aplanatic_illumination, aberrations=aberrations);
    p_psf_3d = psf(sz_psf, pp, sampling=sampling);
    p_psf = p_psf_3d[:, :, 50]
    #sample_data = p_psf ./ maximum(p_psf)

    # normalizing the sample data
    sample_data = p_psf ./ maximum(p_psf)
    # sample_data = dtype.(z[1:sz, 1:sz]./maximum(z)) 
    # sample_data = dtype.(TestImages.shepp_logan(sz)) 
    # sample_data = rand(dtype, (sz, sz))

    # sample_data .+=  rand(dtype, (size(sample_data)...)).*noise_level;
    p_img = n_photons .* (sample_data);# ./ maximum(sample_data))
    n_img = dtype.(poisson(Float64.(p_img)))

    x_cen, y_cen = (size(n_img) ./ 2.0)
    t_to_origin = dtype[1.0 0.0 1*x_cen; 0.0 1.0 y_cen; 0.0 0.0 1.0];
    t_to_center = dtype[1.0 0.0 -1.0*x_cen; 0.0 1.0 -1.0*y_cen; 0.0 0.0 1.0];

    true_vals = dtype[rand(-4.0:0.001:4.0), rand(-4.0:0.001:4.0), 1.0, 1.0, 0.0, 0.0, 0.0];#rand(0.9:0.001:1.1),rand(0.9:0.001:1.1), 0.0, 0.0, 0.0];#rand(0.001:0.001:pi/2.001)]

    if !pure_rand

        shear_mat = dtype[1.0 true_vals[5] 0.0; true_vals[6] 1.0 0.0; 0.0 0.0 1.0];
        scale_mat = dtype[1.0/true_vals[3] 0.0 0.0; 0.0 1/true_vals[4] 0.0; 0.0 0.0 1.0];
    
        shift_mat = dtype[1.0 0.0 true_vals[1]; 0.0 1.0 true_vals[2]; 0.0 0.0 1.0];
        # converting the data to function (DataToFunctions.get_function)
        
        rot_mat =  dtype[cos(true_vals[7])  -1.0*sin(true_vals[7]) 0.0; sin(true_vals[7])  cos(true_vals[7]) 0.0; 0.0 0.0 1.0];
    
        matrix_c = (t_to_origin * scale_mat * shear_mat * rot_mat * shift_mat * t_to_center)
    else
        matrix_c = dtype.(t_to_origin * rand(0.1:0.001:1.0, (3, 3)) * t_to_center)
    end

    f_affine_sim_img = get_function_affine(sample_data);#; super_sampling=1);#, extrapolation_bc=0.0);
    if matrix
        t_img = f_affine_sim_img(SMatrix{3, 3}(matrix_c))#, fitting_data); #.+ dtype.(rand(size(sample_data)...))./5.0;
    else
        t_img = f_affine_sim_img(true_vals)#, fitting_data); #.+ dtype.(rand(size(sample_data)...))./5.0;
    end
    fitting_data  = dtype.(poisson(Float64.(t_img ./ maximum(t_img) .* n_photons))) #.+=  rand(dtype, size(p_img)...).*noise_level;

    heatmap(fitting_data, aspect_ratio=1, size=(600, 600), title="fitting data", titlefont = font(20), legend=:none, axis=([], false))
    annotate!(vec(map(x -> Tuple((reverse(Tuple(x))..., text(@sprintf("%.0f", fitting_data[x]), :center, font(5), :white))), CartesianIndices(sample_data))))
    savefig("figures/fitting/sample_data_1.png")
    
    # plot(heatmap(sample_data, aspect_ratio=1), heatmap(fitting_data, aspect_ratio=1))

    contour(sample_data, length=200, fill=false, title="Sample data", titlefont = font(20), legend=:none,  aspect_ratio=1, size=(600, 600))
    savefig("figures/fitting/sample_data_1_contour.png")
    return sample_data, fitting_data
end


function gauss_psf_comp()
    x = -10.0:0.01:10.0
    p = Normal(0.0, 1.0)
    y = pdf(p, x)
    y_psf(x) = (sin(x) /x)^2
    plot(x, y_psf.(x), label="PSF of a circular aperture", title="Comparison of a Gaussian and a PSF", titlefont=20, size=(800, 400))
    plot!(x, y./maximum(y), label="Gaussian with μ=0.0 & σ=1.0")
    savefig("figures/fitting/comp_psf_gaussian_1.png")
    
    plot(x, map(x -> (gradient(y_psf, x)[1]), x), label="Gradient of the PSF")
    plot!(x, map(x -> (gradient(x -> pdf(p, x), x)[1]), x), label="Gradient of the Gaussian", title="Comparison of the Gradients", titlefont=20, size=(800, 400))
    savefig("figures/fitting/comp_psf_gaussian_1_gradients.png")
end

"""
    main_fitting(;matrix=true, sz=64, dtype=Float32, iterations=20, noise_level=1/20.0, pure_rand=false, from_params=true)

This function is designed for performing fitting operations on sample data using either matrix transformations or 
parametric transformations based on the provided arguments.

# Arguments
`matrix`: if true, the fitting is done using a matrix transformation, otherwise, the fitting is done using a parametric transformation
`sz`: the size of the sample data
`dtype`: the data type of the sample data
`iterations`: the number of iterations to perform the fitting
`noise_level`: the noise level to add to the sample data
`pure_rand`: if true, the fitting is done using a random matrix transformation
`from_params`: if true, the fitting is done using the true values as the initial values

# Returns
a tuple of two arrays: the first array is the fitting data, and the second array is the estimated fitting data

# Example
x, y = main_fitting(matrix=true, sz=32, dtype=Float32, iterations=10, noise_level=1/20.0, pure_rand=false, from_params=true);


"""
function main_fitting(;matrix=true, sz=64, dtype=Float32, iterations=20, use_psf=true, n_photons=1000, pure_rand=false, from_params=true, plotting=false)
    Random.seed!(14)

    # defining the mean and the varixance of the test normal (Gaussian) distribution
    μ = [0, 0]
    Σ = [sz/30  0.0;
        0.0 sz/30]

    # initializing the multivariate normal distribution
    p = MvNormal(μ, Σ)

    # to define the sample array based on a 2D normal distribution
    X = -1*sz/2.0:1*sz/2.0
    Y = -1*sz/2.0:1*sz/2.0 

    z = [pdf(p, [x,y]) for y in Y, x in X]


    # creating a PSF for the widefield microscope
    sz_psf = (sz, sz, 100)
    sampling = (0.040, 0.040, 0.050)
    # simulate a confocal PSF
    aberrations = Aberrations([Zernike_HorizontalComa],[0.8]);
    pp = PSFParams(0.5,1.4,1.52, method=MethodPropagateIterative, aberrations=aberrations);

    #pp_ex = PSFParams(pp_em; λ=0.488);#, method=MethodPropagateIterative, aplanatic=aplanatic_illumination, aberrations=aberrations);
    p_psf_3d = psf(sz_psf, pp, sampling=sampling);
    p_psf = p_psf_3d[:, :, 50]
    #sample_data = p_psf ./ maximum(p_psf)

    # normalizing the sample data
    sample_data = p_psf ./ maximum(p_psf)
    sample_data_gaussian = dtype.(z[1:sz, 1:sz]./maximum(z)) 
    # sample_data = dtype.(TestImages.shepp_logan(sz)) 
    # sample_data = rand(dtype, (sz, sz))

    # sample_data .+=  rand(dtype, (size(sample_data)...)).*noise_level;
    p_img = n_photons .* (sample_data);# ./ maximum(sample_data))
    n_img = dtype.(poisson(Float64.(p_img)))

    y = similar(n_img, (size(n_img)..., iterations))
    x = similar(n_img, (size(n_img)..., iterations))
    pos_res = zeros(Float32, iterations, 2)
    pos_arr = zeros(Float32, iterations, 2)

    for i in ProgressBar(1:iterations)
        # println("iteration: ", i)

        x_cen, y_cen = (size(n_img) ./ 2.0)
        t_to_origin = dtype[1.0 0.0 1*x_cen; 0.0 1.0 y_cen; 0.0 0.0 1.0];
        t_to_center = dtype[1.0 0.0 -1.0*x_cen; 0.0 1.0 -1.0*y_cen; 0.0 0.0 1.0];

        true_vals = dtype[rand(-4.0:0.001:4.0), rand(-4.0:0.001:4.0), 1.0, 1.0, 0.0, 0.0, 0.0];#rand(0.9:0.001:1.1),rand(0.9:0.001:1.1), 0.0, 0.0, 0.0];#rand(0.001:0.001:pi/2.001)]

        if !pure_rand

            shear_mat = dtype[1.0 true_vals[5] 0.0; true_vals[6] 1.0 0.0; 0.0 0.0 1.0];
            scale_mat = dtype[1.0/true_vals[3] 0.0 0.0; 0.0 1/true_vals[4] 0.0; 0.0 0.0 1.0];
        
            shift_mat = dtype[1.0 0.0 true_vals[1]; 0.0 1.0 true_vals[2]; 0.0 0.0 1.0];
            # converting the data to function (DataToFunctions.get_function)
            
            rot_mat =  dtype[cos(true_vals[7])  -1.0*sin(true_vals[7]) 0.0; sin(true_vals[7])  cos(true_vals[7]) 0.0; 0.0 0.0 1.0];
        
            matrix_c = (t_to_origin * scale_mat * shear_mat * rot_mat * shift_mat * t_to_center)
        else
            matrix_c = dtype.(t_to_origin * rand(0.1:0.001:1.0, (3, 3)) * t_to_center)
        end

        f_affine_sim_img = get_function_affine(sample_data);#; super_sampling=1);#, extrapolation_bc=0.0);
        if matrix
            t_img = f_affine_sim_img(SMatrix{3, 3}(matrix_c))#, fitting_data); #.+ dtype.(rand(size(sample_data)...))./5.0;
        else
            t_img = f_affine_sim_img(true_vals)#, fitting_data); #.+ dtype.(rand(size(sample_data)...))./5.0;
        end
        fitting_data  = dtype.(poisson(Float64.(t_img ./ maximum(t_img) .* n_photons))) #.+=  rand(dtype, size(p_img)...).*noise_level;


        if use_psf
            f_affine = get_function_affine(p_img);#; super_sampling=1);#, extrapolation_bc=0.0);
        else
            f_affine = get_function_affine(n_photons .* sample_data_gaussian);#; super_sampling=1);#, extrapolation_bc=0.0);
        end
        #f_affine = get_function_affine(n_img);#; super_sampling=1);#, extrapolation_bc=0.0);
        # defining the loss function based on the gaussian noise
        loss_m(p1::AbstractMatrix) = sum(abs2.(f_affine(SMatrix{size(p1)...}(p1)) .- fitting_data))
        # loss_m(p1::AbstractVector) = sum(abs2.(f_affine(p1::AbstractVector) .- fitting_data))
        loss_m(p1::AbstractVector) = sum(abs2.(f_affine([p1[1], p1[2], 0.0, 0.0, 0.0, 0.0, 0.0]) .- fitting_data))


        if matrix
            if from_params
                st_vals = dtype[1.0 0.0 -1.0*(argmax(fitting_data)[1]-size(fitting_data)[1]/2.0); 0.0 1.0 -1.0*(argmax(fitting_data)[1]-size(fitting_data)[2]/2.0); 0.0 0.0 1.0]
            else
                st_vals = dtype[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
            end
        elseif from_params
            st_vals = dtype[argmax(fitting_data)[1]-size(fitting_data)[1]/2.0, argmax(fitting_data)[2]-size(fitting_data)[2]/2.0, 1.0, 1.0, 0.0, 0.0, 0.0];#pi/8.0]
        else
            st_vals = dtype[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];#pi/8.0]
        end
        
        # perform the main fit to the fitting data by minimizing the loss function
        stats = @timed output, res = perform_fit(loss_m, st_vals)
        
        
        if !matrix
            pos_arr[i, :] = true_vals[1:2]
            pos_res[i, :] = output[1:2]
        else
            pos_arr[i, :] = matrix_c[1:2, 3]
            pos_res[i, :] = output[1:2, 3]
        end

        y[:, :, i] = matrix ? f_affine(SMatrix{size(matrix_c)...}(output)) : f_affine(output)
        x[:, :, i] = fitting_data

        # plotting the output of the fitting pocedure for further illustration
    if plotting
        begin
            p00 = heatmap(n_img, aspect_ratio=1.0, title="Simulated sample PSF", colormap= :gist_gray);
            p01 = heatmap(fitting_data, aspect_ratio=1.0, title="Simulated PSF", colormap= :gist_gray);
            p02 = heatmap(matrix ? f_affine(SMatrix{size(matrix_c)...}(output)) : f_affine(output), aspect_ratio=1.0, title="Estimated fit", colormap= :gist_gray);
            p03 = heatmap(fitting_data .- (matrix ? f_affine(SMatrix{size(matrix_c)...}(output)) : f_affine(output)), aspect_ratio=1.0, title="Residuals", colormap= :bwr, clim=(-maximum((abs.(fitting_data .- (matrix ? f_affine(SMatrix{size(matrix_c)...}(output)) : f_affine(output))))), maximum((abs.(fitting_data .- (matrix ? f_affine(SMatrix{size(matrix_c)...}(output)) : f_affine(output)))))));

            plot(p00, p01, p03, p02, layout=@layout([A B; C D]), 
                #framestyle=nothing, 
                #showaxis=false, 
                #xticks=false, yticks=false, 
                size=(1200, 1200),  
                plot_title=" $(if matrix "Matrix" else "Parametric" end) fitting
True vals:  $(map(x -> @sprintf("%.3f",x), (matrix ? matrix_c : true_vals)))
fitted vals: $(map(x -> @sprintf("%.3f",x), output))
n. of photons: $(@sprintf("%.0f", n_photons))
time elapsed: $(@sprintf("%.1f", 1000.0*stats.time))ms, loss: $(@sprintf("%.2f", res.trace[1].value)) -> $(@sprintf("%.2f", res.trace[end].value))",
                plot_titlevspan=0.14
            )
            savefig("figures/fitting/$(matrix ? "Matrix" : "Parametric")_fitting_$(i).png")
        end
    end
    end
    return x, y, pos_arr, pos_res
end

x, y, pos_arr, pos_res = main_fitting(matrix=false, iterations=1000, sz=65, pure_rand=false, n_photons=10, from_params=true, plotting=false);  println(mean(pos_res[:, 1] .- pos_arr[:, 1])); 
println(std(pos_res[:, 1] .- pos_arr[:, 1])); 

#, title="Positional errors", markersize=2.0, xlabel="X error (pixels)", ylabel="Y error (pixels)", legend=:none, size=(600, 600), xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), alpha=0.4)

histogram2d(pos_res[:, 1] .- pos_arr[:, 1], pos_res[:, 2] .- pos_arr[:, 2], title="Positional errors histogram for 100 photons", xlabel="X error (pixels)", ylabel="Y error (pixels)", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), bins=20, aspect_ratio=1)
#


imgg = Gray{N0f16}.(x./maximum(x));
ff = TiffImages.DenseTaggedImage(imgg);
TiffImages.save("test_4_aberrated.tif", ff);

res_fiji = CSV.File(open(raw"C:\Users\ho82nat\Desktop\thunderstorm_res_100photons.csv"))
res_fiji_x = res_fiji["x [nm]"] ./ 80.0 .- 65.0 ./ 2.0;
res_fiji_y = res_fiji["y [nm]"] ./ 80.0 .- 65.0 ./ 2.0;


scatter(pos_res[:, 1] .- pos_arr[:, 1], pos_res[:, 2] .- pos_arr[:, 2], markershape= :circle, title="Positional errors", markersize=2.0, xlabel="X error (pixels)", ylabel="Y error (pixels)", legend=:none, size=(600, 600), label="DataToFunctions fitting")#, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), alpha=0.4)
#scatter!(res_fiji_y .- pos_arr[:, 1], res_fiji_x .- pos_arr[:, 2], markershape= :rect, markersize=2.0, alpha=0.2, label="ThunderSTORM fitting")

println((std(pos_res[:, 1] .- pos_arr[:, 1]), std(pos_res[:, 2] .- pos_arr[:, 2])), (mean(pos_res[:, 1] .- pos_arr[:, 1]), mean(pos_res[:, 2] .- pos_arr[:, 2])));
#println((std(res_fiji_y .- pos_arr[:, 1]), std(res_fiji_x .- pos_arr[:, 2])), (mean(res_fiji_y .- pos_arr[:, 1]), mean(res_fiji_x .- pos_arr[:, 2])));

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
