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
using Noise
using IndexFunArrays



function poly_test(;sz=64, dtype=Float64, n_photons=1000)
    
    sz_psf = (sz, sz, 100)
    sampling = (0.040, 0.040, 0.050)
    # simulate a confocal PSF
    #aberrations = Aberrations([Zernike_HorizontalComa,Zernike_Tip],[0.8,0.7]);
    pp = PSFParams(0.5,1.4,1.52, method=MethodPropagateIterative);#, aberrations=aberrations);

    #pp_ex = PSFParams(pp_em; λ=0.488);#, method=MethodPropagateIterative, aplanatic=aplanatic_illumination, aberrations=aberrations);
    p_psf_3d = psf(sz_psf, pp, sampling=sampling);
    p_psf = p_psf_3d[:, :, 50]
    #sample_data = p_psf ./ maximum(p_psf)

    # normalizing the sample data
    sample_data = p_psf ./ maximum(p_psf)
    
    #sample_data = make_grid();
    f_1 = get_function_poly(Float64.(sample_data), 1); # get_function_affine(sample_data);
    true_vals = dtype[2.1, 1.05, 0.02, 1.5, 0.05, 1.02] # dtype[2.0, 1.0, 1.01, 1.0, 0.0, 0.0, 0.0]
    # true_vals = dtype[rand(-4.0:0.001:4.0), rand(-4.0:0.001:4.0), rand(0.5:0.001:1.5),rand(0.5:0.001:1.5), 0.0, 0.0, rand(0.001:0.001:pi/2.001)]
    dat2 = f_1(Tuple(true_vals))

    p_img = n_photons .* (dat2 ./ maximum(dat2))
    n_img = dtype.(poisson(Float64.(p_img)))

    s_data = Float64.(sample_data .* n_photons)
    #sample_data = dtype.(TestImages.shepp_logan(sz)) 
    f = get_function_poly(Float64.(s_data), 1);



    loss_m(p1::AbstractVector) = sum(abs2.(f(Tuple(p1)) .- n_img))
    st_vals = dtype[2.1, 1.01, 0.01, 1.5, 0.01, 0.9] #ones(Float64, 6)./10
    #st_vals = Float64[1.0, 0, 0, 0, 0, 0, 1.0, 0, 0,  1.0, 0, 0, 1.0, 0, 0, 0, 0, 0]
    # Float64[9.0, 0, 0, 0, 0, 0, 1, 0, 0,  5.0, 0, 0, 1, 0, 0, 0, 0, 0]
    # @vv f(Tuple(st_vals))
    # loss_m(st_vals)

    
    function g!(G, x)  # (G, x)
        G .= gradient(loss_m, x)[1]
    end
    od = OnceDifferentiable(loss_m, g!, st_vals)
    res = optimize(
        od, 
        st_vals,
        #Newton(),
        BFGS(; initial_stepnorm = 1e-2),#; linesearch=LineSearches.BackTracking(order=2)),
        #LBFGS(),#; linesearch=LineSearches.BackTracking(order=3)),
        #lower, upper, 
        #init_x,
        #Fminbox(inner_optimizer), 
        Optim.Options(store_trace = true, extended_trace = true, iterations=5000, g_tol=1e-3), 
        autodiff = :forward
    )

    # return the estimated parameters
    return true_vals, Optim.minimizer(res), res, f, f_1
end

#a, b, c = poly_test()
#@vt f(Tuple(b)) f_affine(a)


function make_grid!(arr::AbstractArray)
    arr[isinteger.(xx(size(arr))./10) .|| isinteger.(yy(size(arr))./10)] .= 1.0
    return arr
    
end
function make_grid(sz::NTuple{N, Int}=(64, 64)) where {N}
    arr = zeros(Float64, sz)
    make_grid!(arr)
    return arr
end
function make_grid(::Type{T}, sz::NTuple{N, Int}=(64, 64)) where {N, T}
    arr = zeros(T, sz)
    make_grid!(arr)
    return arr
end