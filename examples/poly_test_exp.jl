using Images
using DataToFunctions
using FindShift
using Optim, CUDA
using FourierTools
using View5D, Plots, Statistics

CUDA.allowscalar(false)

file1 = raw"D:\Hossein\Programming\Julia\DataToFunctions.jl\examples\test_polim1.jpeg"
file2 = raw"D:\Hossein\Programming\Julia\DataToFunctions.jl\examples\test_polim2.jpeg"

c1 = Float32.(Gray.(load(file1)))
c2 = Float32.(Gray.(load(file2)))

#TODO increase the size of the images
img11 = c1[30:1519+30, 30:779+30]
img12 = c1[30:1519+30, 800:779+800]
img21 = c2[30:1519+30, 30:779+30]
img22 = c2[30:1519+30, 800:779+800]

img11_c = CuArray(img11)
img12_c = CuArray(img12)
img21_c = CuArray(img21)
img22_c = CuArray(img22)

function resize_img(data, scale)
    new_size = size(data) .÷ scale
    imresize(data, new_size...)
end

resample_size = 2
img1_resampled = img11[1:resample_size:end, 1:resample_size:end] #resize_img(img1, 1)
img2_resampled = img12[1:resample_size:end, 1:resample_size:end] #resize_img(img2, 1)

#im11 = imfilter(img1_resampled, Kernel.gaussian(5));
#im12 = imfilter(img2_resampled, Kernel.gaussian(5));

f = get_interpolated_function(img12_c, PolynomialMode, 2);
loss_p(p1::AbstractArray) = (sum(abs2.(f(Tuple(p1)) .- img11_c)));
loss_updated(p1::AbstractArray) = mapreduce(abs2, +, f(Tuple(p1)) .- img11_c);
#st_vals = [0.0, 1.00, 0.00, 0.0, 0.00, 1.0] #ones(Float64, 6)./10
st_vals = Float32[0.0, 1.0, 0, 0, 0, 0,  0.0, 0, 0, 1.0, 0, 0];
# Float64[9.0, 0, 0, 0, 0, 0, 1, 0, 0,  5.0, 0, 0, 1, 0, 0, 0, 0, 0]
# @vv f(Tuple(st_vals))
# loss_m(st_vals) 

#a = f(Tuple(st_vals))
# @time a = f(Tuple(st_vals));
"""
function do_registeration_step(resample_size, gaussian_kernel_size, img1, img2, st_vals=[0f0, 1f0, 0f0, 0f0, 0f0, 1f0])
    img1_resampled = img1[1:resample_size:end, 1:resample_size:end] #resize_img(img1, 1)
    img2_resampled = img2[1:resample_size:end, 1:resample_size:end] #resize_img(img2, 1)

    im1 = imfilter(img1_resampled, Kernel.gaussian(gaussian_kernel_size));
    im2 = imfilter(img2_resampled, Kernel.gaussian(gaussian_kernel_size));

    im1_c = CuArray(im1)
    im2_c = CuArray(im2)

    f = get_interpolated_function(im2_c, PolynomialMode, 1);
    loss_updated(p1::AbstractArray) = mapreduce(abs2, +, f(Tuple(p1)) .- im1_c);

    #st_vals = Float32[0.0, 1.0, 0, 0, 0, 0,  0.0, 0, 0, 1.0, 0, 0];
    CUDA.@time res = optimize(
        loss_updated,
        st_vals,
        BFGS(),#; initial_stepnorm = 1f-1),#; linesearch=LineSearches.BackTracking(order=2)),
        autodiff = :forward
    )
    #return [res.minimizer[1]*resample_size, 1.0, 0, 0, 0, 0, res.minimizer[7]*resample_size, 0, 0, 1.0, 0, 0]
    return [res.minimizer[1]*resample_size, 1.0, 0, res.minimizer[4]*resample_size, 0, 1.0]
end

res_step1 = do_registeration_step(10, 5, img11, img12)
@vt img11 get_interpolated_function(img12, PolynomialMode, 1)(Tuple(res_step1)) img12
"""


"""
function g!(G, x)  # (G, x)
    G .= gradient(loss_updated, x)[1]
end
od = OnceDifferentiable(loss_updated, g!, st_vals)
"""
aligned_imgs = Array(copy(img11_c))


CUDA.@time res = optimize(
    loss_updated,
    st_vals,
    #Newton(),
    BFGS(; initial_stepnorm = 1f-2),#; linesearch=LineSearches.BackTracking(order=2)),
    #LBFGS(; linesearch=LineSearches.BackTracking(order=3)),
    #lower, upper, 
    #init_x,
    #Fminbox(inner_optimizer), 
    #Optim.Options(store_trace = true, extended_trace = true, iterations=5000), 
    autodiff = :forward
)
"""
10.271416 seconds (8.78 M CPU allocations: 589.752 MiB, 1.06% gc time) (2.25 k GPU allocations: 52.251 GiB, 0.30% memmgmt time)
* Status: success

* Candidate solution
Final objective value:     2.966678e+03

* Found with
Algorithm:     BFGS

* Convergence measures
|x - x'|               = 1.76e-05 ≰ 0.0e+00
|x - x'|/|x'|          = 3.62e-06 ≰ 0.0e+00
|f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
|f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
|g(x)|                 = 1.17e+05 ≰ 1.0e-01

* Work counters
Seconds run:   5  (vs limit Inf)
Iterations:    73
f(x) calls:    450
∇f(x) calls:   450
"""
aligned_imgs = cat(aligned_imgs, Array(f(Tuple(Optim.minimizer(res)))), dims=3);


for img_t in [img21_c, img22_c]
    f = get_interpolated_function(img_t, PolynomialMode, 2);
    loss_p(p1::AbstractArray) = (sum(abs2.(f(Tuple(p1)) .- img11_c)));
    loss_updated(p1::AbstractArray) = mapreduce(abs2, +, f(Tuple(p1)) .- img11_c);
    #st_vals = Float64[0.0, 1.00, 0.00, size(img11)[2], 0.00, -1.0] #ones(Float64, 6)./10
    st_vals = Float32[0.0, 1.0, 0, 0, 0, 0, size(img11)[2], 0, 0, -1.0, 0, 0];
    # Float64[9.0, 0, 0, 0, 0, 0, 1, 0, 0,  5.0, 0, 0, 1, 0, 0, 0, 0, 0]
    # @vv f(Tuple(st_vals))
    # loss_m(st_vals)

    """
    function g!(G, x)  # (G, x)
        G .= gradient(loss_p, x)[1]
    end
    od = OnceDifferentiable(loss_p, g!, st_vals)
    """
    CUDA.@time res = optimize(
        loss_updated,
        st_vals,
        #Newton(),
        BFGS(; initial_stepnorm = 1f-2),#; linesearch=LineSearches.BackTracking(order=2)),
        #LBFGS(),#; linesearch=LineSearches.BackTracking(order=3)),
        #lower, upper, 
        #init_x,
        #Fminbox(inner_optimizer), 
        #Optim.Options(store_trace = true, extended_trace = true, iterations=5000), 
        autodiff = :forward
    )


    aligned_imgs = cat(aligned_imgs, Array(f(Tuple(Optim.minimizer(res)))), dims=3);
end


aligned_imgs = clamp.(aligned_imgs, 0, 1)
@vv aligned_imgs

aligned_imgs = permutedims(aligned_imgs, [2, 1, 3])
save("aligned_all_cuda_polyorder2.gif", Gray.(aligned_imgs))



raw_imgs = cat(img11, img12, img21, img22, dims=3)
raw_imgs = clamp.(raw_imgs, 0, 1)
raw_imgs = permutedims(raw_imgs, [2, 1, 3])
save("raw_all_imgs.gif", Gray.(raw_imgs))
