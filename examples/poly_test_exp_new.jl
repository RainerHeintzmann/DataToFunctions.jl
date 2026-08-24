using Images
using DataToFunctions
using FindShift
using Optim, CUDA
using FourierTools, Zygote
using View5D, Plots, Statistics, LineSearches

CUDA.allowscalar(false)

file1 = raw"D:\Hossein\Programming\Julia\DataToFunctions.jl\examples\markerpen_C1_00001.tif"
file2 = raw"D:\Hossein\Programming\Julia\DataToFunctions.jl\examples\markerpen_C2_00001.tif"

c1 = Float32.(Gray.(load(file1)))
c2 = Float32.(Gray.(load(file2)))

wide=true
if wide
    img11 = c1[180:1990+180, 160:2050+160]
    img12 = c1[160:1990+160, 2180:2050+2180]
    img21 = c2[130:1990+130, 140:2050+140]
    img22 = c2[130:1990+130, 2160:2050+2160]
end

@vt img11 img12 img21 img22



img11_c = CuArray(img11./maximum(img11))
img12_c = CuArray(img12./maximum(img12))
img21_c = CuArray(img21./maximum(img21))
img22_c = CuArray(img22./maximum(img22))


resample_size = 2
img1_resampled = img11[1:resample_size:end, 1:resample_size:end] #resize_img(img1, 1)
img2_resampled = img12[1:resample_size:end, 1:resample_size:end] #resize_img(img2, 1)

#im11 = imfilter(img1_resampled, Kernel.gaussian(5));
#im12 = imfilter(img2_resampled, Kernel.gaussian(5));


"""
resample_size=50
img1_resampled = img11[1:resample_size:end, 1:resample_size:end] #resize_img(img1, 1)
img2_resampled = img12[1:resample_size:end, 1:resample_size:end] #resize_img(img2, 1)

im1 = imfilter(img1_resampled, Kernel.gaussian(3));
im2 = imfilter(img2_resampled, Kernel.gaussian(3));

im1_c = CuArray(im1./maximum(im1))
im2_c = CuArray(im2./maximum(im2)) 

f1 = get_interpolated_function(im2_c, PolynomialMode, 2);
loss_updated1(p1::AbstractArray) = mapreduce(abs2, +, f(Tuple(p1)) .- im1_c);

st_vals = Float32[0.0, 1.0, 0, 0, 0, 0,  0.0, 0, 0, 1.0, 0, 0];
CUDA.@time res1 = optimize(
    loss_updated1,
    st_vals,
    BFGS(; initial_stepnorm = 1f-2),#; linesearch=LineSearches.BackTracking(order=2)),
    autodiff = :forward
)

@vt Array(im1_c) Array(f1(Tuple(res1.minimizer))) Array(im2_c)

function do_registeration_step(resample_size, gaussian_kernel_size, img1, img2, st_vals=Float32[0.0, 1.0, 0, 0, 0, 0,  0.0, 0, 0, 1.0, 0, 0])
    img1_resampled = img1[1:resample_size:end, 1:resample_size:end] #resize_img(img1, 1)
    img2_resampled = img2[1:resample_size:end, 1:resample_size:end] #resize_img(img2, 1)

    im1 = imfilter(img1_resampled, Kernel.gaussian(gaussian_kernel_size));
    im2 = imfilter(img2_resampled, Kernel.gaussian(gaussian_kernel_size));

    im1_c = CuArray(im1)
    im2_c = CuArray(im2)

    f = get_interpolated_function(im2_c, PolynomialMode, 2);
    loss_updated(p1::AbstractArray) = mapreduce(abs2, +, f(Tuple(p1)) .- im1_c);

    #st_vals = Float32[0.0, 1.0, 0, 0, 0, 0,  0.0, 0, 0, 1.0, 0, 0];
    CUDA.@time res = optimize(
        loss_updated,
        st_vals,
        BFGS(),#; initial_stepnorm = 1f-1),#; linesearch=LineSearches.BackTracking(order=2)),
        autodiff = :forward
    )
    return [res.minimizer[1]*resample_size, 1.0, 0, 0, 0, 0, res.minimizer[7]*resample_size, 0, 0, 1.0, 0, 0]
    #return [res.minimizer[1]*resample_size, 1.0, 0, res.minimizer[4]*resample_size, 0, 1.0]
end

res_step1 = do_registeration_step(10, 5, img11, img12)
@vt img11 get_interpolated_function(img12, PolynomialMode, 2)(Tuple(res_step1)) img12
"""


order=2
f = get_interpolated_function(img12_c, PolynomialMode, order);
loss_p(p1::AbstractArray) = (sum(abs2.(f(Tuple(p1)) .- img11_c)));
loss_updated(p1::AbstractArray) = mapreduce(abs2, +, f(Tuple(p1)) .- img11_c);
if order ==1
    st_vals = Float32[0.0, 1.00, 0.00, 0.0, 0.00, 1.0] #ones(Float64, 6)./10
elseif order == 2
    st_vals = Float32[0.0, 1.0, 0, 0, 0, 0,  0.0, 0, 0, 1.0, 0, 0];
end


aligned_imgs = Array(copy(img11_c))


CUDA.@time res = optimize(
    loss_updated,
    st_vals,
    #Newton(),
    BFGS(; initial_stepnorm = 1f-1),#; linesearch=LineSearches.BackTracking(order=2)),
    #LBFGS(; linesearch=LineSearches.BackTracking(order=3)),
    #lower, upper, 
    #init_x,
    #Fminbox(inner_optimizer), 
    Optim.Options(store_trace = true, extended_trace = true, iterations=5000, g_tol=1f-2), 
    autodiff = :forward
)

open("markerpen_new_data.txt", "w") do f
    write(f, "\norder $(order) params = $(res.minimizer)")
end

aligned_imgs = cat(aligned_imgs, Array(f(Tuple(res.minimizer))), dims=3);
@vv aligned_imgs


for img_t in [img21_c, img22_c]
    f = get_interpolated_function(img_t, PolynomialMode, 2);
    loss_p(p1::AbstractArray) = (sum(abs2.(f(Tuple(p1)) .- img11_c)));
    loss_updated(p1::AbstractArray) = mapreduce(abs2, +, f(Tuple(p1)) .- img11_c);
    #st_vals = Float64[0.0, 1.00, 0.00, size(img11)[2], 0.00, -1.0] #ones(Float64, 6)./10
    #st_vals = Float32[0.0, 1.0, 0, 0, 0, 0, size(img11)[1], 0, 0, -1.0, 0, 0];
    if order ==1
        st_vals = Float32[0.0, 1.00, 0.00, 0.0, 0.00, 1.0] #ones(Float64, 6)./10
    elseif order == 2
        st_vals = Float32[0.0, 1.0, 0, 0, 0, 0,  0.0, 0, 0, 1.0, 0, 0];
    end

    CUDA.@time res = optimize(
        loss_updated,
        st_vals,
        #Newton(),
        BFGS(; initial_stepnorm = 1f-1),#; linesearch=LineSearches.BackTracking(order=2)),
        #LBFGS(),#; linesearch=LineSearches.BackTracking(order=3)),
        #lower, upper, 
        #init_x,
        #Fminbox(inner_optimizer), 
        Optim.Options(store_trace = true, extended_trace = true, iterations=5000, g_tol=1f-2), 
        autodiff = :forward
    )
    println(res)
    open("markerpen_new_data.txt", "a") do f
        write(f, "\norder $(order) params = $(res.minimizer)")
    end
    aligned_imgs = cat(aligned_imgs, Array(f(Tuple(Optim.minimizer(res)))), dims=3);
end


aligned_imgs = clamp.(aligned_imgs, 0, 1)
@vv aligned_imgs

aligned_imgs = permutedims(aligned_imgs, [2, 1, 3])
save("aligned_all_cuda_polyorder2_markerpen_new.gif", Gray.(aligned_imgs))



raw_imgs = cat(img11, img12, img21, img22, dims=3)
raw_imgs = clamp.(raw_imgs, 0, 1)
raw_imgs = permutedims(raw_imgs, [2, 1, 3])
save("raw_all_imgs_markerpen_new.gif", Gray.(raw_imgs))
