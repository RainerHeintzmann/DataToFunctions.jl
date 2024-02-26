using DataToFunctions
using Optim, StaticArrays, LinearAlgebra, FourierTools
using Zygote
using ForwardDiff, LineSearches, Plots, Printf
using View5D
using Distributions
using Plots

using Rotations
using CoordinateTransformations

using Interpolations

function get_function2(data::AbstractArray; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear()))
    new_size = super_sampling.*size(data)
    upsampled = fftshift(resample(ifftshift(data), new_size))

    # return upsampled
    # itp = LinearInterpolation(axes(upsampled), upsampled, extrapolation_bc=extrapolation_bc);
    interpolation = Interpolations.interpolate(upsampled, interp_type)
    interpolation = extrapolate(interpolation, extrapolation_bc)
    # center of the original data (too keep the axis and number of datapointsi dentical to the original)
    center_orig = (size(data) .÷2 .+1)
    # create zero-centered original ranges (== axes)
    zero_axes = Tuple(ax .- c  for (ax, c) in zip(axes(data), center_orig))
    # center of the upsampled data. This is where to access the upsampled data
    cen_vec = SVector(size(data)./ 2.0)

    function transform_axes(t::SVector, mat::SMatrix, cen_vec::SVector, shift_vec::SVector)

        return (mat * (t - cen_vec)) + cen_vec - shift_vec

    end

    function zoomed(shift_vec, zoom, theta)
        zoom = zoom .* super_sampling
        # # careful: The center of the original data is not at the expected position! But rather at:
        # center_upsamp = new_size .÷2 .+1 # ((center_orig .-1) .*super_sampling .+1)  # new_size .÷2 .+1
        # scaled_axes = ((ax.-myc) .* z .+ cen for (ax, myc, cen, z) in zip(zero_axes, shift, center_upsamp, zoom))
    
        # #sh_x, sh_y = 0.0, 0.0
        s_x, s_y = zoom
        theta = theta[1]
    
        rot_mat =  @SMatrix [cos(theta)  -1.0*sin(theta); sin(theta)  cos(theta)];
        scale_mat = @SMatrix [s_x 0.0; 0.0 s_y];
        mat = rot_mat * scale_mat

        return interpolation.(transform_axes.(SVector.(Tuple.(CartesianIndices(data))), mat, cen_vec, SVector(shift_vec[1], shift_vec[2]))...)
        # return extrapolate(scale(interpolation, scaled_axes...), extrapolation_bc)
    end
    zoomed(p) = zoomed([p[1], p[2]], [p[3], p[4]], p[5]);
    # zoomed([p[1], p[2]], [p[3], p[4]], p[5]) = zoomed[p] 
    return zoomed

end

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
heatmap(z, aspect_ratio=1)




true_vals =   [1.1, -1.5, 0.75, 1.5, pi/2]


sample_data = z./maximum(z) 
f = get_function2(sample_data; super_sampling=2, extrapolation_bc=0.0);
f(true_vals)

heatmap(f(true_vals), aspect_ratio=1.0, clim=(0.0,1.0), legend = :none);
