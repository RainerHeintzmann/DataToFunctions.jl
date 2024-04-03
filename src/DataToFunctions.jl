module DataToFunctions
using Interpolations
using FourierTools
using StaticArrays

export get_function, get_function_affine, add_dim, red_dim_apply, red_dim, mat_mul, func_transform
export extrapolate, interpolate

"""
    get_function(data::AbstractArray; super_sampling=2, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `dat(shift, zoom)` which generates a shifted and scaled version of the original data. 
This is useful for fitting with a function which is itself defined by measured data.

# Arguments
`data`: The data to represent by the function `dat`
`super_sampling`: The factor by which the data is internally represented as a supersampled version (Fourier-based upsampling, see `FourierTools.resample`)
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

# Example
```jldoctest
```
"""
function get_function(data::AbstractArray; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear()))
    new_size = super_sampling.*size(data)
    upsampled = fftshift(resample(ifftshift(data), new_size))
    # @show upsampled
    # return upsampled
    # itp = LinearInterpolation(axes(upsampled), upsampled, extrapolation_bc=extrapolation_bc);
    interpolation = Interpolations.interpolate(upsampled, interp_type)
    interpolation = extrapolate(interpolation, extrapolation_bc)
    # center of the original data (too keep the axis and number of datapointsi dentical to the original)
    center_orig = (size(data) .÷2 .+1)
    # create zero-centered original ranges (== axes)
    zero_axes = Tuple(ax .- c  for (ax, c) in zip(axes(data), center_orig))
    # center of the upsampled data. This is where to access the upsampled data
    function zoomed(shift, zoom)
        zoom = zoom .* super_sampling
        # careful: The center of the original data is not at the expected position! But rather at:
        center_upsamp = new_size .÷2 .+1 # ((center_orig .-1) .*super_sampling .+1)  # new_size .÷2 .+1
        scaled_axes = ((ax.-myc) .* z .+ cen for (ax, myc, cen, z) in zip(zero_axes, shift, center_upsamp, zoom))
        # @show Tuple(scaled_axes)
        return interpolation[scaled_axes...]
        # return extrapolate(scale(interpolation, scaled_axes...), extrapolation_bc)
    end

    return zoomed

    zoomed(p) = zoomed([p[1], p[2]], [p[3], p[4]])
    # return (pos) -> interp_linear((center .+ pos)...)
    # fitp(t) = interp_linear(t...)
    # @time res1 = fitp.(tcoords);  # 1 sec
    # function my_zoom

end

function add_dim(cind)
    return SVector.((Tuple(cind))..., 1)
end

@inline function red_dim(svec::SVector{S,T})::SVector{S-1,T} where {S,T}
    return @view svec[1:S-1]
end

@inline function red_dim_apply(fct::AbstractArray{R}, svec::SVector{S,T})::R where {S,T, R}
    return fct((@view svec[1:S-1])...)
end

@inline function idx_apply(fct::AbstractArray{R}, svec::SVector{S,T})::R where {S,T, R}
    return fct(svec...)
end

# multiplying the transformation matrix
@inline function mat_mul(t::SVector{N, Int64}, matrix_c::SMatrix{N,N,T})::SVector{N,T} where {N,T}
    return matrix_c * t
end


# Applying the coordinate transformation function
@inline function func_transform(t, coord_transform_func::Function)::SVector
    return coord_transform_func(Tuple(t))
end


# How to run this function:
# f_general((t) -> (t[1]*1.01, t[2]*1.01), out2)
"""
    apply_transform_tuple!(coord_transf_func::Function, data, itp, out)

applies a tuple-based coordinate transformation function to the indices of an array and returns the transformed array

coord_transf_func: A function that takes a tuple of coordinates and returns a new tuple of coordinates
data: The data to transform
itp: The interpolation object to use

# Example
```jldoctest
```
"""
function apply_transform_tuple!(coord_transf_func::Function, data, itp, out)
    out .= red_dim_apply.(Ref(itp), func_transform.(CartesianIndices(data), Ref(coord_transf_func)));
end

"""
    apply_transform_svec!(coord_transf_func::Function, data, itp, out)
applies a homogeneous coordinate-based coordinate transformation function to the indices of an array and returns the transformed array

coord_transf_func: A function that takes a N-+1 dimensional homogeneous SVector returns a new N+1 dimensional SVector
data: The data to transform
itp: The interpolation object to use

"""
function apply_transform_svec!(coord_transf_func::Function, data, itp, out)
    out .= idx_apply.(Ref(itp), coord_transf_func.(CartesianIndices(data)));
end

"""
    apply_transform_homogen!(coord_transf_func::Function, data, itp, out)
applies a homogeneous coordinate-based coordinate transformation function to the indices of an array and returns the transformed array

coord_transf_func: A function that takes a N-+1 dimensional homogeneous SVector returns a new N+1 dimensional SVector
data: The data to transform
itp: The interpolation object to use

"""
function apply_transform_homogen!(coord_transf_func::Function, data, itp, out)
    h_coord_transf_func = (c) -> red_dim(coord_transf_func(add_dim(c)))
    apply_transform_svec!(h_coord_transf_func, data, itp, out);
    # out .= itp.(red_dim.(coord_transf_func.(add_dim.(CartesianIndices(data)))));
    # out .= red_dim_apply.(Ref(itp), coord_transf_func.(add_dim.(CartesianIndices(data))));
end

function apply_transform_affine!(mymat, data, itp, out)
    # red_dim_apply.(Ref(itp), func_transform.(CartesianIndices(data), Ref(coord_transf_func)));
    # return red_dim_apply.(Ref(itp), mat_mul.(add_dim.(CartesianIndices(data)), Ref(mymat)));
    
    homogenous_transform = (c) -> mat_mul(c, mymat)
    apply_transform_homogen!(homogenous_transform, data, itp, out);
    #return out
end

function get_function_homogen(data::AbstractArray{T}, fct_hom::Function; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where T
end

"""
    get_function_affine(data::AbstractArray; super_sampling=1, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(p, [out])` which generates a transformed version of the original data parameterized by transform parameters.
This is useful for fitting with a function which is itself defined by measured data.
The returned function supports two ways to be used, with an affine transform matrix `p` as in input or with a vector `p` of parameters. 
The optional argument `out` can be used to store the result of the transformation.


# Arguments
`data`: The data to represent by the function `dat`
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

"""
function get_function_affine(data::AbstractArray{T}; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where T
    #new_size = super_sampling.*size(data)
    #upsampled = fftshift(resample(ifftshift(data), new_size))

    # building the extraplation + interpolation object
    itp = extrapolate(interpolate(data, interp_type), extrapolation_bc);


    @inline function interpolated(matrix_c::SMatrix, out = similar(data))
        
        # # init a new array for the output
        # out = similar(data)
        # out = data

        # out[CartesianIndices(data)] .= red_dim_apply.(Ref(itp), mat_mul.(add_dim.(CartesianIndices(data)), Ref(matrix_c)));
        return apply_transform_affine!(matrix_c, data, itp, out);
        # return red_dim_apply.(Ref(itp), f.(add_dim.(CartesianIndices(data)), Ref(matrix_c)));
        
        # return out
    end

    # function interpolated!(matrix_c::SMatrix, out::AbstractArray)

    #     out[CartesianIndices(data)] .= red_dim_apply.(Ref(itp), f.(add_dim.(CartesianIndices(data)), Ref(matrix_c)));
        
    #     #return out
    # end

    @inline function interpolated(p::AbstractVector{T}, out = similar(data)) where T     

        # init a new array for the output
        # out = data

        x_cen, y_cen = (size(data) .÷ 2.0 .+1)
        # x_cen_up, y_cen_up = (size(upsampled) .÷ 2.0 .+ 1.0)

        # creating the matrices of rotation, shear, scale, and shift
        rot_mat =  @SMatrix [cos(p[7])  -1.0*sin(p[7]) 0.0; sin(p[7])  cos(p[7]) 0.0; 0.0 0.0 1.0];
        shear_mat = @SMatrix [1.0 p[5] 0.0; p[6] 1.0 0.0; 0.0 0.0 1.0];
        scale_mat = @SMatrix [1/p[3] 0.0 0.0; 0.0 1/p[4] 0.0; 0.0 0.0 1.0];
        shift_mat = @SMatrix [1.0 0.0 -1*p[1]; 0.0 1.0 -1*p[2]; 0.0 0.0 1.0];
        t_to_origin = @SMatrix [1.0 0.0 1*x_cen; 0.0 1.0 y_cen; 0.0 0.0 1.0];
        t_to_center = @SMatrix [1.0 0.0 -1.0*x_cen; 0.0 1.0 -1.0*y_cen; 0.0 0.0 1.0];
        # t_orig_upsampled = @SMatrix [1.0 0.0 -1.0*x_cen_up; 0.0 1.0 -1.0*y_cen_up; 0.0 0.0 1.0]

        # building the overall transformation matrix
        # matrix_c = t_to_origin * scale_mat * shear_mat * rot_mat *shift_mat * t_to_center
        matrix_c = t_to_origin * scale_mat * rot_mat *shift_mat * t_to_center

        # looping all over the catesian indedices of the input image, 
        # first ading a new value to its third dimenstion: 1.0,
        # converting to the new indices using the transformation matrix and then,
        # using the "itp" object, we build the transfomed image "out"
        
        #for I1 in CartesianIndices(data)
        #    out[I1] = itp(f(SVector(Tuple(I1)..., 1), matrix_c)[1:2]...)
        #end

        # out[CartesianIndices(data)] .= red_dim_apply.(Ref(itp), f.(add_dim.(CartesianIndices(data)), Ref(matrix_c)));
        return interpolated(matrix_c, out);
        # return red_dim_apply.(Ref(itp), f.(add_dim.(CartesianIndices(data)), Ref(matrix_c)));

        # return out
    end

    
    # function interpolated(p::AbstractVector{T}, out::AbstractArray) where T     

    #     x_cen, y_cen = (size(data) .÷ 2.0 .+1)
    #     # x_cen_up, y_cen_up = (size(upsampled) .÷ 2.0 .+ 1.0)

    #     # creating the matrices of rotation, shear, scale, and shift
    #     rot_mat =  @SMatrix [cos(p[7])  -1.0*sin(p[7]) 0.0; sin(p[7])  cos(p[7]) 0.0; 0.0 0.0 1.0];
    #     shear_mat = @SMatrix [1.0 p[5] 0.0; p[6] 1.0 0.0; 0.0 0.0 1.0];
    #     scale_mat = @SMatrix [1/p[3] 0.0 0.0; 0.0 1/p[4] 0.0; 0.0 0.0 1.0];
    #     shift_mat = @SMatrix [1.0 0.0 -1*p[1]; 0.0 1.0 -1*p[2]; 0.0 0.0 1.0];
    #     t_to_origin = @SMatrix [1.0 0.0 1*x_cen; 0.0 1.0 y_cen; 0.0 0.0 1.0];
    #     t_to_center = @SMatrix [1.0 0.0 -1.0*x_cen; 0.0 1.0 -1.0*y_cen; 0.0 0.0 1.0];
    #     # t_orig_upsampled = @SMatrix [1.0 0.0 -1.0*x_cen_up; 0.0 1.0 -1.0*y_cen_up; 0.0 0.0 1.0]

    #     # building the overall transformation matrix
    #     # matrix_c = t_to_origin * scale_mat * shear_mat * rot_mat *shift_mat * t_to_center
    #     matrix_c = t_to_origin * scale_mat * rot_mat *shift_mat * t_to_center

    #     out[CartesianIndices(data)] .= red_dim_apply.(Ref(itp), f.(add_dim.(CartesianIndices(data)), Ref(matrix_c)));
    # end

    return interpolated
end



end # module DataToFunctions