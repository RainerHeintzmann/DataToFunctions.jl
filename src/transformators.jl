using Interpolations
using FourierTools
using StaticArrays
using EvalMultiPoly

export get_interpolated_function, get_function_tuple, get_function_svec, get_function_affine, get_function_poly
export add_dim, red_dim_apply, red_dim, mat_mul, func_transform
export extrapolate, interpolate

export PolynomialMode, AffineMode

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


"""
function get_function_old(data::AbstractArray; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear()))
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

"""
    add_dim(cind)

adds a dimension to a CartesianIndex

`cind`: A CartesianIndex
"""
function add_dim(cind)
    return SVector.((Tuple(cind))..., 1)
end

"""
    red_dim(svec::SVector{S,T})

removes the last dimension of a SVector to convert it from a homogeneous to a Cartesian coordinates

`svec::SVector{S,T}`: A SVector
"""
@inline function red_dim(svec::SVector{S,T})::SVector{S-1,T} where {S,T}
    return @view svec[1:S-1]
end

"""
    red_dim_apply(fct, svec::SVector{S,T})

applies a function to a SVector by removing the last dimension to convert it from a homogeneous to a Cartesian coordinates

`fct`: The function to apply
`svec::SVector{S,T}`: A SVector
"""
@inline function red_dim_apply(fct, svec::SVector{S,T}) where {S,T}
    return fct((@view svec[1:S-1])...)
end

"""
    red_dim_apply(fct, tup::NTuple{S,T})

applies a function to a Tuple by removing the last dimension to convert it from a homogeneous to a Cartesian coordinates

`fct`: The function to apply
`tup::NTuple{S,T}`: A Tuple
"""
@inline function red_dim_apply(fct, tup::NTuple{S,T}) where {S,T}
    return fct(tup[1:S-1]...)
end

"""
    idx_apply(fct, svec::SVector{S,T}) where {S,T}

applies a function to a SVector

`fct`: The function to apply
`svec::SVector{S,T}`: A SVector
"""
@inline function idx_apply(fct, svec::SVector{S,T})::Number where {S,T}
    return fct(svec...)
end

"""
    idx_apply(fct, tup::NTuple{S,T}) where {S,T}

applies a function to a Tuple

`fct`: The function to apply
`tup::NTuple{S,T}`: A Tuple
"""
@inline function idx_apply(fct, tup::NTuple{S,T})::Number where {S,T}
    return fct(tup...)
end

"""
    mat_mul(t::SVector{N, T2}, matrix_c::SMatrix{N,N,T})

multiplies a SVector with a SMatrix

`t::SVector{N, T2}`: The SVector to multiply
`matrix_c::SMatrix{N,N,T}`: The SMatrix to multiply with
"""
@inline function mat_mul(t::SVector{N, T2}, matrix_c::SMatrix{N,N,T})::SVector{N,T} where {N,T, T2}
    return matrix_c * t
end

"""
    func_transform(t, coord_transform_func::Function)::SVector

applies a coordinate transformation function to an array or `CartesianIndex` and returns the transformed array

`t`: The array or `CartesianIndex` to transform
`coord_transform_func::Function`: The function to apply the transformation
"""
@inline function func_transform(t, coord_transform_func::Function)::SVector
    return coord_transform_func(Tuple(t))
end

"""
    func_transform_tup(t, coord_transform_func::Function)

applies a coordinate transformation function to a Tuple

`t`: The array or `CartesianIndex` to transform
`coord_transform_func::Function`: The function to apply the transformation
"""
@inline function func_transform_tup(t, coord_transform_func::Function)
    return coord_transform_func(Tuple(t))
end


"""
    apply_transform(coord_transf_func::Function, data::AbstractArray{T}, itp) where {T}

applies a general coordinate transformation function to the indices of an array and returns the transformed array

`coord_transf_func:Function`: A function that takes a N-+1 dimensional CartesianIndex and returns a new N+1 dimensional SVector or Tuple
`data::AbstractArray{T}`: The data to transform
`itp`: The interpolation object to use
"""
function apply_transform(coord_transf_func::Function, data::AbstractArray{T, N}, itp) where {T, N} #, out::AbstractArray{T}) where {T}
    # @info "Applying tuple transformation"
    
    # return map((it) -> idx_apply(Interpolations.adapt(gpu_or_cpu(nothing), itp), coord_transf_func(Tuple(it))), CartesianIndices(data))
    return idx_apply.(Ref(Interpolations.adapt(gpu_or_cpu(nothing), itp)), coord_transf_func.(Tuple.(CartesianIndices(data))));
    #return idx_apply.(Ref(itp), coord_transf_func.(Tuple.(CartesianIndices(data))));
    # return idx_apply.(Ref(itp), coord_transf_func.(CartesianIndices(data)));
end

"""
    apply_transform_homogen(coord_transf_func::Function, data, itp)
applies a homogeneous coordinate-based coordinate transformation function to the indices of an array and returns the transformed array

`coord_transf_func::Function`: A function that takes a N-+1 dimensional homogeneous SVector returns a new N+1 dimensional SVector
`data`: The data to transform
`itp`: The interpolation object to use
"""
function apply_transform_homogen(coord_transf_func::Function, data, itp)#, out)
    h_coord_transf_func = (c) -> red_dim(coord_transf_func(add_dim(c)))
    #@info "Applying homogeneous transformation"
    return apply_transform(h_coord_transf_func, data, itp)#, out);
    # out .= itp.(red_dim.(coord_transf_func.(add_dim.(CartesianIndices(data)))));
    # out .= red_dim_apply.(Ref(itp), coord_transf_func.(add_dim.(CartesianIndices(data))));
end

"""
    apply_transform_affine(mymat::SMatrix{T}, data, itp) where T

applies an affine transformation matrix to the indices of an array and returns the transformed array

`mymat::SMatrix{T}` The affine transformation matrix to apply
`data`: The data to transform
`itp`: The interpolation function (object) to use
"""
function apply_transform_affine(mymat::SMatrix{T}, data, itp) where T #, out) where {T} # The SMatrix spec is important to avoid allocations
    # red_dim_apply.(Ref(itp), func_transform.(CartesianIndices(data), Ref(coord_transf_func)));
    # return red_dim_apply.(Ref(itp), mat_mul.(add_dim.(CartesianIndices(data)), Ref(mymat)));
    # @info "Applying affine transformation"
    homogenous_transform = (c) -> mat_mul(c, mymat)
    return apply_transform_homogen(homogenous_transform, data, itp)#, out);
    #return out
end

"""
    get_function_tuple(data::AbstractArray, fct_tup::Function; super_sampling=2, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `dat(shift, zoom)` which generates a shifted and scaled version of the original data. 
This is useful for fitting with a function which is itself defined by measured data.

# Arguments
`data`: The data to represent by the function `dat`
`fct_tup`: The function to apply to the data
`super_sampling`: The factor by which the data is internally represented as a supersampled version (Fourier-based upsampling, see `FourierTools.resample`)
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

# Example

"""
function get_function_tuple(data::AbstractArray{T, N}, fct_tup::Function; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where {T, N}
    # building the extraplation + interpolation object
    itp = extrapolate(interpolate(data, interp_type), extrapolation_bc);
    function interpolated(params)#, out = similar(data))
        fct_tup_noparams(ci) = fct_tup(ci, params)
        #@show Interpolations.adapt(gpu_or_cpu(1), itp)
        return apply_transform(fct_tup_noparams, data, itp);
    end
    return interpolated
end

"""
    get_function_svec(data::AbstractArray, fct_hom::Function; super_sampling=1, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(params)` which generates a transformed version of the original data parameterized by transform parameters.

# Arguments
`data`: The data to represent by the function `dat`
`fct_hom`: The function to apply to the data
`super_sampling`: The factor by which the data is internally represented as a supersampled version (Fourier-based upsampling, see `FourierTools.resample`)
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.
"""
function get_function_svec(data::AbstractArray{T}, fct_hom::Function; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where T
    # building the extraplation + interpolation object
    itp = extrapolate(interpolate(data, interp_type), extrapolation_bc);
    function interpolated(params::SVector) #, out = similar(data))
        fct_hom_noparams(c) = fct_hom(c, params)
        return apply_transform_homogen(fct_hom_noparams, data, itp)#, out);
        # return out;
    end
    return interpolated
end

"""
    get_function_affine(data::AbstractArray; super_sampling=1, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated()` which generates a transformed version of the original data parameterized by transform parameters or by transformation matrix.
This is useful for fitting with a function which is itself defined by measured data.
The returned function supports two ways to be used, with an affine transform matrix `matrix_c` as in input or with a vector `p` of parameters. 


# Arguments
`data`: The data to represent by the function `dat`
`super_sampling`: The factor by which the data is internally represented as a supersampled version (Fourier-based upsampling, see `FourierTools.resample`)
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

# Example

```julia
julia> dat1 = reshape(1:16,(4,4))
4×4 reshape(::UnitRange{Int64}, 4, 4) with eltype Int64:
 1  5   9  13
 2  6  10  14
 3  7  11  15
 4  8  12  16

julia> affine_func = get_function_affine(Float32.(dat1));

julia> homogeneous_transform = [1 0 -1; 0 1 1; 0 0 1] # translates by [1,1]
3×3 Matrix{Int64}:
 1  0  -1
 0  1   1
 0  0   1

julia> affine_func(SMatrix{3,3}(homogeneous_transform))
4×4 Matrix{Float32}:
 0.0   0.0   0.0  0.0
 5.0   9.0  13.0  0.0
 6.0  10.0  14.0  0.0
 7.0  11.0  15.0  0.0
```
"""
function get_function_affine(data::AbstractArray{T}; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where T
    #new_size = super_sampling.*size(data)
    #upsampled = fftshift(resample(ifftshift(data), new_size))

    # building the extraplation + interpolation object
    itp = extrapolate(interpolate(data, interp_type), extrapolation_bc);

    function interpolated(matrix_c::SMatrix{T}) where T #, out = similar(data)) where T1
        return apply_transform_affine(matrix_c, data, itp)# , out);
        # return out;
    end

    function interpolated(p::AbstractVector{T1}) where {T1} #, out = similar(data)) where T1 
        x_cen, y_cen = (size(data) .÷ 2.0 .+1)
        # x_cen_up, y_cen_up = (size(upsampled) .÷ 2.0 .+ 1.0)

        # creating the matrices of rotation, shear, scale, and shift
        rot_mat =  @SMatrix T[cos(p[7])  -1.0*sin(p[7]) 0.0; sin(p[7])  cos(p[7]) 0.0; 0.0 0.0 1.0];
        shear_mat = @SMatrix T[1.0 p[5] 0.0; p[6] 1.0 0.0; 0.0 0.0 1.0];
        scale_mat = @SMatrix T[1/p[3] 0.0 0.0; 0.0 1/p[4] 0.0; 0.0 0.0 1.0];
        shift_mat = @SMatrix T[1.0 0.0 -1*p[1]; 0.0 1.0 -1*p[2]; 0.0 0.0 1.0];
        t_to_origin = @SMatrix T[1.0 0.0 1*x_cen; 0.0 1.0 y_cen; 0.0 0.0 1.0];
        t_to_center = @SMatrix T[1.0 0.0 -1.0*x_cen; 0.0 1.0 -1.0*y_cen; 0.0 0.0 1.0];
        # t_orig_upsampled = SMatrix{3, 3}(T[1.0 0.0 -1.0*x_cen_up; 0.0 1.0 -1.0*y_cen_up; 0.0 0.0 1.0]);

        # building the overall transformation matrix
        matrix_c = t_to_origin * scale_mat * rot_mat * shear_mat *shift_mat * t_to_center

        return apply_transform_affine(matrix_c, data, itp) #, out); # do not call interolated here for type stability reasons
    end

    
    function interpolated(p::NTuple{N, T}) where {N, T} #, out = similar(data)) where T1 
        x_cen, y_cen = (size(data) .÷ 2.0 .+1)
        # x_cen_up, y_cen_up = (size(upsampled) .÷ 2.0 .+ 1.0)

        # creating the matrices of rotation, shear, scale, and shift
        rot_mat =  @SMatrix T[cos(p[7])  -1.0*sin(p[7]) 0.0; sin(p[7])  cos(p[7]) 0.0; 0.0 0.0 1.0];
        shear_mat = @SMatrix T[1.0 p[5] 0.0; p[6] 1.0 0.0; 0.0 0.0 1.0];
        scale_mat = @SMatrix T[1/p[3] 0.0 0.0; 0.0 1/p[4] 0.0; 0.0 0.0 1.0];
        shift_mat = @SMatrix T[1.0 0.0 -1*p[1]; 0.0 1.0 -1*p[2]; 0.0 0.0 1.0];
        t_to_origin = @SMatrix T[1.0 0.0 1*x_cen; 0.0 1.0 y_cen; 0.0 0.0 1.0];
        t_to_center = @SMatrix T[1.0 0.0 -1.0*x_cen; 0.0 1.0 -1.0*y_cen; 0.0 0.0 1.0];
        # t_orig_upsampled = SMatrix{3, 3}(T[1.0 0.0 -1.0*x_cen_up; 0.0 1.0 -1.0*y_cen_up; 0.0 0.0 1.0]);

        # building the overall transformation matrix
        matrix_c = t_to_origin * scale_mat * rot_mat * shear_mat *shift_mat * t_to_center

        return apply_transform_affine(matrix_c, data, itp) #, out); # do not call interolated here for type stability reasons
    end

    return interpolated
end


"""
    get_function_poly(data::AbstractArray, order; super_sampling=1, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(p, [out])` which generates a transformed version of the original data parameterized by transform parameters.
This is useful for fitting with a function which is itself defined by measured data.
The returned function supports two ways to be used, with an affine transform matrix `p` as in input or with a vector `p` of parameters. 
The optional argument `out` can be used to store the result of the transformation.


# Arguments
`data`: The data to represent by the function `dat`
`order`: The order of the polynomial to use for the transformation
`super_sampling`: The factor by which the data is internally represented as a supersampled version (Fourier-based upsampling, see `FourierTools.resample`)
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.
"""
function get_function_poly(data::AbstractArray{T, N}, order; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where {T, N}
    pm =  get_multi_poly(Val(ndims(data)), Val(order)) 
    return get_function_tuple(data, pm; super_sampling= super_sampling, extrapolation_bc=extrapolation_bc, interp_type=interp_type);
end

"""
    get_interpolated_function(data::AbstractArray, ::Type{AffineMode}; super_sampling=2, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(p)` which generates a transformed version of the original data parameterized by transform parameters.
This is useful for fitting with a function which is itself defined by measured data.
The returned function supports two ways to be used, with an affine transform matrix `p` as in input or with a vector or tuple `p` of parameters.

# Arguments
`data`: The data to represent by the function `dat`
`AffineMode`: The transformation mode to use
`super_sampling`: The factor by which the data is internally represented as a supersampled version (Fourier-based upsampling, see `FourierTools.resample`)
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

# Returns
A function `interpolated(p)` which generates a transformed version of the original data parameterized by transform parameters
"""
function get_interpolated_function(data::AbstractArray{T, N}, ::Type{AffineMode}; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where {T, N}
    return get_function_affine(data; super_sampling=super_sampling, extrapolation_bc=extrapolation_bc, interp_type=interp_type)
end

function get_interpolated_function(data::AbstractArray{T, N}; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where {T, N}
    @warn "No transformation mode provided. `AffineMode` is used as default"
    return get_interpolated_function(data, AffineMode; super_sampling=super_sampling, extrapolation_bc=extrapolation_bc, interp_type=interp_type)    
end

"""
    get_interpolated_function(data::AbstractArray, ::Type{PolynomialMode}, order=nothing; super_sampling=2, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(p)` which generates a transformed version of the original data parameterized by transform parameters.
This is useful for fitting with a function which is itself defined by measured data.
The returned function supports polynomial transformations of the data.

# Arguments
`data`: The data to represent by the function `dat`
`PolynomialMode`: The transformation mode to use
`order`: The order of the polynomial to use for the transformation
`super_sampling`: The factor by which the data is internally represented as a supersampled version (Fourier-based upsampling, see `FourierTools.resample`)
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

# Returns
A function `interpolated(p)` which generates a transformed version of the original data parameterized by polynomial transform parameters
"""
function get_interpolated_function(data::AbstractArray{T, N}, ::Type{PolynomialMode}, order=nothing; super_sampling=2, extrapolation_bc=zero(eltype(data)), interp_type=Interpolations.BSpline(Linear())) where {T, N}
    #TODO from CuArray to CuArray
    if isnothing(order)
        error("Providing the order of the transformation polynomial is mandatory for the `PolynomialMode`")
    end
    return get_function_poly(data, order; super_sampling=super_sampling, extrapolation_bc=extrapolation_bc, interp_type=interp_type)
end