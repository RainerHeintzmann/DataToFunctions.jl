module DataToFunctions
using Interpolations
using FourierTools
using StaticArrays

export get_function, get_function_affine, add_dim, red_dim_apply, red_dim, f
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

@inline function red_dim_apply(fct::AbstractArray{R}, svec::SVector{S,T})::R where {S,T, R}
    return fct((@view svec[1:2])...)
end

@inline function red_dim(svec::SVector{S,T})::SVector{S-1,T} where {S,T}
    return @view svec[1:S-1]
end

# multiplying the transformation matrix
@inline function f(t::SVector{N, Int}, matrix_c::SMatrix{N,N,T})::SVector{N,T} where {N,T}
    return matrix_c * t
end

"""
    get_function_affine(data::AbstractArray; super_sampling=1, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(p)` which generates a transformed version of the original data. 
This is useful for fitting with a function which is itself defined by measured data.

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


    function interpolated(matrix_c::SMatrix)
        
        # init a new array for the output
        out = similar(data, T)

        # x_cen, y_cen = (size(data) .÷ 2.0 .+1)

        # t_orig_upsampled = @SMatrix [1.0 0.0 -1.0*x_cen_up; 0.0 1.0 -1.0*y_cen_up; 0.0 0.0 1.0]

        # building the overall transformation matrix
        #   matrix_c = SMatrix{ndims(data)+1, ndims(data)+1, T}(reshape(p, ndims(data)+1, ndims(data)+1))
        
        #print(eltype(matrix_c))
        # looping all over the catesian indedices of the input image, 
        # first ading a new value to its third dimenstion: 1.0,
        # converting to the new indices using the transformation matrix and then,
        # using the "itp" object, we build the transfomed image "out"
        for I1 in CartesianIndices(data)
            #print("INSIde the loop!!")
            #print(eltype(SVector{ndims(data)+1, T}(Tuple(I1)..., 1)))
            out[I1] = itp(f(SVector{ndims(data)+1, T}(Tuple(I1)..., 1.0), matrix_c)[1:2]...)
            #print(eltype(out[I1]))
        end
        
        return out
    end

    function interpolated(p::AbstractVector{T}) where T     
        # init a new array for the output
        out = similar(data)

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

        out[CartesianIndices(data)] .= red_dim_apply.(Ref(itp), f.(add_dim.(CartesianIndices(data)), Ref(matrix_c)));

        return out
    end

    return interpolated
end



end # module DataToFunctions