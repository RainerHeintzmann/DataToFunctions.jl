module DataToFunctions
using Interpolations
using FourierTools
export get_function

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
    @show upsampled
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

    # return (pos) -> interp_linear((center .+ pos)...)
    # fitp(t) = interp_linear(t...)
    # @time res1 = fitp.(tcoords);  # 1 sec
    # function my_zoom

end

end # module DataToFunctions
