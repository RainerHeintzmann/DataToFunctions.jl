module DataToFunctions
using Interpolations
using FourierTools
using Optim #, LineSearches
using Revise
using StaticArrays #, LinearAlgebra
using Zygote

export get_function, perform_fit, get_function_general, perform_fit_general, get_function_general_matrix

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



"""
    perform_fit(loss_function, fitting_data::AbstractArray)

Performs a fit to the fitting data using a loss function defined by the user

# Arguments
`loss_function`: User-defined loss function which is minimized
`fitting_data`: The data which is being fitted 

# Returns
a vector of 4 parameters: first 2 for the shift and other 2 for the scaling factors

# Example
there is an example of this function in the `examples/star_fitting.jl`

"""
function perform_fit(loss_function, fitting_data::AbstractArray)
    # guess the shift parameters by taking the maximum values of the array and
    # centering the positions
    a, b = Tuple(argmax(fitting_data)) .- size(fitting_data) ./2.0 .- 1.0
    
    # assigning the initial parameter estimates
    init_x = [a, b, 1.0, 1.0]

    # setting the lower and upper boundary of the parameter values based on the limits of the shift and scaling
    lower = [-1*size(fitting_data)[1], -1*size(fitting_data)[2], 0.0, 0.0]
    upper = [size(fitting_data)[1], size(fitting_data)[2], size(fitting_data)[1], size(fitting_data)[2]]

    # initializing the LBFGS optimizer
    inner_optimizer = LBFGS(; m=1, linesearch=LineSearches.BackTracking(order=2))
    
    # Computer, Optimize! :D
    res = optimize(
            loss_function, 
            lower, upper, 
             init_x, 
            Fminbox(inner_optimizer), 
            Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
             autodiff = :forward
        )
    
    # return the estimated parameters
    return Optim.minimizer(res)
end


"""
    get_function_general(data::AbstractArray; super_sampling=1, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(p)` which generates a transformed version of the original data. 
This is useful for fitting with a function which is itself defined by measured data.

# Arguments
`data`: The data to represent by the function `dat`
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

"""
function get_function_general(data::AbstractArray; extrapolation_bc=0.0, interp_type=Interpolations.BSpline(Linear()))
    # new_size = super_sampling.*size(data)
    # upsampled = fftshift(resample(ifftshift(data), new_size))

    # building the extraplation + interpolation object
    itp = extrapolate(interpolate(data, interp_type), extrapolation_bc);

    # multiplying the transformation matrix
    function f(t::SVector, matrix_c::SMatrix) 
        return matrix_c * t
    end

    function interpolated(p)
        
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
        matrix_c = t_to_origin * scale_mat * shear_mat * rot_mat *shift_mat * t_to_center

        # looping all over the catesian indedices of the input image, 
        # first ading a new value to its third dimenstion: 1.0,
        # converting to the new indices using the transformation matrix and then,
        # using the "itp" object, we build the transfomed image "out"
        for I in CartesianIndices(data)
            out[I] = itp(f(SVector(Tuple(I)..., 1), matrix_c)[1:2]...)
        end
        
        return out
    end

    return interpolated
end



"""
    get_function_general_matrix(data::AbstractArray; super_sampling=1, extrapolation_bc=Flat(), interp_type=Interpolations.BSpline(Linear()))

returns a function `interpolated(p)` which generates a transformed version of the original data. 
This is useful for fitting with a function which is itself defined by measured data.

# Arguments
`data`: The data to represent by the function `dat`
`extrapolation_bc`: The extrapolation boundary condition to select for values outside the range. 
    By default the value 0.0 is used. Other options are `Flat()`, or `Line()`, See the package `Interpolation` for details.
`interp_type`: The type of interpolation to use. See the package `Interpolation` for details.

"""
function get_function_general_matrix(data::AbstractArray{T}; extrapolation_bc=0.0, interp_type=Interpolations.BSpline(Linear())) where T
    # new_size = super_sampling.*size(data)
    # upsampled = fftshift(resample(ifftshift(data), new_size))

    # building the extraplation + interpolation object
    itp = extrapolate(interpolate(data, interp_type), extrapolation_bc);

    # multiplying the transformation matrix
    function f(t::AbstractVector{T}, matrix_c::AbstractMatrix{T}) where T
        return matrix_c * t
    end

    function interpolated(p::AbstractMatrix{T}) where T
        
        # init a new array for the output
        out = similar(data, T)

        # x_cen, y_cen = (size(data) .÷ 2.0 .+1)

        # t_orig_upsampled = @SMatrix [1.0 0.0 -1.0*x_cen_up; 0.0 1.0 -1.0*y_cen_up; 0.0 0.0 1.0]

        # building the overall transformation matrix
        matrix_c = SMatrix{ndims(data)+1, ndims(data)+1, T}(reshape(p, ndims(data)+1, ndims(data)+1))
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

    return interpolated
end



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
function perform_fit_general(loss_function, fitting_data::AbstractArray)
    # guess the shift parameters by taking the maximum values of the array and
    # centering the positions
    ##a, b = Tuple(argmax(fitting_data)) .- size(fitting_data) ./2.0 .- 1.0
    #print("INSIDE!!! hehe")
    # assigning the initial parameter estimates
    init_x = reshape(Matrix(1.0*I, ndims(fitting_data)+1, ndims(fitting_data)+1), 1, 9) #[a, b, 1.0, 1.0, 0.001, 0.001, 0.001]

    # setting the lower and upper boundary of the parameter values based on their limits
    ##lower = [-1*size(fitting_data)[1], -1*size(fitting_data)[2], 0.0, 0.0, 0.0, 0.0, 0.0]
    ##upper = [size(fitting_data)[1], size(fitting_data)[2], size(fitting_data)[1], size(fitting_data)[2], 5.0, 5.0, pi]

    # initializing the LBFGS optimizer
    # inner_optimizer = LBFGS(; m=1, linesearch=LineSearches.BackTracking(order=2))
    
    # Computer, Optimize! :D
    res = optimize(
            loss_function, 
            init_x,
            LBFGS(),
            # lower, upper,
            # Fminbox(inner_optimizer), 
            Optim.Options(store_trace = true, extended_trace = true, iterations=500), 
            autodiff = :forward
        )
    
    # return the estimated parameters
    return Optim.minimizer(res), res
end

end # module DataToFunctions