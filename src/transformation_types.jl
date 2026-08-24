abstract type transformation_method end

"""
    AffineMode

Transformation mode selecting affine coordinate transformations.

See also [`get_function_affine`](@ref) and
[`get_interpolated_function`](@ref).
"""
struct AffineMode <: transformation_method end

"""
    PolynomialMode

Transformation mode selecting polynomial coordinate transformations.

When using this mode with [`get_interpolated_function`](@ref), the polynomial
order must also be specified.
"""
struct PolynomialMode <: transformation_method end