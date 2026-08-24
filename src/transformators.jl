using Interpolations
using FourierTools
using EvalMultiPoly
using Adapt
import ChainRulesCore

export get_interpolated_function
export get_function_tuple, get_function_tuple_inplace
export get_function_homogen, get_function_homogen_inplace, get_function_svec
export get_function_affine, get_function_affine_inplace
export get_function_poly, get_function_poly_inplace
# export apply_transform, apply_transform!
# export apply_transform_homogen, apply_transform_homogen!
# export apply_transform_affine, apply_transform_affine!
# export add_dim, red_dim_apply, red_dim, mat_mul
# export func_transform, func_transform_tup
export PolynomialMode, AffineMode

# -----------------------------------------------------------------------------
# Legacy implementation -- intentionally left unchanged
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Internal utilities
# -----------------------------------------------------------------------------

# Interpolations.jl currently supports GPU *usage* of interpolants by first
# constructing the interpolant on CPU and then adapting its storage to the target
# backend. Constructing directly from a GPU array may leave CPU `Array`
# coefficients inside the interpolant, which cannot be passed to a GPU kernel.
#
# CPU path: no copy and no adaptation.
@inline function _make_interpolant(
    data::Array,
    interp_type,
    extrapolation_bc,
)
    return extrapolate(
        interpolate(data, interp_type),
        extrapolation_bc,
    )
end

# Generic backend path (e.g. CuArray):
#   backend array -> CPU values -> CPU interpolant -> backend-adapted interpolant
#
# `typeof(data)` is used as the Adapt target, so DataToFunctions does not need to
# import or depend on CUDA.jl. CUDA.jl registers the corresponding Adapt storage
# rule when the user loads CUDA.
function _make_interpolant(
    data::AbstractArray,
    interp_type,
    extrapolation_bc,
)
    cpu_data = Adapt.adapt(Array, data)

    cpu_itp = extrapolate(
        interpolate(cpu_data, interp_type),
        extrapolation_bc,
    )

    return Adapt.adapt(
        typeof(data),
        cpu_itp,
    )
end

"""
    add_dim(v::Tuple)

Append a homogeneous coordinate equal to one to the coordinate tuple `v`.

The tuple length remains part of the concrete type and the elements are allowed
to have heterogeneous scalar types. This is useful for allocation-free static
coordinate arithmetic and for automatic-differentiation number types.

# Arguments
- `v`: Cartesian coordinate tuple.

# Returns
An `(N + 1)`-tuple whose final element is `one(v[1])`.

# Examples
```julia
julia> add_dim((2.0, 3.0))
(2.0, 3.0, 1.0)
```
"""
@inline function add_dim(v::Tuple{Vararg{Any,N}}) where {N}
    return ntuple(i -> i <= N ? v[i] : one(v[1]), Val(N + 1))
end

"""
    red_dim(v::Tuple)

Convert homogeneous coordinates to Cartesian coordinates.

For an `N`-component homogeneous coordinate, the first `N-1` entries are divided
by the final homogeneous component.

# Arguments
- `v`: Homogeneous coordinate tuple.

# Returns
An `(N - 1)`-tuple containing the normalized Cartesian coordinates.

# Examples
```julia
julia> red_dim((4.0, 6.0, 2.0))
(2.0, 3.0)
```
"""
@inline function red_dim(v::Tuple{Vararg{Any,N}}) where {N}
    last_inv = inv(v[N])
    return ntuple(i -> v[i] * last_inv, Val(N - 1))
end

"""
    red_dim_apply(f, v::Tuple)

Drop the final component of `v` and splat the remaining values into `f`.

This helper is retained for compatibility. It does **not** normalize by the
final homogeneous component. Use [`red_dim`](@ref) when homogeneous
normalization is required.
"""
@inline function red_dim_apply(f, v::Tuple{Vararg{Any,N}}) where {N}
    return f(ntuple(i -> v[i], Val(N - 1))...)
end

@inline idx_apply(f, v::Tuple) = f(v...)
@inline func_transform(t, coord_transform_func) = coord_transform_func(Tuple(t))
@inline func_transform_tup(t, coord_transform_func) = coord_transform_func(Tuple(t))

# Matrices are represented as an outer tuple of row tuples:
#
# ((a11, a12, ...),
#  (a21, a22, ...),
#  ...)
#
# The operations below are fully unrolled for statically-known tuple sizes.

@inline function _tuple_matvec(A::NTuple{N,<:Tuple}, x::NTuple{N}) where {N}
    return ntuple(
        i -> sum(ntuple(j -> A[i][j] * x[j], Val(N))),
        Val(N),
    )
end

@inline function _tuple_matmul(
    A::NTuple{N,<:Tuple},
    B::NTuple{N,<:Tuple},
) where {N}
    return ntuple(
        i -> ntuple(
            j -> sum(ntuple(k -> A[i][k] * B[k][j], Val(N))),
            Val(N),
        ),
        Val(N),
    )
end

"""Multiply tuple matrix `A` by tuple coordinate `v`, returning `A * v`."""
@inline mat_mul(v::NTuple{N}, A::NTuple{N,<:Tuple}) where {N} =
    _tuple_matvec(A, v)

"""Multiply two square tuple matrices."""
@inline mat_mul(A::NTuple{N,<:Tuple}, B::NTuple{N,<:Tuple}) where {N} =
    _tuple_matmul(A, B)

# -----------------------------------------------------------------------------
# Backend-generic indices and transform evaluation
# -----------------------------------------------------------------------------

# Plain Arrays use the lazy CartesianIndices object directly. This keeps the CPU
# in-place hot path allocation-free.
@inline _backend_indices(data::Array) = CartesianIndices(axes(data))

# Other AbstractArray backends (e.g. CuArray) get an index array on the same
# backend. It is constructed once by get_function_* and reused by every call.
#
# CartesianIndex is an isbits type and can therefore be stored in GPU memory.
function _backend_indices(data::AbstractArray{T,N}) where {T,N}
    inds = similar(data, CartesianIndex{N})
    inds .= CartesianIndices(axes(data))
    return inds
end

@inline function _transform_sample(_, I, itp, coord_transform)
    coords = coord_transform(Tuple(I))
    return itp(coords...)
end

# CPU pure path: only the returned image needs to be allocated.
function _apply_transform(
    coord_transform,
    data::Array,
    itp,
    inds::CartesianIndices,
)
    return map(inds) do I
        coords = coord_transform(Tuple(I))
        itp(coords...)
    end
end

# Generic-array path. A GPU array among the broadcast arguments selects the GPU
# broadcast backend; no CUDA-specific code is required here.
function _apply_transform(
    coord_transform,
    data::AbstractArray,
    itp,
    inds,
)
    return _transform_sample.(
        data,                 # backend driver; value intentionally unused
        inds,
        Ref(itp),
        Ref(coord_transform),
    )
end

"""
    apply_transform(coord_transform, data, itp)

Apply a coordinate transform to every output index and sample the interpolation
object `itp` at the transformed coordinates.

`coord_transform` receives an `N`-tuple and must return an `N`-tuple for
`N`-dimensional `data`.

# Arguments
- `coord_transform`: Function mapping output coordinates to sampling coordinates.
- `data`: Array defining the output axes and execution backend.
- `itp`: Callable interpolation/extrapolation object.

# Returns
A newly allocated array on the same backend as `data`.

# Performance
For `Array`, the implementation uses a specialized Cartesian traversal. Other
`AbstractArray` backends use broadcast, which allows GPU arrays such as
`CuArray` to execute without scalar host indexing when the interpolant is
backend-compatible.

See also [`apply_transform!`](@ref).
"""
function apply_transform(coord_transform, data::AbstractArray, itp)
    return _apply_transform(
        coord_transform,
        data,
        itp,
        _backend_indices(data),
    )
end

# Strict CPU performance path.
function _apply_transform!(
    out::Array,
    data::Array,
    coord_transform,
    itp,
    inds::CartesianIndices,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    @inbounds for I in inds
        coords = coord_transform(Tuple(I))
        out[I] = itp(coords...)
    end
    return out
end

# Backend-generic in-place path. On CuArray this becomes a broadcast kernel.
function _apply_transform!(
    out::AbstractArray,
    data::AbstractArray,
    coord_transform,
    itp,
    inds,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    out .= _transform_sample.(
        data,
        inds,
        Ref(itp),
        Ref(coord_transform),
    )
    return out
end

"""
    apply_transform!(out, coord_transform, itp)

In-place counterpart of [`apply_transform`](@ref).

The transformed samples are written into the preallocated array `out`.

# Arguments
- `out`: Destination array. Its axes define the output coordinate grid.
- `coord_transform`: Function mapping output coordinates to sampling coordinates.
- `itp`: Callable interpolation/extrapolation object.

# Returns
`out`.

# Performance
For a plain CPU `Array`, the hot loop is designed to perform no heap allocations
after compilation. Other array backends use an in-place broadcast.
"""
function apply_transform!(out::AbstractArray, coord_transform, itp)
    return _apply_transform!(
        out,
        out,
        coord_transform,
        itp,
        _backend_indices(out),
    )
end

# -----------------------------------------------------------------------------
# Homogeneous transforms
# -----------------------------------------------------------------------------

@inline function _homogeneous_transform(x, coord_transform)
    return red_dim(coord_transform(add_dim(x)))
end

function _apply_transform_homogen(
    coord_transform,
    data::AbstractArray,
    itp,
    inds,
)
    return _apply_transform(
        x -> _homogeneous_transform(x, coord_transform),
        data,
        itp,
        inds,
    )
end

"""
    apply_transform_homogen(coord_transform, data, itp)

Apply a transform expressed in homogeneous coordinates.

For `N`-dimensional data, `coord_transform` receives an `(N + 1)`-tuple and must
return an `(N + 1)`-tuple. The returned homogeneous coordinate is normalized
with [`red_dim`](@ref) before interpolation.

# Returns
A newly allocated transformed array.

See also [`apply_transform_homogen!`](@ref) and [`apply_transform_affine`](@ref).
"""
function apply_transform_homogen(coord_transform, data::AbstractArray, itp)
    return _apply_transform_homogen(
        coord_transform,
        data,
        itp,
        _backend_indices(data),
    )
end

# CPU specialization keeps the homogeneous operations directly in the hot loop.
function _apply_transform_homogen!(
    out::Array,
    data::Array,
    coord_transform,
    itp,
    inds::CartesianIndices,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    @inbounds for I in inds
        hcoords = add_dim(Tuple(I))
        coords = red_dim(coord_transform(hcoords))
        out[I] = itp(coords...)
    end
    return out
end

# Generic-array path (including GPU arrays).
function _apply_transform_homogen!(
    out::AbstractArray,
    data::AbstractArray,
    coord_transform,
    itp,
    inds,
)
    return _apply_transform!(
        out,
        data,
        x -> _homogeneous_transform(x, coord_transform),
        itp,
        inds,
    )
end

"""
    apply_transform_homogen!(out, coord_transform, itp)

In-place counterpart of [`apply_transform_homogen`](@ref).

For a CPU `Array`, the homogeneous conversion, coordinate transformation, and
interpolation are kept directly in the hot loop to support allocation-free
steady-state execution.

# Returns
`out`.
"""
function apply_transform_homogen!(out::AbstractArray, coord_transform, itp)
    return _apply_transform_homogen!(
        out,
        out,
        coord_transform,
        itp,
        _backend_indices(out),
    )
end

# -----------------------------------------------------------------------------
# Affine / projective transforms
# -----------------------------------------------------------------------------

@inline function _check_homogeneous_matrix(matrix, N)
    length(matrix) == N + 1 || throw(
        DimensionMismatch(
            "an $N-dimensional array requires a $(N + 1)×$(N + 1) homogeneous matrix",
        ),
    )
    return nothing
end

function _apply_transform_affine(
    matrix::NTuple{M,<:Tuple},
    data::AbstractArray{T,N},
    itp,
    inds,
) where {M,T,N}
    _check_homogeneous_matrix(matrix, N)
    return _apply_transform_homogen(
        x -> mat_mul(x, matrix),
        data,
        itp,
        inds,
    )
end

"""
    apply_transform_affine(matrix, data, itp)

Apply an affine or projective transform represented by a homogeneous tuple
matrix.

For `N`-dimensional `data`, `matrix` must contain `N + 1` row tuples, each of
length `N + 1`. Matrix storage is row-major at the Julia tuple level and the
operation corresponds to `matrix * coordinate`.

# Arguments
- `matrix`: Homogeneous transformation matrix represented as an outer tuple of row tuples.
- `data`: Array defining output axes and backend.
- `itp`: Callable interpolation/extrapolation object.

# Returns
A newly allocated transformed array.

# Examples
```julia
A = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
y = apply_transform_affine(A, data, itp)
```
"""
function apply_transform_affine(
    matrix::NTuple{M,<:Tuple},
    data::AbstractArray{T,N},
    itp,
) where {M,T,N}
    return _apply_transform_affine(
        matrix,
        data,
        itp,
        _backend_indices(data),
    )
end

# CPU specialization: keep matrix multiplication, homogeneous reduction, and
# interpolation directly in one loop. This is the strict zero-allocation path.
function _apply_transform_affine!(
    out::Array{T,N},
    data::Array,
    matrix::NTuple{M,<:Tuple},
    itp,
    inds::CartesianIndices,
) where {T,N,M}
    _check_homogeneous_matrix(matrix, N)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    @inbounds for I in inds
        hcoords = add_dim(Tuple(I))
        coords = red_dim(mat_mul(hcoords, matrix))
        out[I] = itp(coords...)
    end
    return out
end

# Generic-array path (including GPU arrays).
function _apply_transform_affine!(
    out::AbstractArray{T,N},
    data::AbstractArray,
    matrix::NTuple{M,<:Tuple},
    itp,
    inds,
) where {T,N,M}
    _check_homogeneous_matrix(matrix, N)
    return _apply_transform_homogen!(
        out,
        data,
        x -> mat_mul(x, matrix),
        itp,
        inds,
    )
end

"""
    apply_transform_affine!(out, matrix, itp)

In-place counterpart of [`apply_transform_affine`](@ref).

# Returns
`out`.

# Performance
For CPU `Array`s, matrix multiplication, homogeneous reduction, and
interpolation are fused into the explicit hot loop. With tuple matrices and a
preallocated output, the steady-state path is intended to be allocation-free.
"""
function apply_transform_affine!(
    out::AbstractArray{T,N},
    matrix::NTuple{M,<:Tuple},
    itp,
) where {T,N,M}
    return _apply_transform_affine!(
        out,
        out,
        matrix,
        itp,
        _backend_indices(out),
    )
end

# 2-D seven-parameter model:
#
# p = (
#     shift_x,
#     shift_y,
#     scale_x,
#     scale_y,
#     shear_xy,
#     shear_yx,
#     rotation,
# )

@inline function _check_affine_params_2d(p)
    length(p) == 7 || throw(ArgumentError(
        "the 2-D affine parameterization requires 7 parameters: " *
        "(shift_x, shift_y, scale_x, scale_y, shear_xy, shear_yx, rotation)",
    ))
    return nothing
end

@inline function _affine_matrix_2d(p, x_cen, y_cen)
    _check_affine_params_2d(p)

    z = zero(p[1])
    o = one(p[1])
    c = cos(p[7])
    s = sin(p[7])

    rot_mat = (
        (c, -s, z),
        (s,  c, z),
        (z,  z, o),
    )

    shear_mat = (
        (o,    p[5], z),
        (p[6], o,    z),
        (z,    z,    o),
    )

    scale_mat = (
        (o / p[3], z,        z),
        (z,        o / p[4], z),
        (z,        z,        o),
    )

    shift_mat = (
        (o, z, -p[1]),
        (z, o, -p[2]),
        (z, z, o),
    )

    t_to_origin = (
        (o, z,  o * x_cen),
        (z, o,  o * y_cen),
        (z, z,  o),
    )

    t_to_center = (
        (o, z, -o * x_cen),
        (z, o, -o * y_cen),
        (z, z, o),
    )

    matrix = _tuple_matmul(t_to_origin, scale_mat)
    matrix = _tuple_matmul(matrix, rot_mat)
    matrix = _tuple_matmul(matrix, shear_mat)
    matrix = _tuple_matmul(matrix, shift_mat)
    return _tuple_matmul(matrix, t_to_center)
end

"""
    _affine_warp_2d_params(p, data, itp, inds)

Named pure seven-parameter affine warp.

The named boundary is intentional: the custom ChainRules rule below replaces a
very expensive scalar-by-scalar Zygote reverse pass with a single image-level
vector-Jacobian product.
"""
function _affine_warp_2d_params(p, data::AbstractMatrix, itp, inds)
    _check_affine_params_2d(p)
    x_cen = size(data, 1) ÷ 2 + 1
    y_cen = size(data, 2) ÷ 2 + 1
    matrix = _affine_matrix_2d(p, x_cen, y_cen)
    return _apply_transform_affine(matrix, data, itp, inds)
end

# Tuple reduction helper used by backend-generic VJPs.
#
# This deliberately does not introduce a custom accumulator type and does not
# extend/overload Base arithmetic. The reduction values are plain isbits tuples.
@inline function _tuple_add(
    a::NTuple{N},
    b::NTuple{N},
) where {N}
    return ntuple(
        i -> a[i] + b[i],
        Val(N),
    )
end

@inline function _affine_pixel_vjp(
    I,
    δ,
    p,
    itp,
    x_cen,
    y_cen,
    ::Type{G},
) where {G}
    shift_x, shift_y = p[1], p[2]
    scale_x, scale_y = p[3], p[4]
    shear_xy, shear_yx = p[5], p[6]
    θ = p[7]

    c = cos(θ)
    s = sin(θ)
    inv_scale_x = inv(scale_x)
    inv_scale_y = inv(scale_y)

    x = I[1]
    y = I[2]

    a = x - x_cen - shift_x
    b = y - y_cen - shift_y
    u = a + shear_xy * b
    v = shear_yx * a + b

    r1 = c * u - s * v
    r2 = s * u + c * v

    x′ = x_cen + r1 * inv_scale_x
    y′ = y_cen + r2 * inv_scale_y

    grad = Interpolations.gradient(itp, x′, y′)
    gx = grad[1]
    gy = grad[2]

    inv_scale_x2 = inv_scale_x * inv_scale_x
    inv_scale_y2 = inv_scale_y * inv_scale_y

    dx_dp1 = (-c + s * shear_yx) * inv_scale_x
    dy_dp1 = (-s - c * shear_yx) * inv_scale_y
    dx_dp2 = (-c * shear_xy + s) * inv_scale_x
    dy_dp2 = (-s * shear_xy - c) * inv_scale_y

    vals = (
        δ * (gx * dx_dp1 + gy * dy_dp1),
        δ * (gx * dx_dp2 + gy * dy_dp2),
        δ * gx * (-r1 * inv_scale_x2),
        δ * gy * (-r2 * inv_scale_y2),
        δ * (
            gx * (c * b * inv_scale_x) +
            gy * (s * b * inv_scale_y)
        ),
        δ * (
            gx * (-s * a * inv_scale_x) +
            gy * ( c * a * inv_scale_y)
        ),
        δ * (
            gx * (-r2 * inv_scale_x) +
            gy * ( r1 * inv_scale_y)
        ),
    )

    return ntuple(
        i -> convert(G, vals[i]),
        Val(7),
    )
end

# CPU VJP: one pass, one small preallocated interpolation-gradient buffer.
function _affine_vjp_2d(
    p,
    data::AbstractMatrix,
    itp,
    Δ,
    inds::CartesianIndices,
)
    pt = ntuple(i -> p[i], Val(7))
    P = promote_type(ntuple(i -> typeof(pt[i]), Val(7))...)
    G = promote_type(P, eltype(data), eltype(Δ))

    shift_x, shift_y = pt[1], pt[2]
    scale_x, scale_y = pt[3], pt[4]
    shear_xy, shear_yx = pt[5], pt[6]
    θ = pt[7]

    x_cen = size(data, 1) ÷ 2 + 1
    y_cen = size(data, 2) ÷ 2 + 1

    c = cos(θ)
    s = sin(θ)
    inv_scale_x = inv(scale_x)
    inv_scale_y = inv(scale_y)
    inv_scale_x2 = inv_scale_x * inv_scale_x
    inv_scale_y2 = inv_scale_y * inv_scale_y

    dx_dp1 = (-c + s * shear_yx) * inv_scale_x
    dy_dp1 = (-s - c * shear_yx) * inv_scale_y
    dx_dp2 = (-c * shear_xy + s) * inv_scale_x
    dy_dp2 = (-s * shear_xy - c) * inv_scale_y

    g1 = zero(G)
    g2 = zero(G)
    g3 = zero(G)
    g4 = zero(G)
    g5 = zero(G)
    g6 = zero(G)
    g7 = zero(G)

    grad_itp = Vector{G}(undef, 2)

    @inbounds for I in inds
        x = I[1]
        y = I[2]

        a = x - x_cen - shift_x
        b = y - y_cen - shift_y
        u = a + shear_xy * b
        v = shear_yx * a + b
        r1 = c * u - s * v
        r2 = s * u + c * v
        x′ = x_cen + r1 * inv_scale_x
        y′ = y_cen + r2 * inv_scale_y

        Interpolations.gradient!(grad_itp, itp, x′, y′)
        gx = grad_itp[1]
        gy = grad_itp[2]

        δ = Δ[I]
        δgx = δ * gx
        δgy = δ * gy

        g1 += δgx * dx_dp1 + δgy * dy_dp1
        g2 += δgx * dx_dp2 + δgy * dy_dp2
        g3 += δgx * (-r1 * inv_scale_x2)
        g4 += δgy * (-r2 * inv_scale_y2)
        g5 += δgx * (c * b * inv_scale_x) +
              δgy * (s * b * inv_scale_y)
        g6 += δgx * (-s * a * inv_scale_x) +
              δgy * ( c * a * inv_scale_y)
        g7 += δgx * (-r2 * inv_scale_x) +
              δgy * ( r1 * inv_scale_y)
    end

    return (g1, g2, g3, g4, g5, g6, g7)
end

# Backend-generic VJP. With GPU-backed `inds`/`Δ`, GPUArrays/CUDA can reduce the
# lazy broadcast without scalar host indexing.
function _affine_vjp_2d(
    p,
    data::AbstractMatrix,
    itp,
    Δ,
    inds::AbstractArray,
)
    pt = ntuple(i -> p[i], Val(7))
    P = promote_type(ntuple(i -> typeof(pt[i]), Val(7))...)
    G = promote_type(P, eltype(data), eltype(Δ))

    x_cen = size(data, 1) ÷ 2 + 1
    y_cen = size(data, 2) ÷ 2 + 1

    bc = Base.broadcasted(
        _affine_pixel_vjp,
        vec(inds),
        vec(Δ),
        Ref(pt),
        Ref(itp),
        Ref(x_cen),
        Ref(y_cen),
        Ref(G),
    )

    return mapreduce(
        identity,
        _tuple_add,
        bc;
        init=ntuple(_ -> zero(G), Val(7)),
    )
end

@inline _affine_parameter_tangent(::Tuple, g::Tuple) = g
@inline _affine_parameter_tangent(::AbstractVector, g::Tuple) = collect(g)

function ChainRulesCore.rrule(
    ::typeof(_affine_warp_2d_params),
    p,
    data::AbstractMatrix,
    itp,
    inds,
)
    y = _affine_warp_2d_params(p, data, itp, inds)

    function affine_pullback(Δ_raw)
        Δ = ChainRulesCore.unthunk(Δ_raw)

        if Δ isa ChainRulesCore.AbstractZero
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.ZeroTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )
        end

        gp = _affine_vjp_2d(p, data, itp, Δ, inds)

        return (
            ChainRulesCore.NoTangent(),
            _affine_parameter_tangent(p, gp),
            ChainRulesCore.NoTangent(), # source data is treated as constant
            ChainRulesCore.NoTangent(), # interpolation object is constant
            ChainRulesCore.NoTangent(), # backend indices are constant
        )
    end

    return y, affine_pullback
end

"""
    get_function_affine(data; kwargs...)

Create an interpolation-based affine warp of sampled data.

The returned function accepts either:

1. an `(N + 1) × (N + 1)` homogeneous matrix represented as an outer tuple of
   row tuples, for arbitrary-dimensional data; or
2. for 2-D data, the seven-parameter representation
   `(shift_x, shift_y, scale_x, scale_y, shear_xy, shear_yx, rotation)`.

# Keyword arguments
- `super_sampling=2`: Retained for API compatibility. The current optimized
  affine path operates directly on the interpolation grid.
- `extrapolation_bc=zero(eltype(data))`: Boundary condition or fill value used
  outside the interpolation domain.
- `interp_type=Interpolations.BSpline(Linear())`: Interpolation scheme.

# Returns
A callable `warp(transform)` that allocates and returns the transformed array.

# Automatic differentiation
The 2-D seven-parameter path has a custom `ChainRulesCore.rrule` that computes
an image-level vector-Jacobian product. Reverse-mode AD therefore differentiates
with respect to the transform parameters without tracing every pixel operation.
The source data and constructed interpolation object are treated as constants
by this rule.

# GPU support
For non-`Array` backends, the interpolant is constructed on CPU and adapted to
the input backend with Adapt.jl. The forward transform is expressed using
backend-generic broadcast. In particular, a `CuArray` input can execute on CUDA
without DataToFunctions depending directly on CUDA.jl, provided the selected
interpolation/extrapolation combination is GPU-compatible.

# Examples
```julia
using DataToFunctions

img = rand(Float32, 64, 64)
warp = get_function_affine(img)

p = (0.2f0, -0.1f0, 1.01f0, 0.99f0, 0.0f0, 0.0f0, 0.01f0)
warped = warp(p)
```

For repeated forward evaluation with a reusable output buffer, see
[`get_function_affine_inplace`](@ref).
"""
function get_function_affine(
    data::AbstractArray{T,N};
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    _ = super_sampling # retained for API compatibility

    itp = _make_interpolant(data, interp_type, extrapolation_bc)
    inds = _backend_indices(data)

    function interpolated(matrix::NTuple{M,<:Tuple}) where {M}
        return _apply_transform_affine(matrix, data, itp, inds)
    end

    function interpolated(p::AbstractVector)
        N == 2 || throw(ArgumentError(
            "the 7-parameter affine interface is defined only for 2-D data; " *
            "pass an $(N + 1)×$(N + 1) tuple matrix for $N-D data",
        ))
        return _affine_warp_2d_params(p, data, itp, inds)
    end

    function interpolated(p::NTuple{7})
        N == 2 || throw(ArgumentError(
            "the 7-parameter affine interface is defined only for 2-D data; " *
            "pass an $(N + 1)×$(N + 1) tuple matrix for $N-D data",
        ))
        return _affine_warp_2d_params(p, data, itp, inds)
    end

    return interpolated
end

"""
    get_function_affine_inplace(data; kwargs...)

Create the in-place affine warp `warp!(out, transform)`.

The accepted transform representations and keyword arguments are the same as
for [`get_function_affine`](@ref).

# Returns
A callable that writes transformed samples into `out` and returns `out`.

# Performance
For CPU `Array`s, a preallocated output and tuple parameters/matrices are
intended to give zero heap allocations in steady state. For GPU-backed arrays,
the same API uses backend-generic in-place broadcast.

# Automatic differentiation
Use [`get_function_affine`](@ref) for Zygote differentiation. The mutating form
is intended for forward evaluations where output-buffer reuse is important.

# Examples
```julia
img = rand(Float32, 64, 64)
warp! = get_function_affine_inplace(img)
out = similar(img)
p = (0.2f0, -0.1f0, 1.01f0, 0.99f0, 0.0f0, 0.0f0, 0.01f0)
warp!(out, p)
```
"""
function get_function_affine_inplace(
    data::AbstractArray{T,N};
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    _ = super_sampling

    itp = _make_interpolant(data, interp_type, extrapolation_bc)
    inds = _backend_indices(data)

    function interpolated!(out, matrix::NTuple{M,<:Tuple}) where {M}
        return _apply_transform_affine!(out, data, matrix, itp, inds)
    end

    function interpolated!(out, p::AbstractVector)
        N == 2 || throw(ArgumentError(
            "the 7-parameter affine interface is defined only for 2-D data",
        ))
        _check_affine_params_2d(p)
        x_cen = size(data, 1) ÷ 2 + 1
        y_cen = size(data, 2) ÷ 2 + 1
        matrix = _affine_matrix_2d(p, x_cen, y_cen)
        return _apply_transform_affine!(out, data, matrix, itp, inds)
    end

    function interpolated!(out, p::NTuple{7})
        N == 2 || throw(ArgumentError(
            "the 7-parameter affine interface is defined only for 2-D data",
        ))
        x_cen = size(data, 1) ÷ 2 + 1
        y_cen = size(data, 2) ÷ 2 + 1
        matrix = _affine_matrix_2d(p, x_cen, y_cen)
        return _apply_transform_affine!(out, data, matrix, itp, inds)
    end

    return interpolated!
end

# -----------------------------------------------------------------------------
# Parameterized user-transform helpers
# -----------------------------------------------------------------------------

@inline function _param_transform_sample(_, I, itp, f, params)
    coords = f(Tuple(I), params)
    return itp(coords...)
end

function _apply_param_transform(
    data::Array,
    f,
    params,
    itp,
    inds::CartesianIndices,
)
    return map(inds) do I
        coords = f(Tuple(I), params)
        itp(coords...)
    end
end

function _apply_param_transform(
    data::AbstractArray,
    f,
    params,
    itp,
    inds,
)
    return _param_transform_sample.(
        data,
        inds,
        Ref(itp),
        Ref(f),
        Ref(params),
    )
end

function _apply_param_transform!(
    out::Array,
    data::Array,
    f,
    params,
    itp,
    inds::CartesianIndices,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    @inbounds for I in inds
        coords = f(Tuple(I), params)
        out[I] = itp(coords...)
    end
    return out
end

function _apply_param_transform!(
    out::AbstractArray,
    data::AbstractArray,
    f,
    params,
    itp,
    inds,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    out .= _param_transform_sample.(
        data,
        inds,
        Ref(itp),
        Ref(f),
        Ref(params),
    )
    return out
end

@inline function _param_homogeneous_sample(_, I, itp, f, params)
    hcoords = add_dim(Tuple(I))
    coords = red_dim(f(hcoords, params))
    return itp(coords...)
end

function _apply_param_homogeneous(
    data::Array,
    f,
    params,
    itp,
    inds::CartesianIndices,
)
    return map(inds) do I
        hcoords = add_dim(Tuple(I))
        coords = red_dim(f(hcoords, params))
        itp(coords...)
    end
end

function _apply_param_homogeneous(
    data::AbstractArray,
    f,
    params,
    itp,
    inds,
)
    return _param_homogeneous_sample.(
        data,
        inds,
        Ref(itp),
        Ref(f),
        Ref(params),
    )
end

function _apply_param_homogeneous!(
    out::Array,
    data::Array,
    f,
    params,
    itp,
    inds::CartesianIndices,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    @inbounds for I in inds
        hcoords = add_dim(Tuple(I))
        coords = red_dim(f(hcoords, params))
        out[I] = itp(coords...)
    end
    return out
end

function _apply_param_homogeneous!(
    out::AbstractArray,
    data::AbstractArray,
    f,
    params,
    itp,
    inds,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    out .= _param_homogeneous_sample.(
        data,
        inds,
        Ref(itp),
        Ref(f),
        Ref(params),
    )
    return out
end

# -----------------------------------------------------------------------------
# User-defined tuple and homogeneous transforms
# -----------------------------------------------------------------------------

"""
    get_function_tuple(data, f; kwargs...)

Create a parameterized warp from a user-supplied Cartesian coordinate function.

The callback is called as `f(coord, params)`, where `coord` is an `N`-tuple and
must be mapped to another `N`-tuple of sampling coordinates.

# Arguments
- `data`: Sampled input data.
- `f`: Coordinate transformation callback.

# Keyword arguments
- `super_sampling=2`: Retained for API compatibility.
- `extrapolation_bc=zero(eltype(data))`: Extrapolation boundary condition.
- `interp_type=Interpolations.BSpline(Linear())`: Interpolation scheme.

# Returns
A callable `warp(params)` returning a transformed array.

# Examples
```julia
shift(coord, p) = (coord[1] + p[1], coord[2] + p[2])
warp = get_function_tuple(data, shift)
y = warp((0.2, -0.1))
```

See also [`get_function_tuple_inplace`](@ref).
"""
function get_function_tuple(
    data::AbstractArray{T,N},
    f;
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    _ = super_sampling

    itp = _make_interpolant(data, interp_type, extrapolation_bc)
    inds = _backend_indices(data)

    interpolated(params) =
        _apply_param_transform(data, f, params, itp, inds)

    return interpolated
end

"""
    get_function_tuple_inplace(data, f; kwargs...)

Create the in-place counterpart of [`get_function_tuple`](@ref).

The returned function is called as `warp!(out, params)` and writes the
transformed samples into `out`.

For CPU arrays, the callback and interpolation are evaluated directly in the
preallocated hot loop. Other array backends use in-place broadcast.
"""
function get_function_tuple_inplace(
    data::AbstractArray{T,N},
    f;
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    _ = super_sampling

    itp = _make_interpolant(data, interp_type, extrapolation_bc)
    inds = _backend_indices(data)

    function interpolated!(out, params)
        return _apply_param_transform!(
            out,
            data,
            f,
            params,
            itp,
            inds,
        )
    end

    return interpolated!
end

"""
    get_function_homogen(data, f; kwargs...)

Create a parameterized warp using homogeneous coordinates.

For `N`-dimensional data, the callback is called as `f(hcoord, params)`, where
`hcoord` is an `(N + 1)`-tuple. The callback must return another homogeneous
`(N + 1)`-tuple, which is normalized before interpolation.

# Returns
A callable `warp(params)` returning a transformed array.

# Examples
```julia
shift_h(h, p) = (
    h[1] + p[1] * h[3],
    h[2] + p[2] * h[3],
    h[3],
)
warp = get_function_homogen(data, shift_h)
y = warp((0.2, -0.1))
```

See also [`get_function_homogen_inplace`](@ref).
"""
function get_function_homogen(
    data::AbstractArray{T,N},
    f;
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    _ = super_sampling

    itp = _make_interpolant(data, interp_type, extrapolation_bc)
    inds = _backend_indices(data)

    interpolated(params) =
        _apply_param_homogeneous(data, f, params, itp, inds)

    return interpolated
end

"""
    get_function_homogen_inplace(data, f; kwargs...)

Create the in-place counterpart of [`get_function_homogen`](@ref).

The returned callable has the form `warp!(out, params)`. For CPU arrays the
homogeneous transform and interpolation are kept in a direct loop; other array
backends use in-place broadcast.
"""
function get_function_homogen_inplace(
    data::AbstractArray{T,N},
    f;
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    _ = super_sampling

    itp = _make_interpolant(data, interp_type, extrapolation_bc)
    inds = _backend_indices(data)

    function interpolated!(out, params)
        return _apply_param_homogeneous!(
            out,
            data,
            f,
            params,
            itp,
            inds,
        )
    end

    return interpolated!
end

"""
    get_function_svec(args...; kwargs...)

Compatibility alias for [`get_function_homogen`](@ref).

This function is retained for compatibility with the previous
StaticArrays-based interface. The current implementation uses tuple-based
homogeneous coordinates instead.
"""
# Compatibility name retained from the previous StaticArrays-based API.
get_function_svec(args...; kwargs...) =
    get_function_homogen(args...; kwargs...)

# -----------------------------------------------------------------------------
# Polynomial transforms
# -----------------------------------------------------------------------------

@inline _poly_coeff_tuple(coeffs::Tuple) = coeffs
@inline _poly_coeff_tuple(coeffs::AbstractVector) = Tuple(coeffs)

@inline function _poly_sample(_, I, itp, poly, coeffs)
    coords = poly(Tuple(I), coeffs)
    return itp(coords...)
end

# CPU pure path.
function _poly_warp_params(
    coeffs,
    data::Array,
    itp,
    poly,
    inds::CartesianIndices,
)
    c = _poly_coeff_tuple(coeffs)

    return map(inds) do I
        coords = poly(Tuple(I), c)
        itp(coords...)
    end
end

# Backend-generic pure path.
function _poly_warp_params(
    coeffs,
    data::AbstractArray,
    itp,
    poly,
    inds,
)
    c = _poly_coeff_tuple(coeffs)

    return _poly_sample.(
        data,
        inds,
        Ref(itp),
        Ref(poly),
        Ref(c),
    )
end

# CPU in-place path. Keep polynomial evaluation directly in the loop: this is
# important for the observed zero-allocation EvalMultiPoly hot path.
function _poly_warp_params!(
    out::Array,
    data::Array,
    coeffs,
    itp,
    poly,
    inds::CartesianIndices,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    c = _poly_coeff_tuple(coeffs)

    @inbounds for I in inds
        coords = poly(Tuple(I), c)
        out[I] = itp(coords...)
    end
    return out
end

# Backend-generic in-place polynomial warp.
function _poly_warp_params!(
    out::AbstractArray,
    data::AbstractArray,
    coeffs,
    itp,
    poly,
    inds,
)
    axes(out) == axes(data) ||
        throw(DimensionMismatch("input and output axes must match"))

    c = _poly_coeff_tuple(coeffs)

    out .= _poly_sample.(
        data,
        inds,
        Ref(itp),
        Ref(poly),
        Ref(c),
    )
    return out
end

@inline function _poly_basis_coefficients(
    coeffs::NTuple{M,T},
) where {M,T}
    return ntuple(
        k -> ntuple(
            j -> ifelse(j == k, one(T), zero(T)),
            Val(M),
        ),
        Val(M),
    )
end

@inline function _poly_spatial_vjp(grad_itp, dcoords, ::Val{N}) where {N}
    return sum(
        ntuple(
            d -> grad_itp[d] * dcoords[d],
            Val(N),
        ),
    )
end

# CPU VJP.
function _poly_vjp(
    coeffs::NTuple{M,C},
    data::AbstractArray{T,N},
    itp,
    poly,
    Δ,
    inds::CartesianIndices,
) where {M,C,T,N}
    G = promote_type(C, T, eltype(Δ))
    g = zeros(G, M)
    grad_itp = Vector{G}(undef, N)
    basis = _poly_basis_coefficients(coeffs)

    @inbounds for I in inds
        x = Tuple(I)
        coords = poly(x, coeffs)

        Interpolations.gradient!(
            grad_itp,
            itp,
            coords...,
        )

        δ = Δ[I]

        for k in 1:M
            dcoords = poly(x, basis[k])
            g[k] += δ * _poly_spatial_vjp(
                grad_itp,
                dcoords,
                Val(N),
            )
        end
    end

    return ntuple(k -> g[k], Val(M))
end

@inline function _poly_pixel_vjp(
    I,
    δ,
    coeffs::NTuple{M,C},
    itp,
    poly,
    basis,
    ::Val{N},
    ::Type{G},
) where {M,C,N,G}
    x = Tuple(I)
    coords = poly(x, coeffs)
    grad_itp = Interpolations.gradient(itp, coords...)

    vals = ntuple(
        k -> begin
            dcoords = poly(x, basis[k])
            δ * _poly_spatial_vjp(
                grad_itp,
                dcoords,
                Val(N),
            )
        end,
        Val(M),
    )

    return ntuple(
        k -> convert(G, vals[k]),
        Val(M),
    )
end

# Backend-generic VJP. For GPU-backed arrays this is a reduction of a lazy
# broadcast, avoiding scalar host indexing and avoiding materializing the full
# image-by-coefficient Jacobian.
function _poly_vjp(
    coeffs::NTuple{M,C},
    data::AbstractArray{T,N},
    itp,
    poly,
    Δ,
    inds::AbstractArray,
) where {M,C,T,N}
    G = promote_type(C, T, eltype(Δ))
    basis = _poly_basis_coefficients(coeffs)

    bc = Base.broadcasted(
        _poly_pixel_vjp,
        vec(inds),
        vec(Δ),
        Ref(coeffs),
        Ref(itp),
        Ref(poly),
        Ref(basis),
        Ref(Val(N)),
        Ref(G),
    )

    return mapreduce(
        identity,
        _tuple_add,
        bc;
        init=ntuple(_ -> zero(G), Val(M)),
    )
end

function _poly_vjp(coeffs::AbstractVector, data, itp, poly, Δ, inds)
    return collect(
        _poly_vjp(
            Tuple(coeffs),
            data,
            itp,
            poly,
            Δ,
            inds,
        ),
    )
end

@inline _poly_parameter_tangent(::Tuple, g::Tuple) = g
@inline _poly_parameter_tangent(::AbstractVector, g::Tuple) = collect(g)
@inline _poly_parameter_tangent(::AbstractVector, g::AbstractVector) = g

function ChainRulesCore.rrule(
    ::typeof(_poly_warp_params),
    coeffs,
    data::AbstractArray,
    itp,
    poly,
    inds,
)
    y = _poly_warp_params(coeffs, data, itp, poly, inds)

    function poly_pullback(Δ_raw)
        Δ = ChainRulesCore.unthunk(Δ_raw)

        if Δ isa ChainRulesCore.AbstractZero
            return (
                ChainRulesCore.NoTangent(),
                ChainRulesCore.ZeroTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent(),
            )
        end

        c = _poly_coeff_tuple(coeffs)
        gc = _poly_vjp(c, data, itp, poly, Δ, inds)

        return (
            ChainRulesCore.NoTangent(),
            _poly_parameter_tangent(coeffs, gc),
            ChainRulesCore.NoTangent(), # source data is constant
            ChainRulesCore.NoTangent(), # interpolation object is constant
            ChainRulesCore.NoTangent(), # generated polynomial is constant
            ChainRulesCore.NoTangent(), # backend indices are constant
        )
    end

    return y, poly_pullback
end

"""
    get_function_poly(data, ::Val{order}; kwargs...)
    get_function_poly(data, order::Integer; kwargs...)

Create a polynomial coordinate warp using EvalMultiPoly.jl.

For `N`-dimensional data, the generated polynomial maps each output coordinate
to `N` interpolation coordinates. Coefficients follow the ordering defined by
`EvalMultiPoly.get_multi_poly(Val(N), Val(order))`.

# Arguments
- `data`: Sampled input data.
- `order`: Polynomial order, preferably supplied as `Val(order)` when the order
  is known statically.

# Keyword arguments
- `super_sampling=1`: Retained for API compatibility.
- `extrapolation_bc=zero(eltype(data))`: Extrapolation boundary condition.
- `interp_type=Interpolations.BSpline(Linear())`: Interpolation scheme.
- `extrapolation=nothing`: Optional explicit extrapolation setting. When given,
  it takes precedence over `extrapolation_bc`.

# Returns
A callable `warp(coeffs)` returning the transformed array. Tuple coefficients
are recommended for the performance-sensitive path.

# Automatic differentiation
A custom `ChainRulesCore.rrule` computes the vector-Jacobian product with
respect to the polynomial coefficients. It uses the linearity of polynomial
coordinates in their coefficients and the spatial gradient of the interpolation
object, avoiding materialization of the full pixel-by-coefficient Jacobian.
The source data, interpolation object, and generated polynomial evaluator are
treated as constants by the rule.

# GPU support
For a GPU-backed input, interpolation storage is adapted to the backend and the
forward transform uses backend-generic broadcast. The custom VJP uses a
backend-generic reduction.

# Examples
```julia
using EvalMultiPoly

warp = get_function_poly(data, Val(2))
coeffs = get_identity_multipoly_coeffs(Val(2), Val(2))
y = warp(coeffs)
```

See also [`get_function_poly_inplace`](@ref).
"""
function get_function_poly(
    data::AbstractArray{T,N},
    ::Val{N_order};
    super_sampling=1,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
    extrapolation=nothing,
) where {T,N,N_order}
    _ = super_sampling

    bc = isnothing(extrapolation) ?
         extrapolation_bc :
         extrapolation

    itp = _make_interpolant(data, interp_type, bc)
    poly = get_multi_poly(Val(N), Val(N_order))
    inds = _backend_indices(data)

    interpolated(coeffs) =
        _poly_warp_params(coeffs, data, itp, poly, inds)

    return interpolated
end

function get_function_poly(
    data::AbstractArray,
    order::Integer;
    kwargs...,
)
    return get_function_poly(
        data,
        Val(order);
        kwargs...,
    )
end

"""
    get_function_poly_inplace(data, ::Val{order}; kwargs...)
    get_function_poly_inplace(data, order::Integer; kwargs...)

Create the in-place polynomial warp `warp!(out, coeffs)`.

The polynomial and interpolation options are the same as for
[`get_function_poly`](@ref).

# Returns
A callable that writes into `out` and returns `out`.

# Performance
For CPU `Array`s, polynomial evaluation is intentionally kept directly inside
the hot loop. With tuple coefficients and a preallocated output, the
steady-state forward path is intended to perform zero heap allocations. For
GPU-backed arrays, the same API uses an in-place broadcast.

# Automatic differentiation
Use [`get_function_poly`](@ref) when differentiating with Zygote. This mutating
variant is intended for repeated forward evaluations.
"""
function get_function_poly_inplace(
    data::AbstractArray{T,N},
    ::Val{N_order};
    super_sampling=1,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
    extrapolation=nothing,
) where {T,N,N_order}
    _ = super_sampling

    bc = isnothing(extrapolation) ?
         extrapolation_bc :
         extrapolation

    itp = _make_interpolant(data, interp_type, bc)
    poly = get_multi_poly(Val(N), Val(N_order))
    inds = _backend_indices(data)

    function interpolated!(out, coeffs)
        return _poly_warp_params!(
            out,
            data,
            coeffs,
            itp,
            poly,
            inds,
        )
    end

    return interpolated!
end

function get_function_poly_inplace(
    data::AbstractArray,
    order::Integer;
    kwargs...,
)
    return get_function_poly_inplace(
        data,
        Val(order);
        kwargs...,
    )
end

# -----------------------------------------------------------------------------
# Mode-dispatch convenience API
# -----------------------------------------------------------------------------

"""
    get_interpolated_function(data, ::Type{AffineMode}; kwargs...)
    get_interpolated_function(data, ::Type{PolynomialMode}, order; kwargs...)
    get_interpolated_function(data; kwargs...)

Construct an interpolated transformation function using mode-based dispatch.

`AffineMode` delegates to [`get_function_affine`](@ref). `PolynomialMode`
delegates to [`get_function_poly`](@ref) and requires a polynomial `order`.
When no mode is supplied, `AffineMode` is selected and a warning is emitted.

# Keyword arguments
- `super_sampling=2`: Forwarded to the selected constructor.
- `extrapolation_bc=zero(eltype(data))`: Extrapolation boundary condition.
- `interp_type=Interpolations.BSpline(Linear())`: Interpolation scheme.

# Examples
```julia
f_affine = get_interpolated_function(data, AffineMode)
f_poly = get_interpolated_function(data, PolynomialMode, Val(2))
```
"""
function get_interpolated_function(
    data::AbstractArray{T,N},
    ::Type{AffineMode};
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    return get_function_affine(
        data;
        super_sampling=super_sampling,
        extrapolation_bc=extrapolation_bc,
        interp_type=interp_type,
    )
end

function get_interpolated_function(
    data::AbstractArray{T,N};
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    @warn "No transformation mode provided. `AffineMode` is used as default"
    return get_interpolated_function(
        data,
        AffineMode;
        super_sampling=super_sampling,
        extrapolation_bc=extrapolation_bc,
        interp_type=interp_type,
    )
end

function get_interpolated_function(
    data::AbstractArray{T,N},
    ::Type{PolynomialMode},
    order=nothing;
    super_sampling=2,
    extrapolation_bc=zero(T),
    interp_type=Interpolations.BSpline(Linear()),
) where {T,N}
    isnothing(order) && throw(ArgumentError(
        "providing the polynomial order is mandatory for `PolynomialMode`",
    ))

    return get_function_poly(
        data,
        order;
        super_sampling=super_sampling,
        extrapolation_bc=extrapolation_bc,
        interp_type=interp_type,
    )
end
