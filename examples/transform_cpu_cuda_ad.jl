using CUDA
using DataToFunctions
using Interpolations
using EvalMultiPoly
using BenchmarkTools
using Zygote
using ForwardDiff
using Optim

# -----------------------------------------------------------------------------
# Synthetic image
# -----------------------------------------------------------------------------

N = 256

data = Float32[
    sin(0.031f0 * i) +
    cos(0.027f0 * j) +
    0.0002f0 * i * j
    for i in 1:N, j in 1:N
]

# -----------------------------------------------------------------------------
# 1. Affine forward transform
# -----------------------------------------------------------------------------

affine = get_function_affine(data)
affine! = get_function_affine_inplace(data)

p = (
     0.25f0,   # shift x
    -0.15f0,   # shift y
     1.01f0,   # scale x
     0.99f0,   # scale y
     0.002f0,  # shear xy
    -0.003f0,  # shear yx
     0.01f0,   # rotation [rad]
)

warped = affine(p)

out = similar(data)
affine!(out, p)

@assert warped ≈ out

# Warm up before measuring allocations.
affine!(out, p)

println("Affine in-place CPU allocations:")
@show @allocated affine!(out, p)

println("\nAffine forward benchmark:")
@btime $affine!($out, $p)

# -----------------------------------------------------------------------------
# 2. Affine gradient: custom Zygote rrule vs ForwardDiff
# -----------------------------------------------------------------------------

p_target = (
     0.18f0,
    -0.11f0,
     1.008f0,
     0.994f0,
     0.0015f0,
    -0.002f0,
     0.007f0,
)

target = affine(p_target)

affine_loss(q) =
    sum(abs2, affine(q) .- target)

# Warm up.
g_zygote = Zygote.gradient(affine_loss, p)[1]

println("\nAffine Zygote gradient:")
@show g_zygote

println("\nAffine Zygote gradient benchmark:")
@btime Zygote.gradient($affine_loss, $p)[1]

pvec = collect(p)
affine_loss_fd(q) = affine_loss(Tuple(q))

g_forward = ForwardDiff.gradient(
    affine_loss_fd,
    pvec,
)

println("\nForwardDiff reference:")
@show g_forward
@show maximum(abs.(collect(g_zygote) .- g_forward))

# -----------------------------------------------------------------------------
# 3. Optimize affine parameters with Optim.LBFGS
# -----------------------------------------------------------------------------

# Optim works with vectors. DataToFunctions preserves a vector gradient when
# the affine parameters are supplied as a vector.

p0 = Float32[
    0.0,
    0.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
]

loss_for_optim(q) =
    sum(abs2, affine(q) .- target)

function affine_gradient!(G, q)
    G .= Zygote.gradient(loss_for_optim, q)[1]
    return G
end

result = Optim.optimize(
    loss_for_optim,
    affine_gradient!,
    p0,
    Optim.LBFGS(),
    Optim.Options(
        iterations=50,
        show_trace=true,
    ),
)

println("\nLBFGS affine fit:")
@show Optim.minimum(result)
@show Optim.minimizer(result)
@show collect(p_target)

# -----------------------------------------------------------------------------
# 4. Polynomial transform
# -----------------------------------------------------------------------------

order = Val(2)

poly = get_function_poly(
    data,
    order,
)

poly! = get_function_poly_inplace(
    data,
    order,
)

c_identity = get_identity_multipoly_coeffs(
    Val(2),
    Val(2),
)

# Very small deformation away from the identity.
coeffs = ntuple(
    k -> Float32(c_identity[k] + 1e-6 * k),
    length(c_identity),
)

poly_warped = poly(coeffs)

poly_out = similar(data)
poly!(poly_out, coeffs)

@assert poly_warped ≈ poly_out

poly!(poly_out, coeffs)

println("\nPolynomial in-place CPU allocations:")
@show @allocated poly!(poly_out, coeffs)

println("\nPolynomial forward benchmark:")
@btime $poly!($poly_out, $coeffs)

poly_target_coeffs = ntuple(
    k -> Float32(c_identity[k] - 7e-7 * k),
    length(c_identity),
)

poly_target = poly(poly_target_coeffs)

poly_loss(c) =
    sum(abs2, poly(c) .- poly_target)

g_poly = Zygote.gradient(
    poly_loss,
    coeffs,
)[1]

println("\nPolynomial Zygote gradient:")
@show g_poly

println("\nPolynomial Zygote gradient benchmark:")
@btime Zygote.gradient($poly_loss, $coeffs)[1]

coeffs_vec = collect(coeffs)

poly_loss_fd(c) =
    poly_loss(Tuple(c))

g_poly_forward = ForwardDiff.gradient(
    poly_loss_fd,
    coeffs_vec,
)

println("\nPolynomial ForwardDiff agreement:")
@show maximum(
    abs.(
        collect(g_poly) .-
        g_poly_forward
    ),
)

# -----------------------------------------------------------------------------
# 5. Polynomial optimization with Optim.LBFGS
# -----------------------------------------------------------------------------

# Optimize a Vector so Optim can update it in place.
poly_c0 = collect(c_identity)

function poly_loss_optim(c)
    return sum(
        abs2,
        poly(c) .- poly_target,
    )
end

function poly_gradient!(G, c)
    G .= Zygote.gradient(
        poly_loss_optim,
        c,
    )[1]
    return G
end

poly_result = Optim.optimize(
    poly_loss_optim,
    poly_gradient!,
    poly_c0,
    Optim.LBFGS(),
    Optim.Options(
        iterations=30,
        show_trace=true,
    ),
)

println("\nLBFGS polynomial fit:")
@show Optim.minimum(poly_result)
@show Optim.minimizer(poly_result)

# -----------------------------------------------------------------------------
# 6. Optional CUDA execution
#
# DataToFunctions itself does not import CUDA. If `data` is a CuArray, the
# transform is expressed with GPU-compatible broadcast/reduction abstractions.
#
# The current GPU path is intended first for BSpline(Linear()), which is the
# default and does not require a spline-prefilter solve.
# -----------------------------------------------------------------------------


CUDA.allowscalar(false)

data_gpu = CuArray(data)

affine_gpu = get_function_affine(data_gpu)
affine_gpu! = get_function_affine_inplace(data_gpu)

y_gpu = affine_gpu(p)
out_gpu = similar(data_gpu)
affine_gpu!(out_gpu, p)

@assert Array(y_gpu) ≈ warped rtol=2e-4 atol=2e-4
@assert Array(out_gpu) ≈ warped rtol=2e-4 atol=2e-4

CUDA.synchronize()

println("\nCUDA affine forward benchmark:")
@btime begin
    $affine_gpu!($out_gpu, $p)
    CUDA.synchronize()
end

# Zygote reaches the custom VJP. The VJP is expressed as a reduction
# over backend-resident Cartesian indices and therefore does not use
# scalar host indexing.
target_gpu = affine_gpu(p_target)

affine_loss_gpu(q) =
    sum(abs2, affine_gpu(q) .- target_gpu)

g_gpu = Zygote.gradient(
    affine_loss_gpu,
    p,
)[1]

println("\nCUDA affine gradient:")
@show g_gpu

poly_gpu = get_function_poly(
    data_gpu,
    order,
)

poly_gpu! = get_function_poly_inplace(
    data_gpu,
    order,
)

poly_out_gpu = similar(data_gpu)
poly_gpu!(poly_out_gpu, coeffs)

@assert Array(poly_out_gpu) ≈ poly_warped rtol=2e-4 atol=2e-4

poly_target_gpu = poly_gpu(poly_target_coeffs)

poly_loss_gpu(c) =
    sum(abs2, poly_gpu(c) .- poly_target_gpu)

g_poly_gpu = Zygote.gradient(
    poly_loss_gpu,
    coeffs,
)[1]

println("\nCUDA polynomial gradient:")
@show g_poly_gpu
