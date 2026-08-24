using DataToFunctions
using EvalMultiPoly
using Optim
using NLSolversBase
using Zygote

using View5D
using TestImages

# -----------------------------------------------------------------------------
# Example data
# -----------------------------------------------------------------------------

# n = 64
# data = [
#     sin(0.071 * i) + cos(0.053 * j) + 0.001 * i * j
#     for i in 1:n, j in 1:n
# ]
data = Float32.(testimage("resolution_test_512.tif"));
n = size(data, 1)
# -----------------------------------------------------------------------------
# 1. Affine transform
# -----------------------------------------------------------------------------

f_affine = get_function_affine(data; extrapolation_bc=0.0)
f_affine! = get_function_affine_inplace(data; extrapolation_bc=0.0)

# (shift_x, shift_y, scale_x, scale_y, shear_xy, shear_yx, rotation)
p_affine = (0.25, -0.15, 1.01, 0.99, 0.002, -0.003, 0.01)

warped_affine = f_affine(p_affine)

out_affine = similar(data)
f_affine!(out_affine, p_affine)
@assert warped_affine ≈ out_affine

# Warm up before measuring allocations.
f_affine!(out_affine, p_affine)
println("Affine in-place allocated bytes: ", @allocated(f_affine!(out_affine, p_affine)))

# -----------------------------------------------------------------------------
# 2. Fast affine gradient through the custom ChainRules rule
# -----------------------------------------------------------------------------

p_affine_vec = collect(p_affine)
interior = 4:n-3

affine_loss(p) = begin
    y = f_affine(p)
    sum(abs2, y[interior, interior] .- data[interior, interior])
end

g_affine = Zygote.gradient(affine_loss, p_affine_vec)[1]
println("Affine gradient: ", g_affine)

# -----------------------------------------------------------------------------
# 3. Affine registration with Optim.LBFGS
# -----------------------------------------------------------------------------

p_true = [0.18, -0.12, 1.006, 0.994, 0.0015, -0.0110, 0.007]
target_affine = f_affine(p_true)[interior, interior]

registration_loss(p) = begin
    y = f_affine(p)
    sum(abs2, y[interior, interior] .- target_affine)
end

function registration_fg!(F, G, p)
    if G === nothing
        return registration_loss(p)
    end

    value, back = Zygote.pullback(registration_loss, p)
    gp = back(one(value))[1]
    copyto!(G, gp)
    return F === nothing ? nothing : value
end

p0 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]

result_affine = Optim.optimize(
    NLSolversBase.only_fg!(registration_fg!),
    p0,
    Optim.LBFGS(),
    Optim.Options(iterations=100, g_tol=1e-8, show_trace=true),
)

println("Affine minimum:   ", Optim.minimum(result_affine))
println("Affine estimate:  ", Optim.minimizer(result_affine))
println("Affine true p:    ", p_true)

@vt data f_affine(p_true) f_affine(Optim.minimizer(result_affine)) 
# -----------------------------------------------------------------------------
# 4. Polynomial transform
# -----------------------------------------------------------------------------

order = Val(2)
f_poly = get_function_poly(data, order; extrapolation_bc=0.0)
f_poly! = get_function_poly_inplace(data, order; extrapolation_bc=0.0)

c_identity = map(
    Float64,
    EvalMultiPoly.get_identity_multipoly_coeffs(Val(2), order),
)

M = length(c_identity)
c_poly = ntuple(k -> c_identity[k] + 1e-6 * k, M)

warped_poly = f_poly(c_poly)
out_poly = similar(data)
f_poly!(out_poly, c_poly)
@assert warped_poly ≈ out_poly

f_poly!(out_poly, c_poly) # warmup
println("Polynomial in-place allocated bytes: ", @allocated(f_poly!(out_poly, c_poly)))

# -----------------------------------------------------------------------------
# 5. Polynomial gradient
# -----------------------------------------------------------------------------

c_poly_vec = collect(c_poly)

poly_loss(c) = begin
    y = f_poly(c)
    sum(abs2, y[interior, interior] .- data[interior, interior])
end

g_poly = Zygote.gradient(poly_loss, c_poly_vec)[1]
println("Polynomial gradient length: ", length(g_poly))
println("Polynomial gradient: ", g_poly)

# -----------------------------------------------------------------------------
# 6. Polynomial registration with Optim.LBFGS
# -----------------------------------------------------------------------------

# Generate a target with a small polynomial deformation and recover it locally.
c_true = collect(c_identity)
for k in eachindex(c_true)
    c_true[k] += 1e-6 * k
end

target_poly = f_poly(c_true)[interior, interior]

poly_registration_loss(c) = begin
    y = f_poly(c)
    sum(abs2, y[interior, interior] .- target_poly)
end

function poly_registration_fg!(F, G, c)
    if G === nothing
        return poly_registration_loss(c)
    end

    value, back = Zygote.pullback(poly_registration_loss, c)
    gc = back(one(value))[1]
    copyto!(G, gc)
    return F === nothing ? nothing : value
end

c0 = collect(c_identity)
result_poly = Optim.optimize(
    NLSolversBase.only_fg!(poly_registration_fg!),
    c0,
    Optim.LBFGS(),
    Optim.Options(iterations=100, g_tol=1e-6, show_trace=true),
)

println("Polynomial minimum: ", Optim.minimum(result_poly))
println("Polynomial estimate: ", Optim.minimizer(result_poly))

@vt data f_poly(c_true) f_poly(Optim.minimizer(result_poly)) 
