using BenchmarkTools
using DataToFunctions
using EvalMultiPoly
using ForwardDiff
using Zygote

function benchmark_image(::Type{T}=Float64, n::Int=128) where {T}
    return [
        sin(T(0.071) * i) + cos(T(0.053) * j) + T(0.001) * i * j
        for i in 1:n, j in 1:n
    ]
end

n = 128
data = benchmark_image(Float64, n)
interior = 4:n-3

println("="^72)
println("DataToFunctions transform benchmark")
println("image size: $(n)×$(n), Float64")
println("="^72)

# -----------------------------------------------------------------------------
# Affine
# -----------------------------------------------------------------------------

println("\nAFFINE")
println("-"^72)

f_aff = get_function_affine(data; extrapolation_bc=0.0)
f_aff! = get_function_affine_inplace(data; extrapolation_bc=0.0)
p_aff = (0.25, -0.15, 1.01, 0.99, 0.002, -0.003, 0.01)
p_aff_vec = collect(p_aff)
out_aff = similar(data)

# warmup
f_aff(p_aff)
f_aff!(out_aff, p_aff)

println("Allocating value:")
@btime $f_aff($p_aff)
# 371.700 μs (3 allocations: 128.08 KiB)

println("In-place value:")
@btime $f_aff!($out_aff, $p_aff)
#  148.600 μs (0 allocations: 0 bytes)

println("In-place allocated bytes:")
@show @allocated f_aff!(out_aff, p_aff)
# 0

affine_loss(p) = begin
    y = f_aff(p)
    sum(abs2, y[interior, interior] .- data[interior, interior])
end

# Zygote uses the package rrule. ForwardDiff is an independent comparison.
Zygote.gradient(affine_loss, p_aff_vec)
ForwardDiff.gradient(affine_loss, p_aff_vec)

println("Zygote gradient (custom rrule):")
@btime Zygote.gradient($affine_loss, $p_aff_vec)[1]
#   637.400 μs (110 allocations: 845.17 KiB)

println("ForwardDiff gradient:")
@btime ForwardDiff.gradient($affine_loss, $p_aff_vec)
#   1.502 ms (22 allocations: 2.93 MiB)

println("Zygote gradient allocated bytes:")
@show @allocated Zygote.gradient(affine_loss, p_aff_vec)
# 865537

println("ForwardDiff gradient allocated bytes:")
@show @allocated ForwardDiff.gradient(affine_loss, p_aff_vec)
# 3074556

# -----------------------------------------------------------------------------
# Polynomial
# -----------------------------------------------------------------------------

println("\nPOLYNOMIAL (2-D, order 2)")
println("-"^72)

f_poly = get_function_poly(data, Val(2); extrapolation_bc=0.0)
f_poly! = get_function_poly_inplace(data, Val(2); extrapolation_bc=0.0)

c0 = map(Float64, EvalMultiPoly.get_identity_multipoly_coeffs(Val(2), Val(2)))
M = length(c0)
c_poly = ntuple(k -> c0[k] + 1e-6 * k, M)
c_poly_vec = collect(c_poly)
out_poly = similar(data)

# warmup
f_poly(c_poly)
f_poly!(out_poly, c_poly)

println("Allocating value:")
@btime $f_poly($c_poly)
#   275.500 μs (3 allocations: 128.08 KiB)

println("In-place value:")
@btime $f_poly!($out_poly, $c_poly)
#   275.200 μs (0 allocations: 0 bytes)

println("In-place allocated bytes:")
@show @allocated f_poly!(out_poly, c_poly)
# 0

poly_loss(c) = begin
    y = f_poly(c)
    sum(abs2, y[interior, interior] .- data[interior, interior])
end

Zygote.gradient(poly_loss, c_poly_vec)
ForwardDiff.gradient(poly_loss, c_poly_vec)

println("Zygote gradient (custom rrule):")
@btime Zygote.gradient($poly_loss, $c_poly_vec)[1]
#   3.091 ms (123 allocations: 848.67 KiB)

println("ForwardDiff gradient:")
@btime ForwardDiff.gradient($poly_loss, $c_poly_vec)
#   2.159 ms (40 allocations: 4.70 MiB)

println("Zygote gradient allocated bytes:")
@show @allocated Zygote.gradient(poly_loss, c_poly_vec)
# 869137

println("ForwardDiff gradient allocated bytes:")
@show @allocated ForwardDiff.gradient(poly_loss, c_poly_vec)
# 4927884
# -----------------------------------------------------------------------------
# Numerical gradient agreement
# -----------------------------------------------------------------------------

println("\nGRADIENT AGREEMENT")
println("-"^72)

g_aff_zyg = Zygote.gradient(affine_loss, p_aff_vec)[1]
g_aff_fwd = ForwardDiff.gradient(affine_loss, p_aff_vec)
println("Affine max |Zygote - ForwardDiff|: ", maximum(abs, g_aff_zyg .- g_aff_fwd))
# Affine max |Zygote - ForwardDiff|: 8.640199666842818e-12

g_poly_zyg = Zygote.gradient(poly_loss, c_poly_vec)[1]
g_poly_fwd = ForwardDiff.gradient(poly_loss, c_poly_vec)
println("Polynomial max |Zygote - ForwardDiff|: ", maximum(abs, g_poly_zyg .- g_poly_fwd))
# Polynomial max |Zygote - ForwardDiff|: 1.7462298274040222e-9