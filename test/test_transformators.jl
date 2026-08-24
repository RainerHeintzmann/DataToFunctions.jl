using Test
using DataToFunctions
using Interpolations
using EvalMultiPoly
using Zygote
using ForwardDiff

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

smooth_image(::Type{T}, n=48) where {T} = T[
    sin(T(0.071) * i) +
    cos(T(0.053) * j) +
    T(0.001) * i * j
    for i in 1:n, j in 1:n
]

function tuple_isapprox(a::Tuple, b::Tuple; rtol=1e-10, atol=1e-10)
    length(a) == length(b) || return false
    return all(isapprox.(a, b; rtol=rtol, atol=atol))
end

# -----------------------------------------------------------------------------
# Tuple primitives and general transforms
# -----------------------------------------------------------------------------

@testset "tuple primitives" begin
    h = DataToFunctions.add_dim((2.0, 3.0))
    @test h == (2.0, 3.0, 1.0)

    # Heterogeneous tuple regression: the final coordinate may remain Int.
    @test tuple_isapprox(
        DataToFunctions.red_dim((4.0, 6.0, 2)),
        (2.0, 3.0),
    )

    A = (
        (1.0, 2.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    @test DataToFunctions.mat_mul((2.0, 3.0, 1.0), A) == (8.0, 3.0, 1.0)
end

@testset "tuple and homogeneous transforms" begin
    data = smooth_image(Float64, 24)

    shift_tuple(c, p) = (c[1] + p[1], c[2] + p[2])

    f = get_function_tuple(data, shift_tuple)
    f! = get_function_tuple_inplace(data, shift_tuple)

    p = (0.2, -0.15)
    y = f(p)
    out = similar(data)

    f!(out, p)
    @test y ≈ out

    # Warm first; only the steady-state call is measured.
    f!(out, p)
    @test @allocated(f!(out, p)) == 0

    homogeneous_shift(h, p) = (
        h[1] + p[1] * h[3],
        h[2] + p[2] * h[3],
        h[3], # deliberately leaves this as Int for integer input coordinates
    )

    fh = get_function_homogen(data, homogeneous_shift)
    fh! = get_function_homogen_inplace(data, homogeneous_shift)

    yh = fh(p)
    fh!(out, p)
    @test yh ≈ out

    fh!(out, p)
    @test @allocated(fh!(out, p)) == 0
end

@testset "arbitrary-dimensional affine matrix" begin
    data = reshape(
        collect(Float64, 1:(8 * 7 * 6)),
        8, 7, 6,
    )

    f = get_function_affine(data)
    f! = get_function_affine_inplace(data)

    A4 = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    @test f(A4) ≈ data

    out = similar(data)
    f!(out, A4)
    @test out ≈ data

    f!(out, A4)
    @test @allocated(f!(out, A4)) == 0
end

# -----------------------------------------------------------------------------
# Affine transform: forward, allocation behavior, AD, optimization
# -----------------------------------------------------------------------------

@testset "affine forward and allocation behavior" begin
    data = smooth_image(Float64, 48)

    f = get_function_affine(data)
    f! = get_function_affine_inplace(data)

    identity_p = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    @test f(identity_p) ≈ data

    p = (0.25, -0.15, 1.01, 0.99, 0.002, -0.003, 0.01)

    y = f(p)
    out = similar(data)
    f!(out, p)

    @test y ≈ out

    # Strict zero-allocation CPU hot-path check.
    f!(out, p)
    @test @allocated(f!(out, p)) == 0
end

@testset "affine analytical coordinates" begin
    data = smooth_image(Float64, 32)

    p = (0.25, -0.15, 1.01, 0.99, 0.002, -0.003, 0.01)

    x_cen = size(data, 1) ÷ 2 + 1
    y_cen = size(data, 2) ÷ 2 + 1

    A = DataToFunctions._affine_matrix_2d(p, x_cen, y_cen)

    for (x, y) in ((1, 1), (7, 13), (16, 16), (25, 20))
        matrix_result = DataToFunctions.red_dim(DataToFunctions.mat_mul(DataToFunctions.add_dim((x, y)), A))

        shift_x, shift_y,
        scale_x, scale_y,
        shear_xy, shear_yx,
        θ = p

        a = x - x_cen - shift_x
        b = y - y_cen - shift_y

        u = a + shear_xy * b
        v = shear_yx * a + b

        c = cos(θ)
        s = sin(θ)

        r1 = c * u - s * v
        r2 = s * u + c * v

        analytical_result = (
            x_cen + r1 / scale_x,
            y_cen + r2 / scale_y,
        )

        @test all(
            isapprox.(
                matrix_result,
                analytical_result;
                rtol=1e-12,
                atol=1e-12,
            ),
        )
    end
end

@testset "affine Zygote rrule agrees with ForwardDiff" begin
    data = smooth_image(Float64, 36)
    f = get_function_affine(data)

    p = (0.25, -0.15, 1.01, 0.99, 0.002, -0.003, 0.01)

    target = f((
        0.18, -0.11, 1.008, 0.994, 0.0015, -0.002, 0.007,
    ))

    loss(q) = sum(abs2, f(q) .- target)

    g_zygote = Zygote.gradient(loss, p)[1]

    pvec = collect(p)
    loss_fd(q) = loss(Tuple(q))
    g_forward = ForwardDiff.gradient(loss_fd, pvec)

    @test all(isfinite, g_zygote)
    @test collect(g_zygote) ≈ g_forward rtol=2e-6 atol=2e-7
end

# -----------------------------------------------------------------------------
# Polynomial transform: forward, allocation behavior, AD
# -----------------------------------------------------------------------------

@testset "polynomial forward and allocation behavior" begin
    data = smooth_image(Float64, 40)

    order = Val(2)
    f = get_function_poly(data, order)
    f! = get_function_poly_inplace(data, order)

    c0 = get_identity_multipoly_coeffs(Val(2), Val(2))

    @test f(c0) ≈ data

    c = ntuple(
        k -> c0[k] + 1e-6 * k,
        length(c0),
    )

    y = f(c)
    out = similar(data)
    f!(out, c)

    @test y ≈ out

    f!(out, c)
    @test @allocated(f!(out, c)) == 0
end

@testset "polynomial Zygote rrule agrees with ForwardDiff" begin
    data = smooth_image(Float64, 28)
    f = get_function_poly(data, Val(2))

    c0 = get_identity_multipoly_coeffs(Val(2), Val(2))
    c = ntuple(k -> c0[k] + 5e-7 * k, length(c0))

    target = f(
        ntuple(k -> c0[k] - 3e-7 * k, length(c0)),
    )

    loss(q) = sum(abs2, f(q) .- target)

    g_zygote = Zygote.gradient(loss, c)[1]

    cvec = collect(c)
    loss_fd(q) = loss(Tuple(q))
    g_forward = ForwardDiff.gradient(loss_fd, cvec)

    @test all(isfinite, g_zygote)
    @test collect(g_zygote) ≈ g_forward rtol=2e-6 atol=2e-7
end

# -----------------------------------------------------------------------------
# Optional CUDA tests
#
# CUDA is intentionally not a dependency of DataToFunctions itself.
# These tests run only when CUDA is installed and a functional device exists.
# -----------------------------------------------------------------------------

if Base.find_package("CUDA") !== nothing
    @eval using CUDA

    if CUDA.functional()
        @testset "CUDA affine and polynomial forward paths" begin
            CUDA.allowscalar(false)

            data_cpu = smooth_image(Float32, 48)
            data_gpu = CuArray(data_cpu)

            # Regression test for GPU interpolant storage:
            # constructing directly from a CuArray can leave CPU coefficients
            # inside Interpolations.jl. DataToFunctions must construct on CPU
            # and Adapt the completed interpolation object back to the backend.
            itp_gpu = DataToFunctions._make_interpolant(
                data_gpu,
                Interpolations.BSpline(Linear()),
                0f0,
            )

            coords_gpu = CuArray(Float32[1.25, 2.5, 3.75])
            vals_gpu = itp_gpu.(coords_gpu, coords_gpu)
            @test vals_gpu isa CuArray
            @test all(isfinite, Array(vals_gpu))

            p = (
                0.25f0, -0.15f0,
                1.01f0, 0.99f0,
                0.002f0, -0.003f0,
                0.01f0,
            )

            f_cpu = get_function_affine(data_cpu)
            f_gpu = get_function_affine(data_gpu)
            f_gpu! = get_function_affine_inplace(data_gpu)

            y_cpu = f_cpu(p)
            y_gpu = f_gpu(p)

            @test y_gpu isa CuArray
            @test Array(y_gpu) ≈ y_cpu rtol=2e-5 atol=2e-5

            out_gpu = similar(data_gpu)
            f_gpu!(out_gpu, p)
            @test Array(out_gpu) ≈ y_cpu rtol=2e-5 atol=2e-5

            c0 = get_identity_multipoly_coeffs(Val(2), Val(2))
            c = ntuple(k -> Float32(c0[k] + 1e-6 * k), length(c0))

            pf_cpu = get_function_poly(data_cpu, Val(2))
            pf_gpu = get_function_poly(data_gpu, Val(2))
            pf_gpu! = get_function_poly_inplace(data_gpu, Val(2))

            py_cpu = pf_cpu(c)
            py_gpu = pf_gpu(c)

            @test py_gpu isa CuArray
            @test Array(py_gpu) ≈ py_cpu rtol=2e-5 atol=2e-5

            pout_gpu = similar(data_gpu)
            pf_gpu!(pout_gpu, c)
            @test Array(pout_gpu) ≈ py_cpu rtol=2e-5 atol=2e-5
        end

        @testset "CUDA custom VJPs" begin
            CUDA.allowscalar(false)

            data_cpu = smooth_image(Float32, 32)
            data_gpu = CuArray(data_cpu)

            affine_gpu = get_function_affine(data_gpu)

            p = (
                0.15f0, -0.10f0,
                1.006f0, 0.996f0,
                0.001f0, -0.0015f0,
                0.005f0,
            )

            target_affine = affine_gpu((
                0.10f0, -0.06f0,
                1.003f0, 0.998f0,
                0.0005f0, -0.001f0,
                0.003f0,
            ))

            affine_loss_gpu(q) =
                sum(abs2, affine_gpu(q) .- target_affine)

            g_affine_gpu = Zygote.gradient(
                affine_loss_gpu,
                p,
            )[1]

            @test all(isfinite, g_affine_gpu)

            affine_cpu = get_function_affine(data_cpu)
            target_affine_cpu = Array(target_affine)
            affine_loss_cpu(q) =
                sum(abs2, affine_cpu(q) .- target_affine_cpu)
            g_affine_cpu = Zygote.gradient(
                affine_loss_cpu,
                p,
            )[1]

            @test collect(g_affine_gpu) ≈ collect(g_affine_cpu) rtol=2e-3 atol=2e-3

            poly_gpu = get_function_poly(data_gpu, Val(2))
            c0 = get_identity_multipoly_coeffs(Val(2), Val(2))
            c = ntuple(
                k -> Float32(c0[k] + 5e-7 * k),
                length(c0),
            )

            target_poly = poly_gpu(
                ntuple(
                    k -> Float32(c0[k] - 3e-7 * k),
                    length(c0),
                ),
            )

            poly_loss_gpu(q) =
                sum(abs2, poly_gpu(q) .- target_poly)

            g_poly_gpu = Zygote.gradient(
                poly_loss_gpu,
                c,
            )[1]

            @test all(isfinite, g_poly_gpu)

            poly_cpu = get_function_poly(data_cpu, Val(2))
            target_poly_cpu = Array(target_poly)
            poly_loss_cpu(q) =
                sum(abs2, poly_cpu(q) .- target_poly_cpu)
            g_poly_cpu = Zygote.gradient(
                poly_loss_cpu,
                c,
            )[1]

            @test collect(g_poly_gpu) ≈ collect(g_poly_cpu) rtol=3e-3 atol=3e-3
        end
    else
        @info "CUDA.jl is installed, but no functional CUDA device is available; CUDA tests skipped."
    end
else
    @info "CUDA.jl is not installed; CUDA tests skipped."
end
