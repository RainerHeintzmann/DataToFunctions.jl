# Tutorial

This short tutorial shows the main workflow of `DataToFunctions.jl`: create a
parameterized transformation from sampled data, evaluate it efficiently, and
differentiate it for optimization.

## Affine transformation

Start from a 2-D array:

```julia
using DataToFunctions

data = rand(Float32, 128, 128)
```

Create an affine transformation function:

```julia
f = get_function_affine(data)
```

For 2-D data, the transformation parameters are

```julia
p = (
    0.2f0,    # shift x
   -0.1f0,    # shift y
    1.0f0,    # scale x
    1.0f0,    # scale y
    0.0f0,    # shear xy
    0.0f0,    # shear yx
    0.01f0,   # rotation [rad]
)
```

Evaluate the transformed data with:

```julia
warped = f(p)
```

For repeated forward evaluations, use the in-place version:

```julia
f! = get_function_affine_inplace(data)

out = similar(data)

f!(out, p)
```

The in-place CPU path is intended for preallocated, allocation-free repeated
evaluation after compilation.

## Polynomial transformation

Polynomial coordinate transformations are created by specifying the polynomial
order:

```julia
using EvalMultiPoly

f_poly = get_function_poly(data, Val(2))

coeffs = get_identity_multipoly_coeffs(
    Val(2),
    Val(2),
)

warped_poly = f_poly(coeffs)
```

For repeated evaluation:

```julia
f_poly! = get_function_poly_inplace(data, Val(2))

out_poly = similar(data)

f_poly!(out_poly, coeffs)
```

## Automatic differentiation

The allocating affine and polynomial interfaces can be differentiated with
respect to their transformation parameters.

For example:

```julia
using Zygote

target = f((
    0.1f0,
   -0.05f0,
    1.01f0,
    0.99f0,
    0.0f0,
    0.0f0,
    0.005f0,
))

loss(q) = sum(abs2, f(q) .- target)

g = Zygote.gradient(loss, p)[1]
```

The package provides custom reverse-mode rules for the affine and polynomial
parameterizations, making these gradients suitable for iterative optimization.

## Optimization with Optim.jl

A gradient from Zygote can be used directly with `Optim.LBFGS`:

```julia
using Optim

p0 = Float32[
    0.0,
    0.0,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
]

loss_vec(q) = sum(abs2, f(q) .- target)

function g!(G, q)
    G .= Zygote.gradient(loss_vec, q)[1]
    return G
end

result = optimize(
    loss_vec,
    g!,
    p0,
    LBFGS(),
)
```

The fitted parameters are available with:

```julia
Optim.minimizer(result)
```

## GPU arrays

The same high-level API can be used with GPU-backed arrays when the selected
interpolation mode is supported by the backend.

For CUDA:

```julia
using CUDA

CUDA.allowscalar(false)

data_gpu = CuArray(data)

f_gpu = get_function_affine(data_gpu)

warped_gpu = f_gpu(p)
```

The implementation keeps CUDA optional: `DataToFunctions.jl` does not require
CUDA.jl unless GPU execution is used.

For the complete list of public functions, see the [API Reference](@ref).
