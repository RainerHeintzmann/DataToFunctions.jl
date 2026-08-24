# DataToFunctions.jl

`DataToFunctions.jl` turns sampled array data into continuously evaluated,
parameterized functions.

It is intended for applications such as image registration, fitting, and inverse
problems where measured data itself forms part of the model.

The package provides efficient coordinate transformations combined with
interpolation, including:

- affine transformations,
- polynomial transformations,
- user-defined Cartesian-coordinate transformations,
- user-defined homogeneous-coordinate transformations,
- allocation-free in-place CPU evaluation,
- automatic differentiation support for affine and polynomial parameters,
- GPU-compatible execution for supported array and interpolation backends.

## Quick start

```julia
using DataToFunctions

data = rand(Float32, 128, 128)

f = get_function_affine(data)

p = (
    0.2f0,    # x shift
   -0.1f0,    # y shift
    1.0f0,    # x scale
    1.0f0,    # y scale
    0.0f0,    # xy shear
    0.0f0,    # yx shear
    0.01f0,   # rotation [rad]
)

warped = f(p)
```

For repeated forward evaluations, use the in-place interface:

```julia
f! = get_function_affine_inplace(data)

out = similar(data)

f!(out, p)
```

The in-place CPU implementation is designed to avoid heap allocations after
compilation when using concrete tuple parameters and a preallocated output.

## Polynomial transformations

Polynomial coordinate transformations are provided through `EvalMultiPoly.jl`:

```julia
using EvalMultiPoly

f = get_function_poly(data, Val(2))

coeffs = get_identity_multipoly_coeffs(
    Val(2),
    Val(2),
)

warped = f(coeffs)
```

An allocation-minimized version is also available:

```julia
f! = get_function_poly_inplace(data, Val(2))

out = similar(data)

f!(out, coeffs)
```

## Automatic differentiation

The allocating affine and polynomial interfaces support differentiation with
respect to their transformation parameters.

For example:

```julia
using Zygote

target = f(p)

loss(q) = sum(abs2, f(q) .- target)

g = Zygote.gradient(loss, p)[1]
```

The affine and polynomial implementations provide custom reverse-mode rules
using `ChainRulesCore`, avoiding construction of the full image-to-parameter
Jacobian.

This makes the resulting gradients suitable for optimization packages such as
`Optim.jl`.

## GPU arrays

The transformation implementation is backend-generic.

CPU arrays use specialized CPU paths, while other `AbstractArray` backends use
broadcast and reduction operations. Supported interpolation objects are adapted
to the target array backend using `Adapt.jl`.

For example, with CUDA.jl:

```julia
using CUDA
using DataToFunctions

CUDA.allowscalar(false)

data_gpu = CuArray(data)

f_gpu = get_function_affine(data_gpu)

warped_gpu = f_gpu(p)
```

GPU compatibility depends on the selected interpolation method and array
backend. `BSpline(Linear())`, the default interpolation mode, is the primary
supported GPU path.


See the [API Reference](@ref) for the full list of exported functions.
