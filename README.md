# DataToFunctions.jl

[![CI](https://github.com/RainerHeintzmann/DataToFunctions.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/RainerHeintzmann/DataToFunctions.jl/actions/workflows/ci.yml)
[![Development Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://rainerheintzmann.github.io/DataToFunctions.jl/dev/)

`DataToFunctions.jl` represents data as continuously evaluated, parameterized functions.

The package is intended for fitting, image registration, and inverse problems
where measured data itself forms part of the model. It provides efficient
interpolation-based coordinate transformations with support for affine,
polynomial, and user-defined transformations.

## Features

- Affine transformations in arbitrary dimensions using homogeneous coordinates;
- A convenient seven-parameter affine model for 2-D data;
- Polynomial coordinate transformations based on
  [`EvalMultiPoly.jl`](https://github.com/RainerHeintzmann/EvalMultiPoly.jl);
- User-defined Cartesian and homogeneous-coordinate transformations;
- Allocation-free in-place CPU evaluation for performance-critical workflows;
- Automatic differentiation of affine and polynomial transformation parameters;
- Custom `ChainRulesCore` reverse rules for efficient Zygote gradients;
- Compatibility with gradient-based optimization such as `Optim.jl` with
  `LBFGS`;
- Backend-generic execution, including CUDA arrays for supported interpolation
  methods.

## Quick example

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

For repeated forward evaluations, an in-place version is available:

```julia
f! = get_function_affine_inplace(data)

out = similar(data)
f!(out, p)
```

On CPU, the in-place transformation path is designed to run without heap
allocations after compilation when using concrete tuple parameters and a
preallocated output array.

## Automatic differentiation

The affine and polynomial transformation interfaces provide custom reverse-mode
rules for differentiation with respect to transformation parameters.

For example:

```julia
using Zygote

target = f(p)

loss(q) = sum(abs2, f(q) .- target)

gradient = Zygote.gradient(loss, p)[1]
```

These gradients can be used directly in optimization workflows.

## Documentation

- [Development documentation](https://rainerheintzmann.github.io/DataToFunctions.jl/dev/)

The development documentation follows the `develop` branch.

<!-- Stable documentation can be added after the first tagged release at:

- `https://rainerheintzmann.github.io/DataToFunctions.jl/stable/` -->
