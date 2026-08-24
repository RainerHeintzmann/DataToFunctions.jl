# API Reference

This page lists the public transformation interface provided by
`DataToFunctions.jl`.

## Affine transformations

```@docs
DataToFunctions.get_function_affine
DataToFunctions.get_function_affine_inplace
```

## Polynomial transformations

```@docs
DataToFunctions.get_function_poly
DataToFunctions.get_function_poly_inplace
```

## User-defined transformations

```@docs
DataToFunctions.get_function_tuple
DataToFunctions.get_function_tuple_inplace
DataToFunctions.get_function_homogen
DataToFunctions.get_function_homogen_inplace
```

## Mode-based interface

```@docs
DataToFunctions.get_interpolated_function
```

## Transformation modes

If `AffineMode` and `PolynomialMode` have docstrings, they can be included here:

```@docs
DataToFunctions.AffineMode
DataToFunctions.PolynomialMode
```
