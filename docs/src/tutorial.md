```@meta
EditURL = "tutorial.jl"
```

Tutorial for the DataToFunctions.jl package

````@example tutorial
import Pkg
Pkg.add("SyntheticObjects")
Pkg.add("StaticArrays")
Pkg.add("Plots")
Pkg.add("Random")
````

Load the packages

````@example tutorial
using DataToFunctions
using SyntheticObjects
using StaticArrays
using Random
using Plots
````

We set the Random.seed for reproducibility

````@example tutorial
Random.seed!(14)
````

Define the data

````@example tutorial
sz = 128
data = filaments3D((sz, sz, 1), num_filaments=3)[:, :, 1]
````

define the interpolation function using this package

````@example tutorial
f_affine = get_function_affine(data)
````

define the transformation parameters

````@example tutorial
params = [1.0, 2.0, 1.2, 0.8, 0.0, 0.0, 0.0]
````

apply the transformation

````@example tutorial
data_transformed = f_affine(params)
heatmap(data_transformed, aspect_ratio=1
        , title="Transformed data using a transformation parameters vector"
        , titlefontsize=10, size=(500, 500)
        , xlabel="X", ylabel="Y")


# The next example is to use a transformation matrix
matrix_c = [1.0 0.0 2.0;
            0.0 1.0 3.0;
            0.0 0.0 1.0]

data_transformed_m = f_affine(SMatrix{3, 3}(matrix_c))
heatmap(data_transformed_m, aspect_ratio=1
        , title="Transformed data using a transformation matrix"
        , titlefontsize=10, size=(500, 500)
        , xlabel="X", ylabel="Y")
````

Now we try to do the transformation using a polynomial function
we first define the interpolation object using the DataToFunctions package with a polynomial of order 1

````@example tutorial
f_polynomial = get_function_poly(data, 1)
````

define the polynomial coefficients

````@example tutorial
params = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
````

apply the transformation

````@example tutorial
data_transformed_polynomial = f_polynomial((params))

heatmap(data_transformed_polynomial, aspect_ratio=1
        , title="Transformed data using a polynomial function of order 1"
        , titlefontsize=10, size=(500, 500)
        , xlabel="X", ylabel="Y")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

