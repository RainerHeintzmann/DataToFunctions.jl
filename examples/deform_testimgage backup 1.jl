### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 28975586-853e-4e19-b9eb-65c41fa61a43
using Pkg

# ╔═╡ a2d75cfb-feab-4130-8439-30c543618d04
using DataToFunctions, ImageShow, TestImages

# ╔═╡ 0ae2da4f-3f75-47bb-a899-9e89c5c3f17c
# Pkg.activate(".")

# ╔═╡ 4af0c13d-fc42-4fe7-97e6-2248e36b63e2
# Pkg.add("ImageShow")

# ╔═╡ 5ac1123d-5df3-4c9d-aff1-ffe91d931497
data = testimage("resolution_test_512", super_sampling=1)

# ╔═╡ b0c8d15e-1bd3-4e66-b2b9-57885636eb48


# ╔═╡ 2fce0208-732f-4259-a47d-7f78921bfd87
f = get_function(data)

# ╔═╡ 1bed34fb-b29a-4042-a493-4835fdb69a9d


# ╔═╡ c287fa80-426b-11ef-125e-5fda207e605c
# ╠═╡ disabled = true
#=╠═╡
using DataToFunctions
  ╠═╡ =#

# ╔═╡ 87be45c0-8b2e-4d49-abd1-a274b3c1815e


# ╔═╡ Cell order:
# ╠═28975586-853e-4e19-b9eb-65c41fa61a43
# ╠═0ae2da4f-3f75-47bb-a899-9e89c5c3f17c
# ╠═4af0c13d-fc42-4fe7-97e6-2248e36b63e2
# ╠═a2d75cfb-feab-4130-8439-30c543618d04
# ╠═5ac1123d-5df3-4c9d-aff1-ffe91d931497
# ╠═b0c8d15e-1bd3-4e66-b2b9-57885636eb48
# ╠═2fce0208-732f-4259-a47d-7f78921bfd87
# ╠═1bed34fb-b29a-4042-a493-4835fdb69a9d
# ╠═c287fa80-426b-11ef-125e-5fda207e605c
# ╠═87be45c0-8b2e-4d49-abd1-a274b3c1815e
