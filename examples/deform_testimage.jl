### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 28975586-853e-4e19-b9eb-65c41fa61a43
using Pkg

# ╔═╡ 0ae2da4f-3f75-47bb-a899-9e89c5c3f17c
Pkg.activate(".")

# ╔═╡ 4af0c13d-fc42-4fe7-97e6-2248e36b63e2
Pkg.add("PlutoUI")

# ╔═╡ a2d75cfb-feab-4130-8439-30c543618d04
using DataToFunctions, ImageShow, TestImages, PlutoUI, Images

# ╔═╡ 1c744bec-f085-4812-ab1a-40a32c2ac176
import PlutoUI: combine

# ╔═╡ 5ac1123d-5df3-4c9d-aff1-ffe91d931497
data = Float32.(testimage("resolution_test_512"))

# ╔═╡ b0c8d15e-1bd3-4e66-b2b9-57885636eb48
simshow(data)

# ╔═╡ 2fce0208-732f-4259-a47d-7f78921bfd87
f = get_interpolated_function(data, AffineMode, super_sampling=1)

# ╔═╡ dd535378-3f2a-4429-914c-8b7608b99706
@bind shift_x Slider(-100:0.02:100, default=0)

# ╔═╡ 3e26d4e8-b5d4-4414-8936-379ec63cb4d2
@bind shift_y Slider(-100:0.02:100, default=0)

# ╔═╡ bc3f3659-aa70-4e7f-bef6-4d05229a03c4
@bind zoom_x Slider(0.2:0.02:4, default=1)

# ╔═╡ 39f11dc5-351e-4c6d-8fc4-468222e99976
@bind zoom_y Slider(0.2:0.02:4, default=1)

# ╔═╡ 64b64d3b-092f-4553-916d-f7db1fdfa428
simshow(f((shift_x, shift_y, 1/zoom_x, 1/zoom_y, 0.0, 0.0, 0.0)))

# ╔═╡ c3fe6b25-f4c0-4a80-9d15-9ee30136d43b
typeof(f((shift_x, shift_y, 1/zoom_x, 1/zoom_y, 0.0, 0.0, 0.0)))

# ╔═╡ ff143f0d-b070-4220-8fa6-0b5a93a56303
typeof(data)

# ╔═╡ 6676ba35-3efd-49f5-9819-411ad8f8a95c
md"""
# Polynomial transformations
"""

# ╔═╡ 13461a95-95ea-4bad-8673-e94e06776254
md"""
## First order polynomial

First order polynomial transformation which is as follows:

``x^{\prime} = c_{1} + c_{2}{x}^{1} + c_{3}{y}^{1}``

``y^{\prime} = c_{4} + c_{5}{x}^{1} + c_{6}{y}^{1}``
"""

# ╔═╡ 87be45c0-8b2e-4d49-abd1-a274b3c1815e
h = get_interpolated_function(data, PolynomialMode, 1);

# ╔═╡ 68f771aa-2cde-41cd-990c-9ec7dc2146a4
function coeffs_input(coeffs::Vector)
	
	return combine() do Child
		
		inputs = [
			md""" $(name): $(
				Child(name, Slider(-2f0:0.05f0:2f0, default=0, show_value=true))
			)"""
			
			for name in coeffs
		]
		
		md"""
		#### Transform coefficients
		$(inputs)
		"""
	end
end;

# ╔═╡ 69331d73-75a3-4727-acda-e79779a2bd03
@bind c coeffs_input(["c1", "c2", "c3", "c4", "c5", "c6"])

# ╔═╡ 74228a9d-6cc2-4aaf-97da-67f32670341e
simshow(h((c.c1, c.c2, c.c3, c.c4, c.c5, c.c6)), cmap=:turbo)#,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)))

# ╔═╡ 687a198b-9020-4959-8e27-fd0896d4b1fc
maximum(h((c.c1,c.c2,c.c3,c.c4,c.c5,c.c6)))

# ╔═╡ 6584aacc-440e-457b-bf52-83f8db40c999
h((c.c1,c.c2,c.c3,c.c4,c.c5,c.c6))

# ╔═╡ 74cf9f27-f559-4fe2-9a30-20cb7cf6fe81
md"""
## Second order polynomial

Second order polynomial transformation is as follows:

``x^{\prime} = c_{1} + c_{2}{x}^{1} + c_{3}{x}^{2} + c_{4}{y}^{1} + c_{5}{x}{y} + c_{6}{y}^{2}``

``y^{\prime} = c_{7} + c_{8}{x}^{1} + c_{9}{x}^{2} + c_{10}{y}^{1} + c_{11}{x}{y} + c_{12}{y}^{2}``
"""

# ╔═╡ 1f440592-9bbb-429c-8ba3-d018f5b354b0
h2 = get_interpolated_function(data, PolynomialMode, 2);

# ╔═╡ 12f59a6d-4600-4379-b536-f3b4701bdbe4
@bind c2 coeffs_input(["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11", "c12"])

# ╔═╡ fd60df55-2a80-48a9-95ec-eeeb1cfe7491
simshow(h2((c2.c1, c2.c2, c2.c3, c2.c4, c2.c5, c2.c6, c2.c7, c2.c8, c2.c9, c2.c10, c2.c11, c2.c12)), cmap=:turbo)#,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)))

# ╔═╡ Cell order:
# ╠═28975586-853e-4e19-b9eb-65c41fa61a43
# ╠═0ae2da4f-3f75-47bb-a899-9e89c5c3f17c
# ╟─4af0c13d-fc42-4fe7-97e6-2248e36b63e2
# ╠═1c744bec-f085-4812-ab1a-40a32c2ac176
# ╠═a2d75cfb-feab-4130-8439-30c543618d04
# ╠═5ac1123d-5df3-4c9d-aff1-ffe91d931497
# ╠═b0c8d15e-1bd3-4e66-b2b9-57885636eb48
# ╠═2fce0208-732f-4259-a47d-7f78921bfd87
# ╠═dd535378-3f2a-4429-914c-8b7608b99706
# ╠═3e26d4e8-b5d4-4414-8936-379ec63cb4d2
# ╠═bc3f3659-aa70-4e7f-bef6-4d05229a03c4
# ╠═39f11dc5-351e-4c6d-8fc4-468222e99976
# ╠═64b64d3b-092f-4553-916d-f7db1fdfa428
# ╠═c3fe6b25-f4c0-4a80-9d15-9ee30136d43b
# ╠═ff143f0d-b070-4220-8fa6-0b5a93a56303
# ╟─6676ba35-3efd-49f5-9819-411ad8f8a95c
# ╟─13461a95-95ea-4bad-8673-e94e06776254
# ╠═87be45c0-8b2e-4d49-abd1-a274b3c1815e
# ╠═69331d73-75a3-4727-acda-e79779a2bd03
# ╠═68f771aa-2cde-41cd-990c-9ec7dc2146a4
# ╠═74228a9d-6cc2-4aaf-97da-67f32670341e
# ╠═687a198b-9020-4959-8e27-fd0896d4b1fc
# ╠═6584aacc-440e-457b-bf52-83f8db40c999
# ╟─74cf9f27-f559-4fe2-9a30-20cb7cf6fe81
# ╠═1f440592-9bbb-429c-8ba3-d018f5b354b0
# ╠═12f59a6d-4600-4379-b536-f3b4701bdbe4
# ╠═fd60df55-2a80-48a9-95ec-eeeb1cfe7491
