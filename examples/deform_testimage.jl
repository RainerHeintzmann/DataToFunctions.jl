### A Pluto.jl notebook ###
# v0.19.43

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

# ╔═╡ a2d75cfb-feab-4130-8439-30c543618d04
using DataToFunctions, ImageShow, TestImages, PlutoUI, Images

# ╔═╡ 4af0c13d-fc42-4fe7-97e6-2248e36b63e2
# Pkg.add("PlutoUI")

# ╔═╡ 5ac1123d-5df3-4c9d-aff1-ffe91d931497
data = testimage("resolution_test_512")

# ╔═╡ b0c8d15e-1bd3-4e66-b2b9-57885636eb48


# ╔═╡ 2fce0208-732f-4259-a47d-7f78921bfd87
f = get_function(data, super_sampling=1)

# ╔═╡ dd535378-3f2a-4429-914c-8b7608b99706
@bind shift_x Slider(-100:0.02:100, default=0)

# ╔═╡ 3e26d4e8-b5d4-4414-8936-379ec63cb4d2
@bind shift_y Slider(-100:0.02:100, default=0)

# ╔═╡ bc3f3659-aa70-4e7f-bef6-4d05229a03c4
@bind zoom_x Slider(0.2:0.02:4, default=1)

# ╔═╡ 39f11dc5-351e-4c6d-8fc4-468222e99976
@bind zoom_y Slider(0.2:0.02:4, default=1)

# ╔═╡ 64b64d3b-092f-4553-916d-f7db1fdfa428
f((shift_x, shift_y), (1/zoom_x, 1/zoom_y))

# ╔═╡ c3fe6b25-f4c0-4a80-9d15-9ee30136d43b
typeof(f((shift_x, shift_y), (1/zoom_x, 1/zoom_y)))

# ╔═╡ ff143f0d-b070-4220-8fa6-0b5a93a56303
typeof(data)

# ╔═╡ 1bed34fb-b29a-4042-a493-4835fdb69a9d
g =DataToFunctions.get_function_affine(data, super_sampling=1)

# ╔═╡ c287fa80-426b-11ef-125e-5fda207e605c
# ╠═╡ disabled = true
#=╠═╡
g()
  ╠═╡ =#

# ╔═╡ 87be45c0-8b2e-4d49-abd1-a274b3c1815e
h = get_function_poly(data, 1)

# ╔═╡ 473d635e-d06d-4cdb-991f-22c82a06b491
@bind c1 Slider(-1f0:0.05f0:1f0, default=0)

# ╔═╡ 7fb3ef17-d0a0-4919-b4d5-8e66b4a0fe60
@bind c2 Slider(-2f0:0.05f0:2f0, default=1)

# ╔═╡ 200bab4d-b444-47c6-b4d8-d5d3c450e5f9
@bind c3 Slider(-2f0:0.05f0:2f0, default=1)

# ╔═╡ f17402a6-ba63-44e9-8e22-da027b07ffc3
@bind c4 Slider(-2f0:0.05f0:2f0, default=1)

# ╔═╡ d913278e-1658-4bee-9e12-ad778e530c1b
@bind c5 Slider(-2f0:0.05f0:2f0, default=1)

# ╔═╡ cee301da-2b3e-432f-9551-0fe83bf6c8ec
@bind c6 Slider(-2f0:0.05f0:2f0, default=1)

# ╔═╡ 74228a9d-6cc2-4aaf-97da-67f32670341e
Gray.(h((c1,c2,c3,c4,c5,c6,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)))

# ╔═╡ 687a198b-9020-4959-8e27-fd0896d4b1fc
maximum(h((c1,c2,c3,c4,c5,c6,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)))

# ╔═╡ Cell order:
# ╠═28975586-853e-4e19-b9eb-65c41fa61a43
# ╠═0ae2da4f-3f75-47bb-a899-9e89c5c3f17c
# ╠═4af0c13d-fc42-4fe7-97e6-2248e36b63e2
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
# ╠═1bed34fb-b29a-4042-a493-4835fdb69a9d
# ╠═c287fa80-426b-11ef-125e-5fda207e605c
# ╠═87be45c0-8b2e-4d49-abd1-a274b3c1815e
# ╠═473d635e-d06d-4cdb-991f-22c82a06b491
# ╠═7fb3ef17-d0a0-4919-b4d5-8e66b4a0fe60
# ╠═200bab4d-b444-47c6-b4d8-d5d3c450e5f9
# ╠═f17402a6-ba63-44e9-8e22-da027b07ffc3
# ╠═d913278e-1658-4bee-9e12-ad778e530c1b
# ╠═cee301da-2b3e-432f-9551-0fe83bf6c8ec
# ╠═74228a9d-6cc2-4aaf-97da-67f32670341e
# ╠═687a198b-9020-4959-8e27-fd0896d4b1fc
