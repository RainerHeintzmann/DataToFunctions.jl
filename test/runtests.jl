using Test
using Zygote
using DataToFunctions

include("Aqua.jl")

include("test_transformators.jl")

# @testset "get_function_affine" begin
#     data = rand(40,41)
#     f = get_function_affine(data);
#     @test f([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]) ≈ data
    
# end

# @testset "loss" begin
#     data = rand(11,10)
#     f = get_function_affine(data; super_sampling=2);
#     loss(p) = sum(abs2.(f(p) .- data))
#     @test loss([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]) < 1e-20
#     @test loss([0.001, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]) > 1e-20
#     @test loss([0.0, 0.0, 1.001, 1.0, 0.0, 0.0, 0.0]) > 1e-20
# end

# @testset "gradient" begin
#     data = rand(11,10)
#     f = get_function_affine(data; super_sampling=2);
#     loss(p) = sum(abs2.(f(p) .- data))
#     st_vals = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
#     # throws an error...
#     @test Zygote.gradient(loss, st_vals)[1] ≈ zeros(7)
# end

# @testset "keep center" begin
#     data = ones(5,4); data[3,3] = 5.0;
#     f = get_function(data);
#     @test f([0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0])[3,3] ≈ 5.0
# end
