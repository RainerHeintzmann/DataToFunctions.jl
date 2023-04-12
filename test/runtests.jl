using Test
# using Zygote
using DataToFunctions

@testset "get_function" begin
    data = rand(40,41)
    for supersamp = 1:5
        f = get_function(data; super_sampling=supersamp);
        @test f((0.0,0.0),(1.0,1.0)) ≈ data
    end
end

@testset "gradient" begin
    data = rand(11,10)
    f = get_function(data; super_sampling=2);
    loss(p,z) = sum(abs2.(f(p, z) .- data))
    @test loss((0.0,0.0),(1.0,1.0)) < 1e-20
    @test loss((0.0,0.001),(1.0,1.0)) > 1e-20
    @test loss((0.0,0.0),(1.0001,1.0)) > 1e-20

    # throws an error...
    # Zygote.gradient(loss, (0.0,0.0), (1.0,1.0))
end

@testset "keep center" begin
    data = ones(5,4); data[3,3] = 5.0;
    f = get_function(data; super_sampling=5);
    @test f((0.0,0.0),(2.0,2.0))[3,3] ≈ 5.0

    # throws an error...
    # Zygote.gradient(loss, (0.0,0.0), (1.0,1.0))
end
