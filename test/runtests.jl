using Test
using Zygote

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

    # Zygote.forwarddiff needs only one input
    loss(pz::Vector{Tuple{Float64, Float64}}) = loss(pz[1], pz[2])


    @test loss((0.0,0.0),(1.0,1.0)) < 1e-20
    @test loss((0.0,0.001),(1.0,1.0)) > 1e-20
    @test loss((0.0,0.0),(1.0001,1.0)) > 1e-20

    # to take derivative of a interpolation process, it is better to use the Zygote.forwarddiff
    @test Zygote.forwarddiff(loss, [(0.0, 0.0), (1.0, 1.0)]) < 1e-5
    @test Zygote.forwarddiff(loss, [(0.0, 0.0001), (1.0, 1.0)]) < 1e-5
    @test Zygote.forwarddiff(loss, [(0.0, 0.0), (1.0, 1.0001)]) < 1e-5
    
    # throws an error...
    # Zygote.gradient(loss, (0.0,0.0), (1.0,1.0))
end