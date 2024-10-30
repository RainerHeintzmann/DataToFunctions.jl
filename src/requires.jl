# from https://github.com/RainerHeintzmann/DeconvOptim.jl/blob/362a741224957155fbc046d3297d43339766721d/src/requires.jl

isgpu(x) = false
gpu_or_cpu(x) = nothing

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        @info "DataToFunctions.jl: CUDA.jl is loaded."
        
        gpu_or_cpu(x) = CUDA.CuArray{Float32}
        isgpu(x::CUDA.CuArray) = true
        # prevent slow scalar indexing on GPU
        CUDA.allowscalar(false);

    end
end