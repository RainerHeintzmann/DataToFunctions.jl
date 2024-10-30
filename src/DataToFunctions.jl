module DataToFunctions

using Requires

export gpu_or_cpu

# to include CUDA
include("requires.jl")

include("transformation_types.jl")
include("transformators.jl")
 
end # module DataToFunctions