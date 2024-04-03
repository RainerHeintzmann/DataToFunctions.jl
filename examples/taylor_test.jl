using TaylorSeries
using StaticArrays

function to_SVec(c::CartesianIndex)
    return [Tuple(c)...]
end

function main()
    t = set_variables("x", numvars=3, order=4)
    p = exp.(t)
    @time q = evaluate.(Ref(p), to_SVec.(CartesianIndices(img)));
    @time q .= evaluate.(Ref(p), to_SVec.(CartesianIndices(img)));
end
