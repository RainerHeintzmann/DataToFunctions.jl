import Base: getindex, setindex!, size, axes, eltype, copy, similar, ndims, iterate, length


struct DataFunctionAffine{affineparams{NTuple{N, Number}}, itp, S<:AbstractArray} <: AbstractArray
    params::affineparams
    interpolation::itp
    data::S
end

function DataFunctionAffine(params::affineparams, interpolation::itp) where {affineparams<:NTuple{N, Number}, itp}
    return DataFunctionAffine{affineparams, itp}(params, interpolation)
end

function getindex(dfa::DataFunctionAffine, I::Vararg{Number, N}) where {N}
    return dfa.interpolation(Tuple(I)...)
end

function copy(s::DataFunctionAffine)
    res = similar(s)
    res .= s
end


function similar(dfa::DataFunctionAffine, ::Type{T}=eltype(dfa)) where {T}
    return DataFunctionAffine(dfa.params, similar(dfa.interpolation, T))
end

size(dfa::DataFunctionAffine) = size(dfa.data)
axes(dfa::DataFunctionAffine) = axes(dfa.data)
eltype(dfa::DataFunctionAffine) = eltype(dfa.data)
ndims(dfa::DataFunctionAffine) = ndims(dfa.data)
length(dfa::DataFunctionAffine) = length(dfa.data)


