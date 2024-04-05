export split_tuple

"""
    split_tuple(t::NTuple{S,T},::Val{numvars}) where {S,T,numvars}

Split a tuple into `numvars` parts packed into a tuple of tuples. The tuple `t` is assumed to have a length that is a multiple of `numvars`.

Example:
```juliadoc
julia> t = (1,2,3,4,5,6)
julia> split_tuple(t, Val{2}) 
((1,2), (3,4), (5,6))
```
"""
function split_tuple(t::NTuple{S,T},::Val{numvars}) where {S,T,numvars}
    return ntuple(n->t[1+(n-1)*(S÷numvars):n*(S÷numvars)], Val(numvars))
end

