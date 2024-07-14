export get_polynomial, get_multi_poly, get_num_poly_vars, get_num_multipoly_vars
export polynomial

using Unrolled

"""
    get_polynomial(::Val{numvars}, ::Val{0}) where {numvars, N}

Create a polynomial of order 0 with numvars variables.

returned is a function that takes a tuple of variables and a tuple of coefficients and returns the value of the polynomial
    and the number of coefficients required (here 1).
"""
# function get_polynomial(::Val{numvars}, ::Val{0}) where {numvars}
#     # @info "Creating polynomials of order 0"
function polynomial(::Val{0}, ::T1,  c::T2) where {T1 <: NTuple, T2 <: NTuple}
    # println("c: $(c) $(length(c))");
    return c[1] 
end  #, (t,c) -> ntuple(n->c[1], Val(numvars))
#     return myconst
# end

"""
    get_polynomial(::Val{numvars}, ::Val{N}) where {numvars, N}

Create a polynomial of order N with numvars variables.
returned is a function that takes a tuple of variables and a tuple of coefficients and returns the value of the polynomial.

E.g. to represent a polynomial of order 1 with 2 variables, the coefficients are ordered as follows:
c = (c0, c1, c2) where the polynomial is: c0 + c1*x + c2*y
or for a polynomial of order 2 with 2 variables:
c = (c0, c1, c2, c3, c4, c5) where the polynomial is c0 + c1*x + c2*x^2 + c3*y + c4*x*y + c5*y^2
Note that the coefficients are ordered not by the multiples in which they appear in the polynomial, but by the order of the variables.
"""
# function get_polynomial(::Val{numvars}, ::Val{N}) where {numvars, N}
#     # @info "Creating polynomials with $(numvars) variables of order , $(N). Required constants: $((numvars+1)^N)"

function polynomial(::Val{N}, t::T1, c::T2)::Float32 where {N, T1<:NTuple, T2<:NTuple} # :: NTuple{NV, Float32}, NTuple{M, Float32}
    c_start = 1
    res = c[c_start]
    c_start += 1
    # iterate through the polynomial variables
    for n in eachindex(t) # 1:length(t) # eachindex(t) # 1:length(t)
        # subpoly = get_polynomial(Val(n), Val(N-1)); # calulate polynomial with only n variable
        c_end = c_start + get_num_poly_vars(Val(n), Val(N-1)) - 1
        # @show N
        # @show n
        # @show c_start
        # @show t
        # @show c
        res += t[n] * polynomial(Val(N-1), t[1:n], c[c_start:c_end]) 
        c_start = c_end + 1
    end
    return res
end
#     return mypoly
# end


function get_multi_poly(::Val{numvars}, ::Val{N}) where {numvars, N}
    # cs_per_comp = ((numvars+1)^N)
    @info "Creating polynomials with $(numvars) variables of order , $(N). Required constants: $(numvars*((numvars+1)^N))"
    p = get_polynomial(Val(numvars), Val(N))
    # return p
    function mpol(t,c)#::NTuple{numvars, T} where T
        return ntuple(n->p(t, split_tuple(c,Val(numvars))[n]), Val(numvars))

        # println("N: $(N), c: $(c) $(length(c))");
        # return Tuple(p(t, c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]) for n=1:numvars)
        # return ntuple(n->p(t, c[1+(n-1)*cs_per_comp:n*cs_per_comp]), Val(numvars))
        # return ntuple(n->p(t, c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]), Val(numvars))
        # return (p(t, c[1+(1-1)*((numvars+1)^N):1*((numvars+1)^N)]), p(t, c[1+(2-1)*((numvars+1)^N):2*((numvars+1)^N)]))
    end
    return mpol # (t, c)->ntuple(n->p(t, c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]), Val(numvars))
    # return (t, c) -> Tuple(p(t,c[1+(n-1)*((numvars+1)^N):n*((numvars+1)^N)]) for n=1:numvars)
end

function get_num_poly_vars(::Val{numvars}, ::Val{N}) where {numvars, N}
    return binomial(numvars+N, N) 
end

function get_num_multipoly_vars(::Val{numvars}, ::Val{N}) where {numvars, N}
    return numvars*get_num_poly_vars(Val(numvars), Val(N))
end

function test_poly_allocations()
    get_num_poly_vars(Val(2), Val(3)) # 10 indices
    p = get_polynomial(Val(2), Val(3))  
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)));
    cids = Tuple.(CartesianIndices((200,200)))
    cs = Tuple(Float32.([1.1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]))
    @time polynomial.(Ref(Val(3)), cids, Ref(cs));
    # does allocate 95 Mb !

    p = get_polynomial(Val(3), Val(2))  # 10 indices required
    get_num_poly_vars(Val(3), Val(2))
    @time p.(Tuple.(CartesianIndices((100,100,10))),Ref((1.1,2.1,3.1,4,5,6,7,8,9,10,11,12,13,14,15,16)));
    # does allocate 256 Mb !

    p = get_polynomial(Val(2), Val(2))  # 9 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2.1,3.1,4,5,6,7,8,9)));
    # essentially allocation-free

    p = get_polynomial(Val(2), Val(1))  # 3 indices required
    @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2.1,3.1)));
    # essentially allocation-free
    # p((100,100),((1.1,2.2, 3.3)))

    p = get_polynomial(Val(1), Val(0))  # 27 indices required
    @time p.(Tuple.(CartesianIndices((100,100))),Ref((1.1)));
    # essentially allocation-free

end
