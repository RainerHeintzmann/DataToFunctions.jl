export get_polynomial, get_multi_poly, get_num_poly_vars, get_num_multipoly_vars
export polynomial

# using Unrolled

"""
    polynomial(::Val{0}, ::T1,  c::T2, ::Val{cstart}=Val(1), ::Val{numvars}=Val(1)) where {NV, TS, T1 <: NTuple{NV, Integer}, T2 <: NTuple{TS, Float32}, cstart, numvars}

Create a polynomial of order 0 with numvars variables.

returned is a function that takes a tuple of variables and a tuple of coefficients and returns the value of the polynomial
    and the number of coefficients required (here 1).
"""
function polynomial(::Val{0}, ::T1,  c::T2, ::Val{cstart}=Val(1), ::Val{numvars}=Val(1)) where {NV, TS, T1 <: NTuple{NV, Integer}, T2 <: NTuple{TS, Float32}, cstart, numvars}
    # println("c: $(c) $(length(c))");
    return c[cstart]
end  #, (t,c) -> ntuple(n->c[1], Val(numvars))

"""
    polynomial(::Val{N}, t::T1, c::T2, ::Val{cstart}=Val(1), ::Val{numvars}=Val(length(t)))::Float32 where {N, NV, TS, T1 <: NTuple{NV, Integer}, T2 <: NTuple{TS, Float32}, cstart, numvars}

Represents a polynomial of order N with numvars variables (also implicitely defined via the length of the NTuple `t`).
Note that `numvars` is needed for the internal workings of the polynomial generator, but notmally not by the user.

E.g. to represent a polynomial of order 1 with 2 variables, the coefficients are ordered as follows:
c = (c0, c1, c2) where the polynomial is: c0 + c1*x + c2*y
or for a polynomial of order 2 with 2 variables:
c = (c0, c1, c2, c3, c4, c5) where the polynomial is c0 + c1*x + c2*x^2 + c3*y + c4*x*y + c5*y^2
Note that the coefficients are ordered not by the multiples in which they appear in the polynomial, but by the order of the variables.

Example:
```jldoctest
>julia polynomial(Val(2), (2, 20),  (1f0, 1f0, 1f0, 1f0, 1f0, 1f0))
467.0f0
>julia cs = Tuple(Float32.(collect(1:27)))
  (1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0, 7.0f0, 8.0f0, 9.0f0, 10.0f0, 11.0f0, 12.0f0, 13.0f0, 14.0f0, 15.0f0, 16.0f0, 17.0f0, 18.0f0, 19.0f0, 20.0f0, 21.0f0, 22.0f0, 23.0f0, 24.0f0, 25.0f0, 26.0f0, 27.0f0)
>julia res = zeros(Float32, 200,200)
>julia @time res .= polynomial.(Ref(Val(2)), Tuple.(CartesianIndices((200,200))), Ref(cs)); 
  0.054017 seconds (147.45 k allocations: 10.168 MiB, 99.75% compilation time)
>julia @time res .= polynomial.(Ref(Val(2)), Tuple.(CartesianIndices((200,200))), Ref(cs)); 
  0.000089 seconds (3 allocations: 184 bytes)
```
"""
function polynomial(::Val{N}, t::T1, c::T2, ::Val{cstart}=Val(1), ::Val{numvars}=Val(length(t)))::Float32 where {N, NV, TS, T1 <: NTuple{NV, Integer}, T2 <: NTuple{TS, Float32}, cstart, numvars} 
    c_start = cstart
    res = c[c_start]
    c_start += 1
    # iterate through the polynomial variables: (but this leads to dynamic memory allocation!)
    # for n = 1:numvars # eachindex(t) # 1:length(t) # eachindex(t) # 1:length(t)
    #     # c_end = c_start + get_num_poly_vars(Val(n), Val(N-1)) - 1
    #     res += t[n] * polynomial(Val(N-1), t, c, Val(c_start), Val(n)) 
    #     c_start += get_num_poly_vars(Val(n), Val(N-1)) # c_end + 1
    # end
    # # does not work:
    # @macroexpand Base.Cartesian.@nexprs 4 n -> begin
    #     if (numvars >= n)
    #     res += t[n] * polynomial(Val(N-1), t, c, Val(c_start), Val(n)) 
    #     c_start += get_num_poly_vars(Val(n), Val(N-1)) # c_end + 1
    #     end
    # end

    # this simply unrolls the loop by hand (up to 4D input variables):
    if numvars >= 1
        res += t[1] * polynomial(Val(N-1), t, c, Val(c_start), Val(1)) 
        c_start += get_num_poly_vars(Val(1), Val(N-1)) # c_end + 1
    end
    if numvars >= 2
        res += t[2] * polynomial(Val(N-1), t, c, Val(c_start), Val(2)) 
        c_start += get_num_poly_vars(Val(2), Val(N-1)) # c_end + 1
    end
    if numvars >= 3
        res += t[3] * polynomial(Val(N-1), t, c, Val(c_start), Val(3)) 
        c_start += get_num_poly_vars(Val(3), Val(N-1)) # c_end + 1
    end
    if numvars >= 4
        res += t[4] * polynomial(Val(N-1), t, c, Val(c_start), Val(4)) 
        c_start += get_num_poly_vars(Val(4), Val(N-1)) # c_end + 1
    end
    if numvars >= 5
        error("Only up to 4 dimensions are currently supported for polynomials")
    end
    return res
end


# function unroll_loop(res::Float32, ::Var{0}, ::Var{numvars}) where{n}
#     return 0f0;
# end

# function unroll_loop(res::Float32, t, ::Var{c_start},::Var({n}, ::Var{numvars}) where{c_start, n, numvars}
#     res += t[n] * polynomial(Val(N-1), t, c, Val(c_start), Val(n)) 
#     c_start += get_num_poly_vars(Val(n), Val(N-1)) # c_end + 1
#     return unroll_loop(res::Float32, ::Var({n}, ::Var{numvars}) + 
# end


function get_multi_poly(::Val{numvars}, ::Val{N}) where {numvars, N}
    # cs_per_comp = ((numvars+1)^N)
    @info "Creating polynomials with $(numvars) variables of order , $(N). Required constants: $(numvars*((numvars+1)^N))"
    p = (t,c) -> polynomial(Val(N), Tuple.(t), c)
    # return p
    function mpol(t,c)#::NTuple{numvars, T} where T
        return ntuple(n->p(t, split_tuple(c, Val(numvars))[n]), Val(numvars))

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
    # get_num_poly_vars(Val(2), Val(3)) # 10 indices
    # p = get_polynomial(Val(2), Val(3))  
    # @time p.(Tuple.(CartesianIndices((200,200))),Ref((1.1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27)));

    polynomial(Val(2), (2f0, 20f0),  (1f0, 1f0, 1f0, 1f0, 1f0, 1f0)) == 467

    cids = Tuple.(CartesianIndices((200,200)))
    # cfds = map((t)->Tuple(Float32.([t...])), cids) 
    cs = Tuple(Float32.(collect(1:27)))
    res = zeros(Float32, 200,200)
    get_num_poly_vars(Val(2), Val(2)) # 6 indices required
    @time res .= polynomial.(Ref(Val(2)), cids, Ref(cs)); # 2 orders, two variables
    # 0.000122 seconds (3 allocations: 168 bytes)

    get_num_poly_vars(Val(3), Val(2)) # 10 indices required
    @time res .= polynomial.(Ref(Val(3)), cids, Ref(cs)); # 2 orders, two variables
    # 0.003710 seconds (240.00 k allocations: 13.428 MiB)

    get_num_poly_vars(Val(4), Val(2)) # 15 indices required
    @time res .= polynomial.(Ref(Val(4)), cids, Ref(cs)); # 2 orders, two variables
    # 0.008149 seconds (480.00 k allocations: 26.856 MiB)

    get_num_poly_vars(Val(5), Val(2)) # 21 indices required
    @time res .= polynomial.(Ref(Val(5)), cids, Ref(cs)); # 2 orders, two variables
    #0.014299 seconds (1.08 M allocations: 60.425 MiB, 23.18% gc time)

    @time polynomial.(Ref(Val(0)), cids, Ref(cs));
    # 0.000076 seconds (5 allocations: 156.461 KiB)

end
